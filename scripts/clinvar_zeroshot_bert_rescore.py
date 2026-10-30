"""Score BERT's zero-shot ClinVar, with the input form and the scored positions
both chosen explicitly.

Two axes, set independently, because the campaign fixed one of each without the
pair ever being written down together.

**Input form.** `wrapped` sends [CLS] + 1,024 + [SEP], which is what training
used and what the linear probe builds. `bare` sends the 1,024 base tokens the
evaluation adapter sends today -- this tokeniser has the special tokens in its
vocabulary but no post-processor template, so add_special_tokens=True adds
nothing.

**Scored positions.** `variant` masks the variant position alone, which the
2026-09-07 verdict asked for and which every BERT zero-shot number so far uses.
`window` masks every base position in turn and averages, which is the quantity
GPT-2's whole-window log-likelihood is, and what ProteinGym scores.

The score is the same in all four combinations: the reference sequence's value
minus the variant sequence's, so a higher score means the reference was the
likelier sequence.

`--max-variants` scores a prefix instead of the whole table. It is for timing:
the window mode is 1,024 times the work of the variant mode, so its cost is
measured before it is paid.

Every run writes its per-variant scores beside the aggregate, as
``<out stem>-predictions.jsonl``. This is not optional. Keeping only the
aggregate AUROCs once cost a second pass over 21 runs -- about 90 GPU-hours --
because a bootstrap interval cannot be resampled from a mean. `--no-predictions`
exists for the timing probes, which score a prefix and whose rows are not a
measurement of anything.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

FOLDS = ("21", "22", "X")


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        shared = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    n_pos = sum(1 for _s, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = sum(r for r, (_s, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_rows(path, chroms, limit=None):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    out = []
    for r in csv.DictReader(open(path, newline="")):
        c = (r.get("chrom") or "").replace("chr", "").strip()
        if c not in chroms:
            continue
        sig = (r.get("ClinicalSignificance") or "").lower()
        if "conflicting" in sig:
            continue
        p, b = "pathogenic" in sig, "benign" in sig
        if p == b:
            continue
        out.append((c, r["reference_sequence"], r["variant_sequence"], 1 if p else 0))
        if limit and len(out) >= limit:
            break
    return out


def adopted_checkpoint(run_dir):
    """The checkpoint the run adopted, from the newest trainer_state.

    Returns (path, step). The run moved between checkouts after it finished, so
    ``best_model_checkpoint`` holds an absolute path that may no longer exist;
    the basename still names the step, and that directory is looked for under
    ``run_dir``.

    When the field is missing, or names a directory that is not there, this
    stops. It used to fall back to the largest step, which is the final
    checkpoint -- a value scored from it would be reported as the adopted one
    with nothing in the output saying otherwise.
    """
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(f"{run_dir}/checkpoint-*")
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        raise SystemExit(f"{run_dir} carries no checkpoint-<step> directory")
    state = json.load(open(f"{run_dir}/checkpoint-{max(steps)}/trainer_state.json"))
    best = state.get("best_model_checkpoint")
    if not best:
        raise SystemExit(
            f"{run_dir}/checkpoint-{max(steps)}/trainer_state.json has no "
            f"best_model_checkpoint; the adopted checkpoint cannot be resolved "
            f"and the final step ({max(steps)}) is not a substitute for it")
    name = os.path.basename(best.rstrip("/"))
    for candidate in (best, os.path.join(run_dir, name)):
        if os.path.isdir(candidate):
            return candidate, int(name.rsplit("-", 1)[1])
    raise SystemExit(
        f"{run_dir}: best_model_checkpoint names {best}, which is not a "
        f"directory here either as given or as {run_dir}/{name}")


def sequence_logprob(model, torch, F, ids_row, positions, mask_id, pos_batch):
    """Mean log prob of the true token over `positions`, each masked alone."""
    total = 0.0
    for start in range(0, len(positions), pos_batch):
        chunk = positions[start:start + pos_batch]
        batch = ids_row.unsqueeze(0).repeat(len(chunk), 1).clone()
        truth = ids_row[chunk]
        for row, pos in enumerate(chunk):
            batch[row, pos] = mask_id
        logits = model(input_ids=batch,
                       attention_mask=torch.ones_like(batch)).logits
        lp = F.log_softmax(logits[range(len(chunk)), chunk].float(), dim=-1)
        total += float(lp.gather(1, truth.unsqueeze(1)).sum())
    return total / len(positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input-form", choices=("bare", "wrapped"), required=True)
    ap.add_argument("--positions", choices=("variant", "window"), required=True)
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--variant-batch", type=int, default=64,
                    help="variant mode: sequences per forward pass")
    ap.add_argument("--position-batch", type=int, default=512,
                    help="window mode: masked copies per forward pass")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-predictions", action="store_true",
                    help="timing probes only: skip the per-variant scores")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rows = read_rows(args.csv, set(args.chroms.split(",")), args.max_variants)
    labels = [y for _c, _r, _v, y in rows]
    print(f"  {len(rows):,} 変異（病原性 {sum(labels):,} / 良性 {len(labels) - sum(labels):,}）")

    ckpt, ckpt_step = adopted_checkpoint(args.run_dir)
    print(f"  checkpoint: {ckpt}  (step {ckpt_step:,})")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(ckpt, torch_dtype=dtype)
    model.to(device).eval()

    wrapped = args.input_form == "wrapped"
    window = len(rows[0][1])
    offset = 1 if wrapped else 0
    cls_id, sep_id, mask_id = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id

    def encode(seq):
        ids = tok.convert_tokens_to_ids(list(seq))
        return ([cls_id] + ids + [sep_id]) if wrapped else ids

    t0 = time.time()
    scores = []
    if args.positions == "variant":
        at = window // 2 + offset
        with torch.no_grad():
            for lo in range(0, len(rows), args.variant_batch):
                chunk = rows[lo:lo + args.variant_batch]
                got = []
                for which in (1, 2):
                    ids = torch.tensor([encode(r[which]) for r in chunk],
                                       dtype=torch.long, device=device)
                    truth = ids[:, at].clone()
                    ids[:, at] = mask_id
                    logits = model(input_ids=ids,
                                   attention_mask=torch.ones_like(ids)).logits
                    lp = F.log_softmax(logits[:, at].float(), dim=-1)
                    got.append(lp.gather(1, truth.unsqueeze(1)).squeeze(1))
                scores.extend((got[0] - got[1]).tolist())
                if lo % (args.variant_batch * 50) == 0:
                    print(f"    {lo:,} / {len(rows):,}  {time.time() - t0:.0f}s", flush=True)
    else:
        positions = list(range(offset, offset + window))
        with torch.no_grad():
            for i, (_c, ref, var, _y) in enumerate(rows):
                vals = []
                for seq in (ref, var):
                    ids_row = torch.tensor(encode(seq), dtype=torch.long, device=device)
                    vals.append(sequence_logprob(model, torch, F, ids_row, positions,
                                                 mask_id, args.position_batch))
                scores.append(vals[0] - vals[1])
                if i and i % 10 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"    {i + 1:,} / {len(rows):,}  {rate:.2f} 変異/秒  "
                          f"全 21,350 件なら {21350 / rate / 3600:.1f} 時間", flush=True)
    elapsed = time.time() - t0

    out = {"run_dir": args.run_dir, "checkpoint": ckpt,
           "checkpoint_step": ckpt_step,
           "input_form": args.input_form, "positions": args.positions,
           "bf16": args.bf16, "n_variants": len(rows), "seconds": round(elapsed, 1),
           "variants_per_second": round(len(rows) / elapsed, 3),
           "auroc_all": auroc(labels, scores)}
    three = [(y, s) for (c, _r, _v, y), s in zip(rows, scores) if c in FOLDS]
    if three:
        out["n_three"] = len(three)
        out["auroc_three"] = auroc([y for y, _ in three], [s for _, s in three])
    for fold in FOLDS:
        got = [(y, s) for (c, _r, _v, y), s in zip(rows, scores) if c == fold]
        if got:
            out[f"n_chr{fold}"] = len(got)
            out[f"auroc_chr{fold}"] = auroc([y for y, _ in got], [s for _, s in got])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The per-variant scores go out first, so a crash between the two files
    # leaves the rows and not just the mean.
    if not args.no_predictions:
        stem = args.out[:-len(".json")] if args.out.endswith(".json") else args.out
        preds = f"{stem}-predictions.jsonl"
        with open(preds, "w") as handle:
            for (chrom, _ref, _var, label), score in zip(rows, scores):
                handle.write(json.dumps(
                    {"chrom": chrom, "fold": f"chr{chrom}",
                     "label_pathogenic": label, "score": score}) + "\n")
        out["predictions"] = os.path.basename(preds)
        print(f"  wrote {preds}  ({len(scores):,} 行)")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  {args.input_form}/{args.positions}: AUROC {out['auroc_all']:.6f}  "
          f"{elapsed:.0f}s  {out['variants_per_second']:.3f} 変異/秒")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
