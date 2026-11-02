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
``<out stem>-predictions.jsonl``. There is no flag for leaving them out.
Keeping only the aggregate AUROCs cost one pass over 21 runs -- about 90
GPU-hours -- because a bootstrap interval cannot be resampled from a mean, and
the pass that fixed that cost another, because its rows carried the score
alone: nothing to join a row back to a variant, and no way to change what the
score is summed over.

Each row now carries the variant's identity -- the VCV id, the chromosome, the
position, both alleles, the ClinVar call -- and, for every range below, the
reference sequence's value, the variant sequence's value, and their difference.

In window mode the ranges are the whole 1,024 positions, the 257 within 128
bases of the variant, the 65 within 32, and the variant position alone. All
four come out of one pass: every position is masked and scored anyway, so
narrowing the range only changes which of those values are added up, and the
GPU cost is identical. Whether the range matters is then a question the file
can answer -- a position far from the variant moves with the sequence rather
than with the substitution, and summing 1,024 of them may bury the one
position that differs.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

FOLDS = ("21", "22", "X")

# How far from the variant a range reaches, in bases. None is the whole window,
# 0 the variant position alone. 128 and 32 give 257 and 65 positions.
RANGES = (("window", None), ("flank128", 128), ("flank32", 32), ("point", 0))


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
    """The variants to score, each with what identifies it.

    A score on its own cannot be joined back to a variant, so the VCV id, the
    position, both alleles and the ClinVar call travel with the row and are
    written out beside the values.

    A variant whose ClinVar call conflicts, or which is neither pathogenic nor
    benign, is left out here rather than later, so every count downstream is
    the count of rows actually scored.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    out = []
    for r in csv.DictReader(open(path, newline="")):
        c = (r.get("chrom") or "").replace("chr", "").strip()
        if c not in chroms:
            continue
        sig = (r.get("ClinicalSignificance") or "").strip()
        low = sig.lower()
        if "conflicting" in low:
            continue
        p, b = "pathogenic" in low, "benign" in low
        if p == b:
            continue
        pos = (r.get("pos") or "").strip()
        out.append({"vcv_id": r.get("vcv_id"), "chrom": c,
                    "pos": int(pos) if pos.isdigit() else pos,
                    "ref_allele": r.get("ref"), "alt_allele": r.get("alt"),
                    "clinical_significance": sig,
                    "label_pathogenic": 1 if p else 0,
                    "reference_sequence": r["reference_sequence"],
                    "variant_sequence": r["variant_sequence"]})
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


def position_logprobs(model, torch, F, ids_row, positions, mask_id, pos_batch):
    """Log prob of the true token at each of `positions`, each masked alone.

    Returns (values, total). `values` holds one float per position, in the
    order of `positions`, which is what the narrower ranges are summed from.
    `total` is accumulated the way it was before the ranges existed -- one
    reduction per chunk on the device, added in chunk order -- so the
    whole-window value is arrived at by the same arithmetic as the runs of
    2026-10-30 and can be checked against them.
    """
    values, total = [], 0.0
    for start in range(0, len(positions), pos_batch):
        chunk = positions[start:start + pos_batch]
        batch = ids_row.unsqueeze(0).repeat(len(chunk), 1).clone()
        truth = ids_row[chunk]
        for row, pos in enumerate(chunk):
            batch[row, pos] = mask_id
        logits = model(input_ids=batch,
                       attention_mask=torch.ones_like(batch)).logits
        lp = F.log_softmax(logits[range(len(chunk)), chunk].float(), dim=-1)
        got = lp.gather(1, truth.unsqueeze(1))
        total += float(got.sum())
        values.extend(got.squeeze(1).tolist())
    return values, total


def range_slice(n, at, half):
    """The index range a half-width covers, clipped to the window."""
    if half is None:
        return 0, n
    return max(0, at - half), min(n, at + half + 1)


def range_means(values, at, total=None):
    """Mean log prob over each range, from one pass's per-position values.

    The mean rather than the sum, because AUROC does not change when every
    score is divided by the same constant, and a mean is the same quantity
    whether the range holds 1,024 positions or one.

    `total` is the whole window's sum as the device reduced it; when it is
    given, the window's mean is that number divided by the count instead of a
    second summation in a different order.
    """
    out = {}
    for name, half in RANGES:
        lo, hi = range_slice(len(values), at, half)
        if half is None and total is not None:
            out[name] = total / (hi - lo)
        else:
            out[name] = sum(values[lo:hi]) / (hi - lo)
    return out


def range_counts(n, at):
    """How many positions each range covers."""
    return {name: (lambda lo, hi: hi - lo)(*range_slice(n, at, half))
            for name, half in RANGES}


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rows = read_rows(args.csv, set(args.chroms.split(",")), args.max_variants)
    labels = [r["label_pathogenic"] for r in rows]
    print(f"  {len(rows):,} 変異（病原性 {sum(labels):,} / 良性 {len(labels) - sum(labels):,}）")

    ckpt, ckpt_step = adopted_checkpoint(args.run_dir)
    print(f"  checkpoint: {ckpt}  (step {ckpt_step:,})")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(ckpt, torch_dtype=dtype)
    model.to(device).eval()

    wrapped = args.input_form == "wrapped"
    window = len(rows[0]["reference_sequence"])
    offset = 1 if wrapped else 0
    cls_id, sep_id, mask_id = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id

    def encode(seq):
        ids = tok.convert_tokens_to_ids(list(seq))
        return ([cls_id] + ids + [sep_id]) if wrapped else ids

    # Variant mode scores one position, so the only range it can report is that
    # position. Window mode scores all of them and reports all four.
    names = [name for name, _half in RANGES] if args.positions == "window" else ["point"]
    counts = (range_counts(window, window // 2) if args.positions == "window"
              else {"point": 1})
    scored = {name: [] for name in names}
    lls = {"ref": {name: [] for name in names},
           "var": {name: [] for name in names}}

    t0 = time.time()
    if args.positions == "variant":
        at = window // 2 + offset
        with torch.no_grad():
            for lo in range(0, len(rows), args.variant_batch):
                chunk = rows[lo:lo + args.variant_batch]
                got = {}
                for side, key in (("ref", "reference_sequence"),
                                  ("var", "variant_sequence")):
                    ids = torch.tensor([encode(r[key]) for r in chunk],
                                       dtype=torch.long, device=device)
                    truth = ids[:, at].clone()
                    ids[:, at] = mask_id
                    logits = model(input_ids=ids,
                                   attention_mask=torch.ones_like(ids)).logits
                    lp = F.log_softmax(logits[:, at].float(), dim=-1)
                    got[side] = lp.gather(1, truth.unsqueeze(1)).squeeze(1).tolist()
                for ref_lp, var_lp in zip(got["ref"], got["var"]):
                    lls["ref"]["point"].append(ref_lp)
                    lls["var"]["point"].append(var_lp)
                    scored["point"].append(ref_lp - var_lp)
                if lo % (args.variant_batch * 50) == 0:
                    print(f"    {lo:,} / {len(rows):,}  {time.time() - t0:.0f}s", flush=True)
    else:
        positions = list(range(offset, offset + window))
        at = window // 2          # index into `positions`, where the variant is
        with torch.no_grad():
            for i, row in enumerate(rows):
                got = {}
                for side, key in (("ref", "reference_sequence"),
                                  ("var", "variant_sequence")):
                    ids_row = torch.tensor(encode(row[key]), dtype=torch.long,
                                           device=device)
                    values, total = position_logprobs(
                        model, torch, F, ids_row, positions, mask_id,
                        args.position_batch)
                    got[side] = range_means(values, at, total)
                for name in names:
                    lls["ref"][name].append(got["ref"][name])
                    lls["var"][name].append(got["var"][name])
                    scored[name].append(got["ref"][name] - got["var"][name])
                if i and i % 10 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"    {i + 1:,} / {len(rows):,}  {rate:.2f} 変異/秒  "
                          f"全 21,350 件なら {21350 / rate / 3600:.1f} 時間", flush=True)
    elapsed = time.time() - t0

    primary = "window" if args.positions == "window" else "point"
    out = {"run_dir": args.run_dir, "checkpoint": ckpt,
           "checkpoint_step": ckpt_step,
           "input_form": args.input_form, "positions": args.positions,
           "bf16": args.bf16, "n_variants": len(rows), "seconds": round(elapsed, 1),
           "variants_per_second": round(len(rows) / elapsed, 3),
           "primary_range": primary, "ranges": {}}
    for name in names:
        values = scored[name]
        block = {"n_positions": counts[name], "auroc_all": auroc(labels, values)}
        three = [(r["label_pathogenic"], s) for r, s in zip(rows, values)
                 if r["chrom"] in FOLDS]
        if three:
            block["n_three"] = len(three)
            block["auroc_three"] = auroc([y for y, _ in three], [s for _, s in three])
        for fold in FOLDS:
            got = [(r["label_pathogenic"], s) for r, s in zip(rows, values)
                   if r["chrom"] == fold]
            if got:
                block[f"n_chr{fold}"] = len(got)
                block[f"auroc_chr{fold}"] = auroc([y for y, _ in got],
                                                  [s for _, s in got])
        out["ranges"][name] = block

    # The primary range keeps the top-level key names the aggregates had before
    # the ranges existed, because the window table and the fold table read them.
    for key, value in out["ranges"][primary].items():
        if key != "n_positions":
            out[key] = value

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The per-variant scores go out first, so a crash between the two files
    # leaves the rows and not just the mean. The sequences are not repeated
    # here: they are in the CSV, and the VCV id joins a row back to them.
    stem = args.out[:-len(".json")] if args.out.endswith(".json") else args.out
    preds = f"{stem}-predictions.jsonl"
    columns = None
    with open(preds, "w") as handle:
        for i, row in enumerate(rows):
            rec = {"vcv_id": row["vcv_id"], "chrom": row["chrom"],
                   "pos": row["pos"], "ref_allele": row["ref_allele"],
                   "alt_allele": row["alt_allele"],
                   "clinical_significance": row["clinical_significance"],
                   "fold": f"chr{row['chrom']}",
                   "label_pathogenic": row["label_pathogenic"]}
            for name in names:
                rec[f"ll_ref_{name}"] = lls["ref"][name][i]
                rec[f"ll_var_{name}"] = lls["var"][name][i]
                rec[f"score_{name}"] = scored[name][i]
            columns = columns or list(rec)
            handle.write(json.dumps(rec) + "\n")
    out["predictions"] = os.path.basename(preds)
    out["prediction_columns"] = columns
    out["prediction_rows"] = len(rows)
    print(f"  wrote {preds}  ({len(rows):,} 行 / {len(columns or [])} 列)")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  {args.input_form}/{args.positions}: AUROC {out['auroc_all']:.6f}  "
          f"{elapsed:.0f}s  {out['variants_per_second']:.3f} 変異/秒")
    for name in names:
        block = out["ranges"][name]
        print(f"    {name:9s} ({block['n_positions']:,} 箇所): "
              f"AUROC {block['auroc_all']:.6f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
