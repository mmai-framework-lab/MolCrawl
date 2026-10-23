"""Does the BERT zero-shot score change when the window is wrapped as in training?

The evaluation adapter encodes through the tokeniser, and this tokeniser has
[CLS] and [SEP] in its vocabulary but no post-processor template, so
add_special_tokens=True returns 1,024 bare base tokens. The model was trained on
[CLS] + 1,024 + [SEP], and the linear probe builds that form itself. Zero-shot
therefore scores an input shape the model never saw in training.

Whether that moves the number is a separate question from whether it is tidy,
and it had not been measured. This scores one run both ways and reports both.

**The bare mode is the control.** If it does not reproduce the AUROC the
campaign recorded for this run, the wrapped number means nothing either, and the
run says so rather than reporting a difference between two things that are both
wrong.

The score is the campaign's: mask the variant position, take the log
probability of the base that is actually there, and subtract the variant
sequence's value from the reference's.
"""
import argparse
import csv
import json
import sys


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


def read_rows(path, chroms):
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
        out.append((r["reference_sequence"], r["variant_sequence"], 1 if p else 0))
    return out


def score_all(rows, tok, model, device, wrapped, batch_size):
    """Log prob of the base at the variant position, masked, per sequence.

    `wrapped` chooses the token stream: the bare 1,024 bases the adapter sends,
    or [CLS] + bases + [SEP], which is what training used and what the probe
    builds.
    """
    import torch
    import torch.nn.functional as F

    mask_id = tok.mask_token_id
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    window = len(rows[0][0])
    at = window // 2 + (1 if wrapped else 0)

    def encode(seq):
        ids = tok.convert_tokens_to_ids(list(seq))
        return ([cls_id] + ids + [sep_id]) if wrapped else ids

    scores = []
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo:lo + batch_size]
            got = []
            for which in (0, 1):                      # 0 reference, 1 variant
                ids = torch.tensor([encode(r[which]) for r in chunk],
                                   dtype=torch.long, device=device)
                truth = ids[:, at].clone()
                ids[:, at] = mask_id
                logits = model(input_ids=ids,
                               attention_mask=torch.ones_like(ids)).logits
                lp = F.log_softmax(logits[:, at].float(), dim=-1)
                got.append(lp.gather(1, truth.unsqueeze(1)).squeeze(1))
            scores.extend((got[0] - got[1]).tolist())
            if lo % (batch_size * 50) == 0:
                print(f"    {lo:,} / {len(rows):,}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--recorded-metrics", default="",
                    help="metrics.json of the campaign's run, for the control")
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rows = read_rows(args.csv, set(args.chroms.split(",")))
    labels = [y for _r, _v, y in rows]
    print(f"  {len(rows):,} 変異（病原性 {sum(labels):,} / 良性 {len(labels) - sum(labels):,}）")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint).to(device).eval()
    print(f"  max_position_embeddings = {model.config.max_position_embeddings}")

    out = {}
    for wrapped in (False, True):
        name = "wrapped" if wrapped else "bare"
        print(f"\n  === {name} ===", flush=True)
        s = score_all(rows, tok, model, device, wrapped, args.batch_size)
        out[name] = auroc(labels, s)
        print(f"  {name}: AUROC {out[name]:.6f}")

    if args.recorded_metrics:
        recorded = json.load(open(args.recorded_metrics))["metrics"]["auroc"]
        out["recorded"] = recorded
        gap = out["bare"] - recorded
        print(f"\n  対照: campaign の記録 {recorded:.6f} / 再現 {out['bare']:.6f} "
              f"（差 {gap:+.6f}）")
        if abs(gap) > 1e-3:
            print("  再現できていない。wrapped の値は比較に使えない。")
        else:
            print("  再現できている。wrapped の値は比較に使える。")

    out["wrapped_minus_bare"] = out["wrapped"] - out["bare"]
    print(f"\n  wrapped − bare = {out['wrapped_minus_bare']:+.6f}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
