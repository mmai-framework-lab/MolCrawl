"""Which token position the zero-shot score actually reads.

The evaluator builds its position list from `flank` alone:

    lo = max(0, flank - score_window_half)
    hi = flank + score_window_half + 1

and hands those indices to the adapter, which filters raw token-stream
indices. The BERT tokeniser encodes with add_special_tokens=True, so index 0
is [CLS] and base k sits at index k + 1. The linear probe adds that offset
explicitly (variant_token_index); the zero-shot path does not.

If that is right, BERT's single-position score reads the base to the left of
the variant. This encodes a real row and says what sits where, rather than
arguing it from the source.
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--flank", type=int, default=512)
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(args.csv, newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    ref, var = row["reference_sequence"], row["variant_sequence"]
    diff = [i for i, (a, b) in enumerate(zip(ref, var)) if a != b]
    print(f"  変異 {row.get('vcv_id', '?')}  ref={row['ref']} alt={row['alt']}")
    print(f"  窓長 {len(ref)}  参照と変異で異なる塩基の位置: {diff}")
    print(f"  flank = {args.flank}  → 評価器が渡す位置は [{args.flank}]")
    print(f"  塩基列での中央 {len(ref) // 2}: ref '{ref[len(ref) // 2]}' / var '{var[len(ref) // 2]}'")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = list(tok.encode(ref, add_special_tokens=True))
    specials = set(getattr(tok, "all_special_ids", ()) or ())
    print(f"\n  トークン列の長さ {len(ids)}（塩基 {len(ref)} + 特別トークン {len(ids) - len(ref)}）")
    print(f"  先頭 3 個: {[tok.convert_ids_to_tokens(i) for i in ids[:3]]}")
    print(f"  末尾 2 個: {[tok.convert_ids_to_tokens(i) for i in ids[-2:]]}")

    for idx in (args.flank, args.flank + 1):
        if idx >= len(ids):
            continue
        token = tok.convert_ids_to_tokens(ids[idx])
        base_at = idx - (1 if ids[0] in specials else 0)
        print(f"  トークン位置 {idx}: '{token}'  = 塩基 {base_at} "
              f"（ref '{ref[base_at]}'）{'  ← 変異位置' if base_at == diff[0] else ''}")

    scored = args.flank
    base_scored = scored - (1 if ids[0] in specials else 0)
    print(f"\n  結論: 評価器は位置 {scored} を採点する。"
          f"それは塩基 {base_scored} であり、変異は塩基 {diff[0]} にある。")
    print(f"  ずれ: {base_scored - diff[0]} 塩基")


if __name__ == "__main__":
    main()
