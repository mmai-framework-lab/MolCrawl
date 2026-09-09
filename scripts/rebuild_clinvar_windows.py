"""Rebuild the ClinVar evaluation windows at the length the models were trained on.

The shipped table carries 129 bases per variant (+/-64). The genome models were
pretrained on 1,024-token windows, so scoring them on 129 gives them an eighth of
the context they were built for. Only the length changes here: the recipe is the
same one prepare_clinvar.py uses, and the variant still sits at the centre.

The window is 512 bases before, the variant, then 511 after -- 1,024 in total,
with the variant 513th. That is deliberately asymmetric: 512 either side would be
1,025 and overrun both GPT-2's 1,024-token block and BERT's [CLS] + 1,024 +
[SEP]. One base of offset against 500+ bases of context either way costs nothing;
a window the model cannot take costs a silent truncation.

Two exclusions, counted separately because they have different causes:

  end-of-chromosome  the window would run past either end
  N in window        the reference carries an unresolved base

The second is not optional. fasta_to_raw.py splits contigs at runs of N before
chunking, so no training window ever contained one -- the model has never been
asked to read an N and has no representation to bring to it. chr21 and chr22
carry large N blocks at the short arm and centromere, so this is expected to be
the larger of the two.

Every excluded variant is dropped for all runs alike, which keeps the campaign's
comparison paired.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

LEFT = 512          # bases before the variant
RIGHT = 511         # bases after it; LEFT + 1 + RIGHT = 1024
CENTRE = LEFT       # 0-based index of the variant within the window


def build_chrom_mapping(ref_genome):
    """chromosome name -> FASTA sequence id, from the assembly's own headers."""
    import re
    mapping = {}
    for seq_id in ref_genome.keys():
        m = re.search(r"^(CM\d+\.\d+).*chromosome (\w+)", ref_genome[seq_id].long_name)
        if m:
            chrom = m.group(2)
            mapping["MT" if chrom.lower().startswith("mito") else chrom] = m.group(1)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="existing clinvar_sequences.csv")
    ap.add_argument("--ref-fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    from pyfaidx import Fasta

    want = {c.strip() for c in args.chroms.split(",") if c.strip()}
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    ref_genome = Fasta(args.ref_fasta)
    mapping = build_chrom_mapping(ref_genome)
    missing = sorted(want - set(mapping))
    if missing:
        raise SystemExit(f"no FASTA sequence for chromosome(s): {', '.join(missing)}")

    counts = Counter()
    per_chrom = {c: Counter() for c in sorted(want)}
    kept_rows = []

    with open(args.source, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            chrom = (row.get("chrom") or "").replace("chr", "").strip()
            if chrom not in want:
                continue
            counts["considered"] += 1
            per_chrom[chrom]["considered"] += 1

            pos = int(row["pos"])
            seq = ref_genome[mapping[chrom]]
            start0 = pos - 1 - LEFT               # 0-based slice start
            end0 = pos + RIGHT                    # 0-based slice end (exclusive)
            if start0 < 0 or end0 > len(seq):
                counts["excluded_end_of_chromosome"] += 1
                per_chrom[chrom]["excluded_end_of_chromosome"] += 1
                continue

            window = str(seq[start0:end0]).upper()
            if len(window) != LEFT + 1 + RIGHT:   # defensive: pyfaidx clips silently
                counts["excluded_end_of_chromosome"] += 1
                per_chrom[chrom]["excluded_end_of_chromosome"] += 1
                continue
            if "N" in window:
                counts["excluded_n_in_window"] += 1
                per_chrom[chrom]["excluded_n_in_window"] += 1
                continue

            # The centre must be the reference allele. A systematic mismatch means
            # the coordinates and the assembly disagree, and nothing below it is
            # worth computing -- so it is counted, not warned about.
            ref = (row.get("ref") or "").upper()
            if window[CENTRE] != ref:
                counts["centre_ref_mismatch"] += 1
                per_chrom[chrom]["centre_ref_mismatch"] += 1
                continue

            var = list(window)
            var[CENTRE] = (row.get("alt") or "").upper()
            out = dict(row)
            out["reference_sequence"] = window
            out["variant_sequence"] = "".join(var)
            kept_rows.append(out)
            counts["kept"] += 1
            per_chrom[chrom]["kept"] += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept_rows)

    print(f"  window {LEFT} + 1 + {RIGHT} = {LEFT + 1 + RIGHT} bases, variant at "
          f"{CENTRE + 1}th")
    print(f"  {'chrom':>6s} {'対象':>8s} {'採用':>8s} {'端':>8s} {'N':>8s} {'ref不一致':>10s}")
    for c in sorted(per_chrom):
        d = per_chrom[c]
        print(f"  {c:>6s} {d['considered']:>8,} {d['kept']:>8,}"
              f" {d['excluded_end_of_chromosome']:>8,} {d['excluded_n_in_window']:>8,}"
              f" {d['centre_ref_mismatch']:>10,}")
    print(f"  {'計':>6s} {counts['considered']:>8,} {counts['kept']:>8,}"
          f" {counts['excluded_end_of_chromosome']:>8,} {counts['excluded_n_in_window']:>8,}"
          f" {counts['centre_ref_mismatch']:>10,}")
    if counts["centre_ref_mismatch"]:
        frac = counts["centre_ref_mismatch"] / max(1, counts["considered"])
        print(f"\n  ⚠️  中央塩基が ref と一致しない変異が {frac:.2%}。"
              f"多い場合は座標系が違う可能性があり、先に確認が要ります。")
    print(f"\n  wrote {args.out}")

    if args.report:
        json.dump({"window": {"left": LEFT, "right": RIGHT,
                              "length": LEFT + 1 + RIGHT, "variant_index": CENTRE},
                   "total": dict(counts),
                   "per_chrom": {c: dict(d) for c, d in per_chrom.items()}},
                  open(args.report, "w"), indent=2)
        print(f"  wrote {args.report}")


if __name__ == "__main__":
    main()
