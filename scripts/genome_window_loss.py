"""Per-subset window loss and segment length distribution.

A raw line is not a contig. fasta_to_raw.py splits each contig at runs of N and
then wraps any piece longer than RAW_LINE_LEN (261,120) across several lines,
repeating the contig id on each. In mammal_centered's human assembly that is
12,114 lines carrying 701 contig ids, and every wrapped line is exactly
RAW_LINE_LEN, so counting lines as contigs caps the mean at 261k and dilutes the
share that yields no window -- worst exactly where the assemblies are
chromosome-scale.

The two halves of this file therefore work in different units, deliberately:

  windows / bases lost   per LINE, because raw_to_parquet_single_nuc.py chunks
                         per line too. Line arithmetic is what the pipeline does,
                         which is why the dataset cross-check below passes and
                         would not have caught the confusion on its own.
  length distribution    per SEGMENT, reassembled from the lines. N-splitting
                         happened upstream, so the unit is a segment, not a
                         contig, and it is named that way here.

Chunking into fixed windows throws away two things: the remainder at the end of
every line, and every line shorter than one window. Which of the two dominates
decides what a longer window costs, and it is not the same across subsets -- a
subset of many short segments loses whole segments, one of few long segments
loses only tails.

analyze_genome_window_loss.py answers a different question and stays: it walks
one subset and compares two chunk lengths against each other, which is what
settled where the 512-to-1,024 shortfall came from. This one walks all 21 at a
single chunk length, cross-checks the count against the dataset that was built,
and writes JSON the results table reads back.

Reads raw_files/*.raw (contig_id TAB sequence) and counts windows arithmetically
rather than re-chunking, then cross-checks the total against the GPT-2 dataset
actually built from it. A mismatch means the pipeline does something this model
does not, and the numbers below it should not be trusted.
"""
import argparse
import json
import os
from collections import Counter

RAW_LINE_LEN = 261_120          # fasta_to_raw.py wraps lines at this
BUCKETS = [0, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576]


def bucket(n):
    for i in range(len(BUCKETS) - 1, -1, -1):
        if n >= BUCKETS[i]:
            return BUCKETS[i]
    return 0


def scan(raw_dir, window):
    """Window arithmetic per line; length statistics per reassembled segment.

    A segment ends when a line is shorter than ``RAW_LINE_LEN`` or when the contig
    id changes -- the wrap writes full-length lines and then a short one. A
    segment whose length is an exact multiple of RAW_LINE_LEN has no short final
    line and merges with whatever follows it under the same contig id; that is
    one segment boundary in about 261,120, which does not move any figure here.
    """
    lines = 0
    segments = 0
    bases = 0
    windows = 0
    remainder = 0
    too_short_lines = 0
    short_line_bases = 0
    seg_hist = Counter()
    seg_no_window = 0

    cur_id = None
    cur_len = 0

    def close_segment(n):
        nonlocal segments, seg_no_window
        if not n:
            return
        segments += 1
        seg_hist[bucket(n)] += 1
        if n < window:
            seg_no_window += 1

    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".raw"):
            continue
        with open(os.path.join(raw_dir, name)) as fh:
            for line in fh:
                tab = line.find("\t")
                if tab < 0:
                    continue
                cid = line[:tab]
                n = len(line) - tab - 1
                if line.endswith("\n"):
                    n -= 1

                lines += 1
                bases += n
                w = n // window
                if w:
                    windows += w
                    remainder += n - w * window
                else:
                    too_short_lines += 1
                    short_line_bases += n

                if cur_id is not None and cid != cur_id:
                    close_segment(cur_len)
                    cur_len = 0
                cur_id = cid
                cur_len += n
                if n < RAW_LINE_LEN:          # the wrap's final piece
                    close_segment(cur_len)
                    cur_len = 0
                    cur_id = None
        close_segment(cur_len)
        cur_len = 0
        cur_id = None

    return dict(lines=lines, segments=segments, bases=bases, windows=windows,
                remainder_bases=remainder, lines_too_short=too_short_lines,
                bases_in_short_lines=short_line_bases,
                segments_yielding_no_window=seg_no_window,
                segment_hist=dict(sorted(seg_hist.items())),
                mean_segment_len=(bases / segments) if segments else 0.0,
                mean_line_len=(bases / lines) if lines else 0.0,
                no_window_segment_fraction=(seg_no_window / segments) if segments else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--subsets", default="")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--check-against-dataset", action="store_true",
                    help="compare the arithmetic window count with the built rows")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    subsets = ([s for s in args.subsets.split(",") if s] or
               sorted(d for d in os.listdir(args.src_root)
                      if os.path.isdir(os.path.join(args.src_root, d, "raw_files"))))
    out = []
    for s in subsets:
        print(f"\n########## {s} ##########", flush=True)
        r = scan(os.path.join(args.src_root, s, "raw_files"), args.window)
        kept = r["windows"] * args.window
        lost = r["bases"] - kept
        r["subset"] = s
        r["window"] = args.window
        r["bases_kept"] = kept
        r["bases_lost"] = lost
        r["loss_fraction"] = lost / r["bases"] if r["bases"] else 0.0

        print(f"  segments {r['segments']:>10,}   mean {r['mean_segment_len']:>10,.0f} bases"
              f"   (lines {r['lines']:,}, mean {r['mean_line_len']:,.0f})", flush=True)
        print(f"  windows  {r['windows']:>10,}   bases kept {kept:>15,}", flush=True)
        print(f"  lost     {lost:>15,} bases  ({100*r['loss_fraction']:.2f}%)", flush=True)
        if lost:
            print(f"    of which remainder    {r['remainder_bases']:>15,}"
                  f" ({100*r['remainder_bases']/lost:.1f}% of loss)", flush=True)
            print(f"    of which short lines  {r['bases_in_short_lines']:>15,}"
                  f" ({100*r['bases_in_short_lines']/lost:.1f}% of loss)", flush=True)
        print(f"  segments yielding no window: {r['segments_yielding_no_window']:,}"
              f" ({100*r['no_window_segment_fraction']:.1f}%)", flush=True)

        if args.check_against_dataset:
            from datasets import load_from_disk
            p = os.path.join(args.src_root, s, "training_ready_hf_dataset_gpt2")
            try:
                d = load_from_disk(p)
                built = sum(len(d[k]) for k in d)
                r["dataset_rows"] = built
                r["matches_dataset"] = (built == r["windows"])
                print(f"  built rows {built:,} vs arithmetic {r['windows']:,}"
                      f"  {'MATCH' if built == r['windows'] else 'DIFFER'}", flush=True)
            except Exception as e:                                   # noqa: BLE001
                r["dataset_rows"] = None
                print(f"  could not read built dataset: {e}", flush=True)
        out.append(r)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
