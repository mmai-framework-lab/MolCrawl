"""The interval comparison against the controls, one row per run and range.

§4 of the instruction of 2026-11-02. ``clinvar_final_analysis.py`` writes one
JSON per (control, summing range); this gathers them into one table so the
question the instruction asks -- whether beating the control depends on how
wide a neighbourhood is summed -- can be read off a single file.

A run is said to beat the control on a chromosome when the lower end of the
bootstrap interval on the paired difference is above zero, not when the point
estimate is higher. The interval is over the variants, resampled 2,000 times
with the seed the instruction fixes, and the difference is taken variant by
variant because both terms score the same variants.

The two controls cover slightly different variants -- the 5-mer Markov file
carries chrY's 29 and the model-free control does not -- so the row records
which control it was measured against and over how many variants.
"""
import argparse
import csv
import json
import os

FOLDS = ("21", "22", "X")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True, metavar="RANGE:PATH",
                    help="a clinvar_final_analysis.py output, prefixed by the "
                         "summing range it was run on, e.g. window:/path/x.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.json:
        if ":" not in spec:
            raise SystemExit(f"{spec!r}: expected RANGE:PATH")
        rng, path = spec.split(":", 1)
        d = json.load(open(path))
        if d.get("score_key") and d["score_key"] != f"score_{rng}" \
                and d["score_key"] != rng:
            raise SystemExit(f"{path}: was run on {d['score_key']!r}, labelled "
                             f"{rng!r}")
        for subset, t in d["runs"].items():
            for c in FOLDS:
                f = t["per_fold"][c]
                lo, hi = f["diff_ci"]
                rows.append({
                    "subset": subset, "range": rng,
                    "control": d.get("baseline_name") or d["baseline_key"],
                    "chrom": f"chr{c}",
                    "n_variants": d["folds"][c]["n"],
                    "pathogenic_rate": round(d["folds"][c]["pathogenic_rate"], 4),
                    "auroc": round(f["auroc"], 6),
                    "control_auroc": round(d["folds"][c]["baseline_auroc"], 6),
                    "diff": round(f["diff_vs_baseline"], 6),
                    "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                    "beats_control": int(f["beats_baseline"]),
                    "rounds": d.get("rounds", ""), "seed": d.get("seed", ""),
                    "n_common": d.get("n_variants", ""),
                })

    if not rows:
        raise SystemExit("no rows; check --json")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    n_runs = len({r["subset"] for r in rows})
    print(f"  wrote {args.out}  ({len(rows)} 行 = {n_runs} run x "
          f"{len({r['range'] for r in rows})} 範囲 x "
          f"{len({r['control'] for r in rows})} 対照 x {len(FOLDS)} 染色体)")

    print("\n  === 3 染色体すべてで対照を越えた run の数 ===")
    print(f"  {'対照':28s} " + "  ".join(f"{r:>10s}" for r in
                                        ("window", "flank128", "flank32", "point")))
    for ctl in sorted({r["control"] for r in rows}):
        line = []
        for rng in ("window", "flank128", "flank32", "point"):
            sel = [r for r in rows if r["control"] == ctl and r["range"] == rng]
            if not sel:
                line.append(f"{'-':>10s}")
                continue
            by = {}
            for r in sel:
                by.setdefault(r["subset"], 0)
                by[r["subset"]] += r["beats_control"]
            won = sum(1 for s in by if by[s] == len(FOLDS))
            line.append(f"{won:>6d}/{len(by):<3d}")
        print(f"  {ctl:28s} " + "  ".join(line))


if __name__ == "__main__":
    main()
