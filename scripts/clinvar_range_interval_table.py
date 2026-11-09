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

``--summary-out`` writes the second table the judgment of 2026-11-09 asks for:
one row per control and range, carrying the number of runs that cleared the
control on all three chromosomes **beside the control's own per-chromosome
AUROCs**. §4.1: the counts differ (20 of 21 against the 5-mer Markov model, 6
of 21 against the model-free control) because the controls differ in strength,
and the two are not to be put side by side without the values that explain it.
No AUROC pooled over the three chromosomes appears in either table.

The same row carries the group test. §4.2: it ranks the count-weighted mean,
not one AUROC over the three chromosomes pooled, and the row names the quantity
it ranked so the table cannot be read as the earlier one.
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
    ap.add_argument("--summary-out", default="",
                    help="TSV of one row per control and range: how many runs "
                         "cleared the control on all three chromosomes, the "
                         "control's per-chromosome AUROCs, and the group test")
    args = ap.parse_args()

    rows, summary = [], []
    for spec in args.json:
        if ":" not in spec:
            raise SystemExit(f"{spec!r}: expected RANGE:PATH")
        rng, path = spec.split(":", 1)
        d = json.load(open(path))
        if d.get("score_key") and d["score_key"] != f"score_{rng}" \
                and d["score_key"] != rng:
            raise SystemExit(f"{path}: was run on {d['score_key']!r}, labelled "
                             f"{rng!r}")
        g = d.get("group_test", {})
        beat_all = sum(1 for t in d["runs"].values()
                       if t["folds_beaten"] == len(FOLDS))
        srow = {"control": d.get("baseline_name") or d["baseline_key"],
                "range": rng, "n_runs": len(d["runs"]),
                "runs_beating_all_three": beat_all,
                "n_common": d.get("n_variants", ""),
                "rounds": d.get("rounds", ""), "seed": d.get("seed", "")}
        for c in FOLDS:
            srow[f"control_auroc_chr{c}"] = round(d["folds"][c]["baseline_auroc"], 6)
            srow[f"n_chr{c}"] = d["folds"][c]["n"]
            srow[f"pathogenic_rate_chr{c}"] = round(d["folds"][c]["pathogenic_rate"], 4)
        # The group test ranks the runs, so it does not depend on the control;
        # it is carried on the row because the row is per control and range.
        srow["group_metric"] = g.get("metric", "")
        srow["global_random_median"] = (round(g["global_random_median"], 6)
                                        if "global_random_median" in g else "")
        srow["eukaryote_matched_median"] = (
            round(g["eukaryote_matched_median"], 6)
            if "eukaryote_matched_median" in g else "")
        srow["mann_whitney_u"] = g.get("u", "")
        srow["z"] = round(g["z"], 4) if "z" in g else ""
        srow["p_two_sided"] = (f"{g['p_two_sided']:.3e}"
                               if "p_two_sided" in g else "")
        srow["prob_global_gt_eukaryote"] = (round(g["prob_x_gt_y"], 4)
                                            if "prob_x_gt_y" in g else "")
        if "baseline_overall" in d:
            raise SystemExit(f"{path}: carries a pooled baseline AUROC, which "
                             f"§4.1 of the judgment of 2026-11-09 withdrew; "
                             f"it was scored before that and has to be redone")
        summary.append(srow)

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

    if args.summary_out:
        with open(args.summary_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]), delimiter="\t")
            w.writeheader()
            w.writerows(summary)
        print(f"  wrote {args.summary_out}  ({len(summary)} 行 = "
              f"{len({r['control'] for r in summary})} 対照 x "
              f"{len({r['range'] for r in summary})} 範囲)")

    print("\n  === 群間（件数加重平均で検定、mammal_centered は除外） ===")
    print(f"  {'範囲':10s} {'global 中央':>12s} {'eukaryote 中央':>15s} "
          f"{'U':>5s} {'z':>7s} {'p':>10s}")
    seen = set()
    for r in summary:
        if r["range"] in seen or not r["group_metric"]:
            continue
        seen.add(r["range"])
        print(f"  {r['range']:10s} {r['global_random_median']:>12} "
              f"{r['eukaryote_matched_median']:>15} {r['mann_whitney_u']:>5} "
              f"{r['z']:>7} {r['p_two_sided']:>10}")

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
