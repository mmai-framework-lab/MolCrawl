"""One table for the ClinVar results: 21 subsets, three folds, both architectures.

The campaign produced four sets of numbers -- zero-shot and a linear probe, for
BERT and for GPT-2 -- each in its own file, each with its own fold naming and
its own idea of what a row is. Read separately they invite two mistakes: that a
value can be compared across folds (chr21 is 21% pathogenic, chrX 48%, and
AUROC moves with class balance), and that a run's number means anything without
the two reference values beside it.

So this writes one long-form row per cell -- one subset, one architecture, one
method, one fold -- and carries the references in the same table under `kind`:

  run          a trained subset
  model-free   the 14-feature control, the line a model has to beat
  untrained    the same architecture at its initialisation, three seeds

A wide table would have to pick an ordering and bake it in. Long form does not,
and every spreadsheet pivots it.

`mammal_centered` is one run. It is in the table as one row like any other, and
nothing here aggregates it into a group of one.
"""
import argparse
import csv
import json
import os
import statistics

# The table's columns, declared once. Three places used to unpack a row by
# position, so adding one meant finding all three -- and the same shape of
# mistake cost a twenty-five minute job on the loss table. Rows are built and
# read through this.
COLUMNS = ["kind", "subset", "family", "arch", "series", "epochs", "seed",
           "method", "fold", "auroc", "checkpoint_step", "note"]
_C = {name: i for i, name in enumerate(COLUMNS)}

FOLDS = ["21", "22", "X"]            # chrY is 29 variants, never a fold
FAMILIES = ("mammal_centered", "eukaryote_matched", "global_random")


def fold_key(fold, keys):
    """Fold naming differs between files: fold-analysis says 21, probe.json
    says chr21. Neither is wrong; the table has to pick one and translate."""
    for candidate in (fold, f"chr{fold}"):
        if candidate in keys:
            return candidate
    return None


def family_of(subset):
    if subset.startswith("mammal_centered"):
        return "mammal_centered"
    return "eukaryote_matched" if subset.startswith("eukaryote_matched") else "global_random"


def _load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except ValueError:
        return None


def parse_source(spec):
    """`series:epochs:seed:path` -- what a fold-analysis file is of.

    The file itself does not say. It holds AUROCs per subset and nothing about
    the schedule or the seed they came from, so a table built from two of them
    cannot tell three-epoch rows from nine-epoch ones, nor seed 45 from 42.
    The judgment of 2026-10-22 asks for both on the row, and the re-run at
    seed 42 will sit beside the present values until it is complete.
    """
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise SystemExit(f"--... expects series:epochs:seed:path, got {spec!r}")
    series, epochs, seed, path = parts
    return series, epochs, seed, path


def run_rows(analysis, probe, arch, method, series="", epochs="", seed=""):
    """One row per subset per fold, with the checkpoint the run adopted.

    The step comes from the probe's record even for the zero-shot rows: both
    score the checkpoint the run itself selected, so there is one step per run,
    not one per method. Taking it from two places would let them disagree.
    """
    steps = {}
    if probe:
        for subset, got in probe.get("runs", {}).items():
            name = str(got.get("checkpoint", ""))
            steps[subset] = name.rsplit("-", 1)[1] if "-" in name else name
    rows = []
    for subset, got in sorted(analysis.get("runs", {}).items()):
        per_fold = got.get("per_fold", {})
        for fold in FOLDS:
            key = fold_key(fold, per_fold)
            if key is None:
                continue
            rows.append(["run", subset, family_of(subset), arch, series,
                         epochs, seed, method, fold,
                         f"{per_fold[key]['auroc']:.6f}", steps.get(subset, ""), ""])
    return rows


def reference_rows(analysis, untrained, arch):
    """The model-free control and the untrained floor, as rows of the same table.

    The control is what a model has to beat to have learned anything from the
    sequence; the untrained floor is what the same architecture scores before
    it is trained at all. A run's AUROC says little without both.
    """
    rows = []
    folds = analysis.get("folds", {}) if analysis else {}
    for fold in FOLDS:
        key = fold_key(fold, folds)
        if key and folds[key].get("baseline_auroc") is not None:
            rows.append(["model-free", "-", "-", arch, "-", "", "", "probe",
                         fold, f"{folds[key]['baseline_auroc']:.6f}", "",
                         "14 features"])
    for seed, got in sorted((untrained or {}).get("runs", {}).items()):
        per_fold = got.get("per_fold", {})
        for fold in FOLDS:
            key = fold_key(fold, per_fold)
            if key is None:
                continue
            rows.append(["untrained", seed, "-", arch, "-", "", seed, "probe",
                         fold, f"{per_fold[key]['auroc']:.6f}", "",
                         "random weights"])
    return rows


def summarise(rows):
    """Mean and range per family, per architecture, per method, per fold.

    A family of one is reported as its single value with no range, because a
    range over one run is not a measure of anything.
    """
    groups = {}
    for row in rows:
        if row[_C["kind"]] != "run":
            continue
        # The series is part of the key: a three-epoch and a nine-epoch row for
        # one subset are two measurements, and averaging them would report a
        # number that describes neither.
        key = (row[_C["arch"]], row[_C["series"]], row[_C["epochs"]],
               row[_C["method"]], row[_C["fold"]], row[_C["family"]])
        groups.setdefault(key, []).append(float(row[_C["auroc"]]))
    out = []
    for (arch, series, epochs, method, fold, family), values in sorted(groups.items()):
        out.append({"arch": arch, "series": series, "epochs": epochs,
                    "method": method, "fold": fold, "family": family,
                    "n": len(values), "mean": statistics.fmean(values),
                    "min": min(values), "max": max(values),
                    "range": (max(values) - min(values)) if len(values) > 1 else None})
    return out


def markdown(rows, summary):
    """A rendering for the report. The TSV stays the record."""
    lines = ["# ClinVar 結果表", "",
             "AUROC。fold ごとに陽性の割合が違うため、fold をまたいで比べない。", ""]
    refs = {(r[_C["arch"]], r[_C["fold"]]): r[_C["auroc"]]
            for r in rows if r[_C["kind"]] == "model-free"}
    untrained = {}
    for r in rows:
        if r[_C["kind"]] == "untrained":
            untrained.setdefault((r[_C["arch"]], r[_C["fold"]]), []).append(
                float(r[_C["auroc"]]))

    for arch in ("bert", "gpt2"):
        series_seen = sorted({(s["series"], s["epochs"]) for s in summary
                              if s["arch"] == arch})
        lines += [f"## {arch}", "",
                  "| 系列 | エポック | seed | 群 | 手法 | "
                  + " | ".join(f"chr{f}" for f in FOLDS) + " |",
                  "|---|---|---|---|---|" + "---|" * len(FOLDS)]
        seeds = {(r[_C["arch"]], r[_C["series"]]): r[_C["seed"]]
                 for r in rows if r[_C["kind"]] == "run"}
        for series, epochs in series_seen:
            for method in ("zeroshot", "probe"):
                for family in FAMILIES:
                    cells = []
                    for fold in FOLDS:
                        got = [s for s in summary
                               if (s["arch"], s["series"], s["method"], s["fold"],
                                   s["family"]) == (arch, series, method, fold, family)]
                        if not got:
                            cells.append("-")
                            continue
                        s = got[0]
                        cells.append(f"{s['mean']:.4f}" if s["n"] == 1 else
                                     f"{s['mean']:.4f} ({s['min']:.4f}〜{s['max']:.4f})")
                    if set(cells) == {"-"}:
                        continue
                    lines.append(f"| {series} | {epochs} | {seeds.get((arch, series), '')} "
                                 f"| {family} | {method} | " + " | ".join(cells) + " |")
        model_free = [refs.get((arch, f), "-") for f in FOLDS]
        lines.append("| - | - | - | モデル不使用の予測器 | - | " +
                     " | ".join(f"{float(v):.4f}" if v != "-" else "-" for v in model_free) + " |")
        floor = []
        for fold in FOLDS:
            got = untrained.get((arch, fold), [])
            floor.append(f"{statistics.fmean(got):.4f} ({min(got):.4f}〜{max(got):.4f})"
                         if got else "-")
        lines.append("| - | - | - | 未学習 3 seed | - | " + " | ".join(floor) + " |")
        lines.append("")
    lines += ["`mammal_centered` は 1 本である。群としての幅は出していない。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    for name in ("zeroshot-bert", "zeroshot-gpt2", "probe-bert", "probe-gpt2"):
        ap.add_argument(f"--{name}", action="append", default=[],
                        metavar="series:epochs:seed:path",
                        help=f"{name} fold-analysis.json, repeatable: one per "
                             "series, so three-epoch and nine-epoch rows can "
                             "sit in the same table and be told apart")
    ap.add_argument("--probe-json-bert", default="", help="probe.json, for checkpoint steps")
    ap.add_argument("--probe-json-gpt2", default="")
    ap.add_argument("--untrained-bert", default="", help="untrained probe.json")
    ap.add_argument("--untrained-gpt2", default="")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    rows = []
    for arch in ("bert", "gpt2"):
        probe_json = _load(getattr(args, f"probe_json_{arch}"))
        last_probe = None
        for method in ("zeroshot", "probe"):
            for spec in getattr(args, f"{method}_{arch}"):
                series, epochs, seed, path = parse_source(spec)
                analysis = _load(path)
                if analysis is None:
                    print(f"  {arch}/{method}/{series}: {path} が読めないので飛ばす")
                    continue
                rows += run_rows(analysis, probe_json, arch, method,
                                 series, epochs, seed)
                if method == "probe":
                    last_probe = analysis
        # The controls come from a probe analysis and do not belong to a
        # series: the model-free predictor has no schedule, and the untrained
        # floor is the architecture before any training. One set per
        # architecture, taken from whichever probe analysis was read last.
        rows += reference_rows(last_probe,
                               _load(getattr(args, f"untrained_{arch}")), arch)

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(COLUMNS)
        bad = next((r for r in rows if len(r) != len(COLUMNS)), None)
        if bad is not None:
            raise SystemExit(f"row has {len(bad)} fields, COLUMNS has "
                             f"{len(COLUMNS)}: {bad}")
        writer.writerows(rows)
    print(f"  wrote {args.out_tsv}  ({len(rows):,} rows)")

    summary = summarise(rows)
    if args.out_md:
        open(args.out_md, "w").write(markdown(rows, summary))
        print(f"  wrote {args.out_md}")

    counts = {}
    for kind, _s, _f, arch, series, epochs, _seed, method, *_ in rows:
        key = (kind, arch, series, epochs, method)
        counts[key] = counts.get(key, 0) + 1
    print(f"\n  {'kind':12s} {'arch':6s} {'series':18s} {'ep':>3s} {'method':9s} 行数")
    for key in sorted(counts):
        print(f"  {key[0]:12s} {key[1]:6s} {key[2]:18s} {key[3]:>3s} "
              f"{key[4]:9s} {counts[key]:>4d}")


if __name__ == "__main__":
    main()
