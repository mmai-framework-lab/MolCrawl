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


def run_rows(analysis, probe, arch, method):
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
            rows.append(["run", subset, family_of(subset), arch, method, fold,
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
            rows.append(["model-free", "-", "-", arch, "probe", fold,
                         f"{folds[key]['baseline_auroc']:.6f}", "", "14 features"])
    for seed, got in sorted((untrained or {}).get("runs", {}).items()):
        per_fold = got.get("per_fold", {})
        for fold in FOLDS:
            key = fold_key(fold, per_fold)
            if key is None:
                continue
            rows.append(["untrained", seed, "-", arch, "probe", fold,
                         f"{per_fold[key]['auroc']:.6f}", "", "random weights"])
    return rows


def summarise(rows):
    """Mean and range per family, per architecture, per method, per fold.

    A family of one is reported as its single value with no range, because a
    range over one run is not a measure of anything.
    """
    groups = {}
    for kind, _subset, family, arch, method, fold, auroc, _step, _note in rows:
        if kind != "run":
            continue
        groups.setdefault((arch, method, fold, family), []).append(float(auroc))
    out = []
    for (arch, method, fold, family), values in sorted(groups.items()):
        out.append({"arch": arch, "method": method, "fold": fold, "family": family,
                    "n": len(values), "mean": statistics.fmean(values),
                    "min": min(values), "max": max(values),
                    "range": (max(values) - min(values)) if len(values) > 1 else None})
    return out


def markdown(rows, summary):
    """A rendering for the report. The TSV stays the record."""
    lines = ["# ClinVar 結果表", "",
             "AUROC。fold ごとに陽性の割合が違うため、fold をまたいで比べない。", ""]
    refs = {(r[3], r[5]): r[6] for r in rows if r[0] == "model-free"}
    untrained = {}
    for r in rows:
        if r[0] == "untrained":
            untrained.setdefault((r[3], r[5]), []).append(float(r[6]))

    for arch in ("bert", "gpt2"):
        lines += [f"## {arch}", "",
                  "| 群 | 手法 | " + " | ".join(f"chr{f}" for f in FOLDS) + " |",
                  "|---|---|" + "---|" * len(FOLDS)]
        for method in ("zeroshot", "probe"):
            for family in FAMILIES:
                cells = []
                for fold in FOLDS:
                    got = [s for s in summary if (s["arch"], s["method"], s["fold"],
                                                  s["family"]) == (arch, method, fold, family)]
                    if not got:
                        cells.append("-")
                        continue
                    s = got[0]
                    cells.append(f"{s['mean']:.4f}" if s["n"] == 1 else
                                 f"{s['mean']:.4f} ({s['min']:.4f}〜{s['max']:.4f})")
                lines.append(f"| {family} | {method} | " + " | ".join(cells) + " |")
        model_free = [refs.get((arch, f), "-") for f in FOLDS]
        lines.append("| モデル不使用の予測器 | - | " +
                     " | ".join(f"{float(v):.4f}" if v != "-" else "-" for v in model_free) + " |")
        floor = []
        for fold in FOLDS:
            got = untrained.get((arch, fold), [])
            floor.append(f"{statistics.fmean(got):.4f} ({min(got):.4f}〜{max(got):.4f})"
                         if got else "-")
        lines.append("| 未学習 3 seed | - | " + " | ".join(floor) + " |")
        lines.append("")
    lines += ["`mammal_centered` は 1 本である。群としての幅は出していない。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    for name in ("zeroshot-bert", "zeroshot-gpt2", "probe-bert", "probe-gpt2"):
        ap.add_argument(f"--{name}", default="", help=f"{name} fold-analysis.json")
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
        for method in ("zeroshot", "probe"):
            analysis = _load(getattr(args, f"{method}_{arch}"))
            if analysis is None:
                print(f"  {arch}/{method}: 入力が無いので飛ばす")
                continue
            rows += run_rows(analysis, probe_json, arch, method)
        probe_analysis = _load(getattr(args, f"probe_{arch}"))
        rows += reference_rows(probe_analysis, _load(getattr(args, f"untrained_{arch}")), arch)

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kind", "subset", "family", "arch", "method", "fold",
                         "auroc", "checkpoint_step", "note"])
        writer.writerows(rows)
    print(f"  wrote {args.out_tsv}  ({len(rows):,} rows)")

    summary = summarise(rows)
    if args.out_md:
        open(args.out_md, "w").write(markdown(rows, summary))
        print(f"  wrote {args.out_md}")

    counts = {}
    for kind, _s, _f, arch, method, *_ in rows:
        counts[(kind, arch, method)] = counts.get((kind, arch, method), 0) + 1
    print(f"\n  {'kind':12s} {'arch':6s} {'method':9s} 行数")
    for key in sorted(counts):
        print(f"  {key[0]:12s} {key[1]:6s} {key[2]:9s} {counts[key]:>4d}")


if __name__ == "__main__":
    main()
