#!/usr/bin/env python3
"""Every evaluation point of every molecule_nat_lang run, in one table.

A report that describes a curve has to ship the curve. This walks the run directories
rather than taking a list, so a run that exists cannot be left out by a typo, and it
keeps the ones that diverged and the ones that were stopped -- those are the rows a
reader checks first.

Two things the rows have to carry beyond the loss, because the columns are not comparable
without them:

  metric    eval_loss_mask scores the 20% of positions BERT masked; val_loss scores every
            token a GPT-2 run predicted; val_loss_full_split is a checkpoint re-scored on
            the whole validation split afterwards. Three different numbers.
  eval_set  whether the evaluation read a fixed set of positions or drew them again at
            every evaluation. The 12,000-step BERT grid redrew its mask; the 24,000-step
            grid holds it. nanoGPT resamples its validation batches. Values from the two
            kinds differ by more than the differences usually being argued about.

    python scripts/molnl_all_curves_tsv.py --artifact-root <dir> --out <file.tsv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

BERT_NAME = re.compile(r"^bert_(?P<size>small|medium|large|xl)_(?:(?P<sched>24k)_)?"
                       r"lr(?P<tag>[0-9p]+e\d)(?:_(?P<variant>seed\d+|warmup|clip|both))?$")
GPT2_GRID = re.compile(r"^gpt2_(?P<size>small|medium|large|xl)_(?P<iters>\d+)_"
                       r"lr(?P<tag>[0-9p]+e\d)(?:_seed(?P<seed>\d+))?$")
GPT2_LADDER = re.compile(r"^molecule_nat_lang-(?P<size>small|medium|large|xl|ex-large)$")
STEP_LINE = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", re.M)


def rate_of(tag):
    mantissa, exponent = tag.split("e")
    return float(mantissa.replace("p", ".")) * 10 ** -int(exponent)


def jobs_from_logs(log_dirs):
    """{run directory: job id}, read from each launcher's "outputs:" banner.

    The run directory does not record which job wrote it (BERT's segments.log aside), and
    the slurm log carries both: its name ends in the job id and its banner names the
    output directory.
    """
    # Keyed by the full output path. Three passes of the GPT-2 ladder write directories
    # whose basenames are identical (molecule_nat_lang-small in each), so a basename key
    # hands every pass whichever job happened to be read first.
    found = {}
    name_fallback = {}
    for log_dir in log_dirs:
        for path in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
            base = os.path.basename(path)
            job = base.rsplit("-", 1)[-1].removesuffix(".out")
            if not job.isdigit():
                continue
            try:
                with open(path, errors="ignore") as fh:
                    head = [fh.readline() for _ in range(40)]
            except OSError:
                continue
            for line in head:
                if line.startswith("outputs:"):
                    found.setdefault(line.split()[1].rstrip("/"), job)
                    break
            else:
                # The 2026-08-17 launcher printed no resolved output path. Its name
                # carries the size, and the tree it wrote is known per launcher.
                m = re.match(r"mc-molnl-(gpt2|s1|s2)-(small|medium|large|xl)-\d+\.out$", base)
                if m:
                    tree = {"gpt2": "gpt2-output", "s1": "gpt2-output-shuffled",
                            "s2": "gpt2-output-shuffled-s2"}[m.group(1)]
                    size = m.group(2)
                    if tree == "gpt2-output" and size == "xl":
                        size = "ex-large"
                    name_fallback.setdefault(f"{tree}/molecule_nat_lang-{size}", job)
    found.update({k: v for k, v in name_fallback.items() if k not in found})
    return found


def bert_rows(root, jobs):
    rows, seen = [], []
    for run_dir in sorted(glob.glob(os.path.join(root, "bert_*"))):
        name = os.path.basename(run_dir)
        m = BERT_NAME.match(name)
        if not m:
            continue
        cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                     key=lambda p: int(p.rsplit("-", 1)[1]))
        if not cks:
            continue
        state = json.load(open(os.path.join(cks[-1], "trainer_state.json")))
        points = sorted((e["step"], e["eval_loss_mask"]) for e in state["log_history"]
                        if "eval_loss_mask" in e)
        if not points:
            continue
        variant = m["variant"] or ""
        seed = int(variant[4:]) if variant.startswith("seed") else 42
        # The fixed mask arrived with the 24,000-step grid; before it every evaluation
        # drew its masked positions again.
        fixed = os.path.exists(os.path.join(run_dir, "run_manifest.json")) and \
            (json.load(open(os.path.join(run_dir, "run_manifest.json")))
             .get("eval", {}).get("fixed_mask", {}).get("enabled", False))
        job = jobs.get(run_dir) or jobs.get(name, "")
        if not job:
            log = os.path.join(run_dir, "segments.log")
            if os.path.exists(log):
                for line in open(log):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        job = parts[1]
        for step, value in points:
            rows.append((name, "bert", m["size"], f"{rate_of(m['tag']):g}", seed, step,
                         f"{value:.6f}", "eval_loss_mask", job,
                         "fixed" if fixed else "resampled"))
        seen.append(name)
    return rows, seen


def gpt2_rows(roots, jobs):
    rows, seen = [], []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(run_dir):
                continue
            name = os.path.basename(run_dir)
            grid, ladder = GPT2_GRID.match(name), GPT2_LADDER.match(name)
            if not (grid or ladder):
                continue
            series = {}
            for path in glob.glob(os.path.join(run_dir, "logging_*.csv")):
                for row in csv.reader(open(path)):
                    if row and row[0].strip().isdigit():
                        series[int(row[0])] = float(row[2])
            if not series:
                continue
            if grid:
                size, lr = grid["size"], f"{rate_of(grid['tag']):g}"
                seed = int(grid["seed"] or 42)
            else:
                size = "xl" if ladder["size"] == "ex-large" else ladder["size"]
                # The ladder predates the grid: its rate is the one in gpt2_<size>.py,
                # and ex-large is what the 2026-08-17 tree calls xl.
                lr = {"small": "0.0006", "medium": "0.0003",
                      "large": "0.00025", "xl": "0.0002"}[size]
                seed = {"small": 66, "medium": 55, "large": 60, "xl": 68}.get(size, 0)
            # The run directory distinguishes the three passes; the name alone does not.
            label = name if grid else f"{os.path.basename(root)}/{name}"
            job = jobs.get(run_dir) or jobs.get(label, "")
            for step, value in sorted(series.items()):
                rows.append((label, "gpt2", size, lr, seed, step, f"{value:.6f}",
                             "val_loss", job, "resampled"))
            seen.append(label)
    return rows, seen


def rescored_rows(path, iters, jobs_hint):
    rows, seen = [], []
    if not os.path.exists(path):
        return rows, seen
    scores = json.load(open(path))
    for name, row in sorted(scores["results"].items()):
        m = GPT2_GRID.match("gpt2_" + name)
        if not m:
            continue
        rows.append((f"gpt2_{name}", "gpt2", m["size"], f"{rate_of(m['tag']):g}",
                     int(m["seed"] or 42), row.get("checkpoint_iter") or iters,
                     f"{row['loss']:.6f}", "val_loss_full_split", jobs_hint, "fixed"))
        seen.append(f"gpt2_{name}:rescored")
    return rows, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-root", required=True,
                    help="the molecule_nat_lang directory holding bert-output*/ and gpt2-output*/")
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--rescored", default=None, help="the re-scoring JSON")
    ap.add_argument("--rescore-job", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    jobs = jobs_from_logs(a.log_dirs)
    rows, seen = bert_rows(os.path.join(a.artifact_root, "bert-output-shuffled"), jobs)
    g_rows, g_seen = gpt2_rows(
        [os.path.join(a.artifact_root, d) for d in
         ("gpt2-output-grid", "gpt2-output", "gpt2-output-shuffled", "gpt2-output-shuffled-s2")],
        jobs)
    r_rows, r_seen = rescored_rows(a.rescored, 1500, a.rescore_job) if a.rescored else ([], [])
    rows += g_rows + r_rows
    seen += g_seen + r_seen

    columns = ["run", "arch", "size", "lr", "seed", "step", "value", "metric", "job", "eval_set"]
    with open(a.out, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")

    missing = sorted({r[0] for r in rows if not r[8]})
    print(f"  {len(seen)} runs, {len(rows):,} rows -> {a.out}")
    print(f"  rows without a job id: {sum(1 for r in rows if not r[8]):,}"
          + (f" (runs: {', '.join(missing)})" if missing else ""))
    for arch in ("bert", "gpt2"):
        names = sorted({r[0] for r in rows if r[1] == arch})
        print(f"  {arch}: {len(names)} runs")
        for n in names:
            pts = [r for r in rows if r[0] == n]
            print(f"      {n:<42}{len(pts):>5} points  last step {pts[-1][5]:>7}  "
                  f"{pts[-1][7]}  {pts[-1][9]}  job {pts[-1][8] or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
