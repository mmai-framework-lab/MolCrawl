#!/usr/bin/env python3
"""Curves and the numbers behind them for the molecule_nat_lang GPT-2 ladder.

The 2026-09-24 order asks for the GPT-2 runs in the same form as the BERT grid. Three
things make them a different picture, and the figures keep them apart:

- there is one learning rate per size, not a grid, so there is nothing to overlay per
  learning rate. The size's rate is written into its label.
- the ladder ran three times: once on the corpus before it was shuffled, once on the
  shuffled corpus, and once on the shuffled corpus with a 3,000-iteration schedule
  instead of 373. Each pass gets its own axes; their schedules are not the same length.
- the metric is nanoGPT's validation loss over whole sequences, not BERT's
  eval_loss_mask over masked positions. No floor is drawn, because the unigram floor
  measured for this corpus (3.8638, job 22503) was measured on masked positions and is
  not the reference for next-token loss. None has been measured for this one.

    python scripts/molnl_gpt2_ladder_curves.py --runs-root <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molnl_loss_figures import (SIZE_COLOUR, axes, best_point, label_bests,  # noqa: E402
                                new_figure, save)

# Facts the run directories do not carry. Each comes from the job that wrote the tree,
# in workflows/slurm-logs/ -- the launcher passes --max_iters and --eval_interval, so the
# config's own max_iters (373) is not by itself what ran.
#   directory                job ids     what the log's "Overriding:" lines say
class Stage(NamedTuple):
    tree: str        # the directory the pass wrote
    slug: str        # what its figure is called
    label: str       # what it is called in a caption
    job: int         # the slurm job whose log the schedule was read from
    iters: int       # --max_iters as that log's "Overriding:" lines report it
    every: int       # --eval_interval, likewise


STAGES = (
    Stage("gpt2-output", "before-shuffle", "before the shuffle", 28845, 373, 25),
    Stage("gpt2-output-shuffled", "shuffled", "shuffled corpus", 33746, 373, 25),
    Stage("gpt2-output-shuffled-s2", "shuffled-long", "shuffled corpus, long schedule", 35218, 3000, 100),
)
GLOBAL_BATCH = 2560          # sequences; batch_size * gradient_accumulation_steps
BLOCK = 1024                 # tokens per sequence
TRAIN_BLOCKS = 318118        # mol_nl train split, 325,752,832 tokens
# The tree written on 2026-08-17 calls the largest size ex-large; the job that wrote it
# was launched as xl and read gpt2_xl.py (job 28849's out_dir).
CONFIG_OF = {"small": "gpt2_small", "medium": "gpt2_medium", "large": "gpt2_large",
             "xl": "gpt2_xl", "ex-large": "gpt2_xl"}
SIZE_ORDER = ("small", "medium", "large", "xl", "ex-large")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "molcrawl", "tasks", "pretrain", "configs", "molecule_nat_lang")


def learning_rate_of(size):
    """The size's peak learning rate, read from the config rather than written here.

    The configs assign it literally, so the line is read directly: importing them pulls
    in the dataset paths, which this has no business resolving.
    """
    path = os.path.join(CONFIG_DIR, CONFIG_OF[size] + ".py")
    for line in open(path):
        m = re.match(r"^learning_rate\s*=\s*([0-9.eE+-]+)", line)
        if m:
            return float(m.group(1))
    return None


def read_run(run_dir):
    """(steps, val, train) from the run's own csv log -- the one nanoGPT writes itself.

    A tree can hold more than one: a launch that died before its first evaluation leaves
    a header-only file beside the real one. The longest is the run.
    """
    best = []
    for path in glob.glob(os.path.join(run_dir, "logging_*.csv")):
        rows = [r for r in csv.reader(open(path)) if r and r[0].strip().isdigit()]
        if len(rows) > len(best):
            best = rows
    steps = [int(r[0]) for r in best]
    return steps, [float(r[2]) for r in best], [float(r[1]) for r in best]


def collect(runs_root):
    runs = {}
    for stage in STAGES:
        for d in sorted(glob.glob(os.path.join(runs_root, stage.tree, "molecule_nat_lang-*"))):
            size = os.path.basename(d).split("molecule_nat_lang-", 1)[1]
            steps, val, train = read_run(d)
            if steps:
                runs[(stage.tree, size)] = {"steps": steps, "val": val, "train": train,
                                            "stage": stage}
    return runs


def preconditions(iters, every):
    tokens = iters * GLOBAL_BATCH * BLOCK
    return (f"max_iters {iters:,} ({iters * GLOBAL_BATCH / TRAIN_BLOCKS:.2f} epochs) · "
            f"train {TRAIN_BLOCKS:,} sequences of {BLOCK:,} tokens · "
            f"global batch {GLOBAL_BATCH:,} sequences · {tokens / 1e9:.2f} G tokens seen · "
            f"metric val loss, evaluated every {every} iterations")


def fig_stage(runs, out_dir):
    """One figure per pass over the corpus, sizes overlaid."""
    made = []
    for stage in STAGES:
        arms = [(size, runs[(stage.tree, size)]) for size in SIZE_ORDER
                if (stage.tree, size) in runs]
        if not arms:
            continue
        fig, ax = new_figure()
        items = []
        for size, run in arms:
            lr = learning_rate_of(size)
            ax.plot(run["steps"], run["val"], color=SIZE_COLOUR[size], linewidth=1.3)
            bx, by = best_point(run["steps"], run["val"])
            items.append((bx, by, f"{size} lr {lr:g}: {by:.4f} @ {bx:,}", SIZE_COLOUR[size]))
        ax.set_yscale("log")   # the first evaluation is near 11 and the last near 0.5
        axes(ax, "val loss", xlabel="iteration")
        label_bests(ax, items, room=0.40)
        note = (f"{preconditions(stage.iters, stage.every)}\n{stage.label}; "
                f"slurm job {stage.job} and its siblings.")
        reached = max(max(r["steps"]) for _, r in arms)
        if reached < stage.iters:
            note += (f" The last evaluation is at {reached:,} rather than {stage.iters:,}: "
                     f"evaluation happens on multiples of {stage.every}.")
        short = [size for size, r in arms if max(r["steps"]) < reached]
        if short:
            note += (" Stopped early, after 10 evaluations without improvement: "
                     + ", ".join(f"{size} at {max(dict(arms)[size]['steps']):,}" for size in short) + ".")
        made.append(save(fig, ax, f"molecule_nat_lang GPT-2 ladder - {stage.label}", note,
                         os.path.join(out_dir, f"gpt2-{stage.slug}.png")))
    return made


def fig_tail(runs, out_dir, frac=0.20):
    """The last 20% of the longest schedule: does the best sit at the end?"""
    stage = max(STAGES, key=lambda s: s.iters)
    arms = [(size, runs[(stage.tree, size)]) for size in SIZE_ORDER if (stage.tree, size) in runs]
    if not arms:
        return []
    cut = max(max(r["steps"]) for _, r in arms) * (1 - frac)
    fig, ax = new_figure()
    items = []
    for size, run in arms:
        pts = [(s, v) for s, v in zip(run["steps"], run["val"]) if s >= cut]
        ax.plot([s for s, _ in pts], [v for _, v in pts], color=SIZE_COLOUR[size], linewidth=1.3)
        bx, by = best_point([s for s, _ in pts], [v for _, v in pts])
        items.append((bx, by, f"{size}: {by:.4f} @ {bx:,}", SIZE_COLOUR[size]))
    axes(ax, "val loss", xlabel="iteration")
    label_bests(ax, items, room=0.36)
    note = (f"{preconditions(stage.iters, stage.every)}\nLast {int(frac * 100)}% of the longest "
            f"schedule ({stage.label}). Every size ends worse than its own best, and xl's run stopped at 2,900 "
            "on 10 evaluations without improvement: the schedule runs past where these runs peak. "
            "Marked points are the best inside this window, which for medium and xl is not the run's best.")
    return [save(fig, ax, f"molecule_nat_lang GPT-2 ladder - last {int(frac * 100)}% of the long schedule",
                 note, os.path.join(out_dir, "gpt2-tail-last20pct.png"))]


def write_tsv(runs, out_dir):
    """The table, for whoever recomputes from it. Pure ASCII, comment lines included."""
    path = os.path.join(out_dir, "molnl-gpt2-ladder-val-loss.tsv")
    with open(path, "w") as fh:
        fh.write("# molecule_nat_lang GPT-2 ladder, val loss at every evaluation\n")
        fh.write(f"# train {TRAIN_BLOCKS:,} sequences of {BLOCK:,} tokens | "
                 f"global batch {GLOBAL_BATCH:,} sequences | metric val loss (whole sequence)\n")
        fh.write("# pass: which run over the corpus; see the figures for each one's schedule\n")
        fh.write("pass\tsize\tlearning_rate\tmax_iters\tstep\tval_loss\ttrain_loss\n")
        for stage in STAGES:
            for size in SIZE_ORDER:
                run = runs.get((stage.tree, size))
                if not run:
                    continue
                lr = learning_rate_of(size)
                for s, v, t in zip(run["steps"], run["val"], run["train"]):
                    fh.write(f"{stage.label}\t{size}\t{lr:g}\t{stage.iters}\t{s}"
                             f"\t{v:.6f}\t{t:.6f}\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True, help="directory holding gpt2-output*/")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    runs = collect(a.runs_root)
    if not runs:
        print(f"no runs under {a.runs_root}")
        return 1
    made = fig_stage(runs, a.out_dir) + fig_tail(runs, a.out_dir)
    tsv = write_tsv(runs, a.out_dir)
    print(f"runs read: {len(runs)}")
    for p in made + [tsv]:
        print(f"  {os.path.basename(p)}  {os.path.getsize(p):,} bytes")
    print(f"\n  {'pass':<32}{'size':<10}{'lr':>8}{'best val':>10}{'at':>8}{'last':>10}")
    for stage in STAGES:
        for size in SIZE_ORDER:
            run = runs.get((stage.tree, size))
            if not run:
                continue
            bx, by = best_point(run["steps"], run["val"])
            print(f"  {stage.label:<32}{size:<10}{learning_rate_of(size):>8g}{by:>10.4f}{bx:>8,}"
                  f"{run['val'][-1]:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
