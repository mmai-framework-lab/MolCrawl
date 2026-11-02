"""Training and validation loss for the four compounds BERT arms, step 1 to 45,000.

One row per (step, split), the eight columns of the 2026-10-20 curve table:
``arch size lr seed step split value source``. ``split`` is ``train`` or
``valid`` and the same step normally carries both.

The three stages are stitched here rather than taken from one file. The HF
Trainer restores ``log_history`` on resume, so the furthest checkpoint of the
45,000-step stage already holds all three stages' points -- but a run restored
from a copied checkpoint is exactly the case where that history could be the
wrong one, so each stage's own ``trainer_state.json`` supplies the steps that
stage trained and ``source`` names it. What the later stages carry for an
earlier stage's steps is then compared against it, and any disagreement is
reported rather than silently preferred.

``lr`` is the stage-1 peak, which is how the ladder names an arm. It is not the
rate the later stages ran at: stages 2 and 3 hold a tenth of it. The three-stage
shape is in the configs and in scripts/lr_schedule_end_table.py; a reader who
wants the rate per step needs those, not this file.

The training loss is not comparable across sizes -- it is a running mean over
whichever windows the interval covered -- and is here for the shape of the curve.
The validation metric is ``eval_loss_mask``, the [MASK] positions alone, which is
the value the ladder adopts.

    compounds_trainval_curve_tsv.py --bert-root DIR --out FILE

Read-only. The root is an argument: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

COLUMNS = ["arch", "size", "lr", "seed", "step", "split", "value", "source"]

SEED = "42"
EVAL_INTERVAL = 100  # log_interval in every one of the twelve stage configs

# (size as the ladder labels it, stage-1 peak rate as the config writes it,
#  run directory of stage 1 -- the later stages add the suffixes below)
#
# The directory says "xl" and the ladder says "xlarge". Both are kept: the
# directory is what is on disk and the label is what the table is read with.
ARMS = [
    ("small", "1e-3", "compounds-small-lr1e3-seed42"),
    ("medium", "2.8e-4", "compounds-medium-lr2p8e4-seed42"),
    ("large", "1.5e-4", "compounds-large-lr1p5e4-seed42"),
    ("xlarge", "4.5e-5", "compounds-xl-lr4p5e5-seed42"),
]

# (stage name, directory suffix, first step this stage trained, last step)
STAGES = [
    ("stage1", "", 1, 15_000),
    ("ext30k", "-ext30k", 15_001, 30_000),
    ("ext45k", "-ext45k", 30_001, 45_000),
]


def newest_trainer_state(run: Path) -> Path:
    """The trainer_state.json of the furthest checkpoint, which holds the history."""
    steps = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir())
    if not steps:
        raise SystemExit(f"no checkpoint-* directory under {run}")
    path = run / f"checkpoint-{steps[-1]}" / "trainer_state.json"
    if not path.exists():
        raise SystemExit(f"no trainer_state.json in {path.parent}")
    return path


def read_points(path: Path, val_metric: str) -> dict[tuple[int, str], float]:
    """{(step, split): value} out of one log_history.

    A step logged twice keeps the last value, which is what a resume writes: the
    point is reported as a duplicate by the caller rather than hidden here.
    """
    state = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, str], float] = {}
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            out[(int(step), "train")] = float(entry["loss"])
        if val_metric in entry:
            out[(int(step), "valid")] = float(entry[val_metric])
    return out


def duplicate_steps(path: Path, val_metric: str) -> Counter:
    """How many times each (step, split) appears, for the ones appearing twice."""
    state = json.loads(path.read_text(encoding="utf-8"))
    seen: Counter = Counter()
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            seen[(int(step), "train")] += 1
        if val_metric in entry:
            seen[(int(step), "valid")] += 1
    return Counter({k: n for k, n in seen.items() if n > 1})


def saved_steps(run: Path) -> list[int]:
    """The checkpoints still on disk. save_total_limit prunes, so this is what is
    retained now, not every step that was ever written."""
    return sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert-root", type=Path, required=True,
                    help="directory holding the compounds-*-seed42* run directories")
    ap.add_argument("--val-metric", default="eval_loss_mask")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows: list[dict] = []
    problems = 0

    for size, lr, base in ARMS:
        print(f"\n=== bert {size}  peak lr {lr}  seed {SEED}")
        native: dict[tuple[int, str], tuple[float, str]] = {}
        histories: dict[str, dict[tuple[int, str], float]] = {}

        for stage, suffix, lo, hi in STAGES:
            run = args.bert_root / (base + suffix)
            state = newest_trainer_state(run)
            points = read_points(state, args.val_metric)
            histories[stage] = points
            source = f"{run.name}/{state.parent.name}/{state.name}"

            kept = {k: v for k, v in points.items() if lo <= k[0] <= hi}
            for key, value in kept.items():
                native[key] = (value, source)

            steps = sorted({s for s, _ in kept})
            dups = duplicate_steps(state, args.val_metric)
            print(f"  {stage:7s} {source}")
            print(f"          history {len(points):5d} 点 / この段の {lo:,}-{hi:,} に "
                  f"{len(kept):5d} 点 ({len(steps)} step)")
            print(f"          step {min(steps):,} .. {max(steps):,}"
                  f"   save 済み checkpoint {saved_steps(run)}")
            if dups:
                problems += 1
                print(f"          ** 同じ step が 2 回: {sorted(dups)[:6]}")

        # What the later stages carry for an earlier stage's steps must agree with
        # the stage that trained them, or the stitch is joining two different runs.
        for stage, _suffix, lo, hi in STAGES:
            for later, _s2, lo2, _hi2 in STAGES:
                if lo2 <= hi:
                    continue
                shared = [k for k in histories[later] if lo <= k[0] <= hi and k in histories[stage]]
                if not shared:
                    continue
                worst = max(abs(histories[later][k] - histories[stage][k]) for k in shared)
                verdict = "一致" if worst == 0 else f"** 最大差 {worst:.3e}"
                if worst != 0:
                    problems += 1
                print(f"  照合  {later} が持つ {stage} の {len(shared)} 点: {verdict}")

        steps = sorted({s for s, _ in native})
        expected = list(range(EVAL_INTERVAL, 45_000 + 1, EVAL_INTERVAL))
        missing = [s for s in expected if s not in steps]
        extra = [s for s in steps if s % EVAL_INTERVAL]
        if missing:
            problems += 1
            print(f"  ** {EVAL_INTERVAL} step 刻みの {len(expected)} 点のうち "
                  f"{len(missing)} 点が無い: {missing[:8]}")
        else:
            print(f"  刻み  {EVAL_INTERVAL} step x {len(expected)} 点、欠けなし")
        if extra:
            print(f"  刻み外 {len(extra)} step: {extra[:8]}")

        for step in steps:
            have = [sp for sp in ("train", "valid") if (step, sp) in native]
            for split in have:
                value, source = native[(step, split)]
                note = source
                if len(have) == 1:
                    other = "valid" if split == "train" else "train"
                    note = f"{source}; {other} not recorded at this step"
                rows.append({"arch": "bert", "size": size, "lr": lr, "seed": SEED,
                             "step": step, "split": split, "value": f"{value:.6f}",
                             "source": note})
        singles = [s for s in steps
                   if ((s, "train") in native) != ((s, "valid") in native)]
        print(f"  片側だけの step {len(singles)} 件" + (f": {singles[:8]}" if singles else ""))

        # The joins. Both sides are printed so that a step across a boundary can be
        # read without opening the file.
        for boundary in (15_000, 30_000):
            for split in ("train", "valid"):
                before = native.get((boundary, split))
                after = native.get((boundary + EVAL_INTERVAL, split))
                if before and after:
                    print(f"  つなぎ {boundary:,} -> {boundary + EVAL_INTERVAL:,} "
                          f"{split:5s} {before[0]:.6f} -> {after[0]:.6f}"
                          f"  ({before[1].split('/')[0]} -> {after[1].split('/')[0]})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8 without a BOM and CRLF, which is what csv writes by default and what
    # compounds-loss-curves-20261020.tsv is. Every field here is ASCII.
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    per_arm = Counter((r["size"], r["split"]) for r in rows)
    print(f"\n{len(rows):,} 行 -> {args.out}")
    for key in sorted(per_arm):
        print(f"  {key[0]:8s} {key[1]:6s} {per_arm[key]:5d} 行")
    print(f"報告すべき不一致 {problems} 件")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
