"""The arms the 2026-11-27b directive adds to the compounds batch-size grid.

The grid ran two or three rates in each (size, batch) cell. Where the rate that gave
the cell's best sat at an end of what that cell swept, the cell's optimum may lie
outside the swept range, and the directive extends the cell in that direction by three
more rates at the same 1.4 ratio the grid already uses:

    the best came from the lowest rate   ->  multipliers 0.5, 0.35, 0.25
    the best came from the highest rate  ->  1.4, 2.0, 2.8   (cell swept two rates)
                                        ->  2.0, 2.8, 4.0   (cell swept three)
    the best came from the middle        ->  nothing added

Which cell is which is not copied from the directive: it is read off the measured
per-cell table (``compounds_batch_sweep_cells.py``) and then compared against the
directive's printed list, so a cell the two disagree about is reported rather than
silently resolved. xl is left out because its arms are still running and no cell of it
has a settled best yet.

Everything else about an added arm is its cell's existing arm: the batch, the budget in
blocks, the step count, the evaluation and save intervals, the warmup, the seed and one
GPU's block count. Those are checked against the 68-arm inventory rather than
re-derived here, which is what makes the added arm comparable with the arms it will be
tabulated beside.

Two values are new, and only the added arms carry them (2026-11-27 judgment §3 and §4):

    degenerate_loss_threshold = 2.4733   the [MASK]-position unigram loss
    degenerate_grace_steps               11,520,000 blocks, 10 % of the budget

    compounds_batch_sweep_added_arms.py --cells rt1-...cells-bert-DATE.tsv \
        --arms rt1-compounds-batch-sweep-arms-2026-11-20b.tsv --out rt1-...tsv

Read-only: it writes one TSV and launches nothing.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compounds_batch_sweep_arms import (  # noqa: E402
    BERT_EVAL_BATCH,
    BUDGET_ROWS,
    COLUMNS,
    JUDGE_METRIC,
    SEQUENCE_LENGTH,
    accumulation,
    interval,
    lr_slug,
    scaled_rate,
    steps_for,
    warmup_for,
)
from report_tsv import check_name, stamp, write_tsv  # noqa: E402

ARCH = "bert"

#: The ratio the grid's own rates step by. The directive continues it rather than
#: introducing a second spacing, so a cell's rates stay one geometric series.
RATIO = 1.4

#: §6. Three rates below the cell's lowest, at the same ratio: 0.7 / 1.4 = 0.5, and so
#: on. Written out rather than computed so the file states the numbers the directive
#: printed.
DOWN_MULTIPLIERS = (0.5, 0.35, 0.25)
#: Three rates above the cell's highest. A cell that swept two rates tops out at 1.0, a
#: cell that swept three at 1.4, so the continuation starts from a different place.
UP_MULTIPLIERS = {2: (1.4, 2.0, 2.8), 3: (2.0, 2.8, 4.0)}

#: §6. xl is not extended: its arms are at 60.4-64.4 % of the budget, so which rate is
#: best in any of its cells is not decided. The directive says it will be ordered
#: separately once the 11 running arms finish.
SIZES_HELD_BACK = {"xl": "still running; best rate not decided"}

#: 2026-11-27 judgment §3, as the 11-27b directive settles it (option a): the threshold
#: moves to the [MASK]-position unigram loss, written to the four figures the directive
#: prints. The measurement is 2.4732663515556292
#: (report/mlm-baselines-20261102/full/compounds.json, valid split, 10,000 rows,
#: 1,595,521 masked positions, mlm_probability 0.2).
DEGENERATE_THRESHOLD = 2.4733
#: The measurement the figure above is rounded from, for the record.
DEGENERATE_THRESHOLD_MEASURED = 2.4732663515556292

#: 2026-11-27 judgment §4: the grace period is a number of blocks, not a number of
#: steps, so every batch waits over the same amount of data. 10 % of the 115,200,000
#: block budget.
GRACE_ROWS = 11_520_000

#: The directive's printed step counts for that grace period.
EXPECTED_GRACE = {256: 45_000, 512: 22_500, 1024: 11_250, 2048: 5_625, 4096: 2_812}

#: §6's printed tables. Nothing below reads these except check().
EXPECTED_ADDED_LR = {
    ("small", 256): ["1.58e-4", "1.11e-4", "7.91e-5"],
    ("small", 512): ["2.24e-4", "1.57e-4", "1.12e-4"],
    ("small", 1024): ["3.16e-4", "2.21e-4", "1.58e-4"],
    ("small", 2048): ["4.47e-4", "3.13e-4", "2.24e-4"],
    ("small", 4096): ["6.32e-4", "4.43e-4", "3.16e-4"],
    ("large", 256): ["2.37e-5", "1.66e-5", "1.19e-5"],
    ("large", 512): ["3.35e-5", "2.35e-5", "1.68e-5"],
    ("medium", 256): ["1.24e-4", "1.77e-4", "2.48e-4"],
    ("medium", 512): ["1.75e-4", "2.50e-4", "3.51e-4"],
    ("medium", 1024): ["2.48e-4", "3.54e-4", "4.96e-4"],
    ("large", 1024): ["1.33e-4", "1.90e-4", "2.66e-4"],
    ("medium", 2048): ["5.01e-4", "7.01e-4", "1.00e-3"],
    ("medium", 4096): ["7.08e-4", "9.92e-4", "1.42e-3"],
    ("large", 4096): ["3.79e-4", "5.31e-4", "7.59e-4"],
}
#: §6's direction for each of those cells, and the one cell it holds back by name.
EXPECTED_DIRECTION = {
    ("small", 256): "down", ("small", 512): "down", ("small", 1024): "down",
    ("small", 2048): "down", ("small", 4096): "down",
    ("large", 256): "down", ("large", 512): "down",
    ("medium", 256): "up", ("medium", 512): "up", ("medium", 1024): "up",
    ("medium", 2048): "up", ("medium", 4096): "up",
    ("large", 1024): "up", ("large", 4096): "up",
    ("large", 2048): "none",
}
EXPECTED_ARMS = 42

#: The columns of the 68-arm inventory, plus what only an added arm carries.
ADDED_COLUMNS = COLUMNS + ["direction", "degenerate_threshold",
                           "degenerate_grace_steps", "arch_arms"]

#: BERT arms in the grid once these are added: the 48 that ran plus these 42. It goes
#: into the inventory because the configs quote it when they say what the fixed
#: evaluation set is held identical across.
BERT_ARMS_AFTER = 90

#: The inventory values an added arm must share with its cell's existing arms (§6).
SHARED_WITH_CELL = ["batch", "steps", "warmup_steps", "eval_interval", "save_interval",
                    "rows", "tokens", "micro_batch", "grad_accum", "world_size",
                    "expected_global_batch", "eval_batch", "micro_batch_set"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def grace_for(batch: int) -> int:
    """§4: the grace period in steps, from a fixed number of blocks."""
    return int(GRACE_ROWS / batch)


def direction_of(cell) -> str:
    """Which way a cell is extended, from where its best rate sat (§6)."""
    position = cell.get("lr_position", "")
    if position == "lowest":
        return "down"
    if position == "highest":
        return "up"
    return "none"


def multipliers_for(cell, direction: str):
    if direction == "down":
        return DOWN_MULTIPLIERS
    if direction == "up":
        swept = int(cell["lrs_tried"])
        if swept not in UP_MULTIPLIERS:
            raise SystemExit(f"{cell['size']} b{cell['batch']}: {swept} rates swept, "
                             f"and no continuation is defined for that")
        return UP_MULTIPLIERS[swept]
    return ()


def added_arms(cells):
    """One row per added arm, in the order the cells are listed."""
    rows = []
    for cell in cells:
        size, batch = cell["size"], int(cell["batch"])
        if size in SIZES_HELD_BACK:
            continue
        direction = direction_of(cell)
        if direction == "none":
            continue
        steps = steps_for(ARCH, batch)
        # The shape the grid runs on; the check below proves it is the same one the
        # cell's existing arms were derived with.
        micro, accum, world = accumulation(ARCH, size, batch, "micro-order")
        for multiplier in multipliers_for(cell, direction):
            lr = scaled_rate(ARCH, size, batch, multiplier)
            slug = lr_slug(lr)
            stem = f"{ARCH}_{size}_b{batch}_lr{slug}"
            rows.append({
                "arch": ARCH, "size": size, "batch": batch, "lr": f"{lr:.2e}",
                "lr_multiplier": f"{multiplier:.2f}", "lr_slug": slug,
                "steps": steps,
                "warmup_steps": warmup_for(ARCH, steps),
                "eval_interval": interval(ARCH, batch, "eval"),
                "save_interval": interval(ARCH, batch, "save"),
                "rows": BUDGET_ROWS[ARCH],
                "tokens": BUDGET_ROWS[ARCH] * SEQUENCE_LENGTH,
                "micro_batch": micro, "grad_accum": accum, "world_size": world,
                "expected_global_batch": batch,
                "eval_batch": BERT_EVAL_BATCH,
                "min_lr": None, "lr_decay_iters": None,
                "judge_metric": JUDGE_METRIC[ARCH],
                "micro_batch_set": "micro-order",
                "config": f"molcrawl/tasks/pretrain/configs/compounds/{stem}.py",
                "out_dir": f"compounds-{size}-b{batch}-lr{slug}",
                "direction": direction,
                "degenerate_threshold": DEGENERATE_THRESHOLD,
                "degenerate_grace_steps": grace_for(batch),
                "arch_arms": BERT_ARMS_AFTER,
            })
    return rows


def check(rows, cells, existing) -> int:
    """Compare the derivation against the directive's tables and the running arms."""
    problems = 0

    if len(rows) != EXPECTED_ARMS:
        problems += 1
        print(f"  ** {len(rows)} arms against directive {EXPECTED_ARMS}")

    # The direction of every cell, including the ones nothing is added to.
    for cell in cells:
        key = (cell["size"], int(cell["batch"]))
        if cell["size"] in SIZES_HELD_BACK:
            if key in EXPECTED_DIRECTION:
                problems += 1
                print(f"  ** {key}: held back here, but the directive lists it")
            continue
        got = direction_of(cell)
        want = EXPECTED_DIRECTION.get(key)
        if want is None:
            problems += 1
            print(f"  ** {key}: measured {got}, and the directive does not list it")
        elif got != want:
            problems += 1
            print(f"  ** {key}: measured {got} against directive {want}")

    # The rates themselves.
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["size"], int(row["batch"])), []).append(row)
    for key, want in EXPECTED_ADDED_LR.items():
        got = by_cell.get(key, [])
        if len(got) != len(want):
            problems += 1
            print(f"  ** {key}: {len(got)} rates against directive {len(want)}")
            continue
        for row, printed in zip(got, want):
            if float(row["lr"]) != float(printed):
                problems += 1
                print(f"  ** {key}: {row['lr']} against directive {printed}")
    for key in by_cell:
        if key not in EXPECTED_ADDED_LR:
            problems += 1
            print(f"  ** {key}: arms added here, and the directive does not list it")

    # A new rate must not be one the cell already ran.
    ran = {(r["size"], r["batch"], f"{float(r['lr']):.2e}") for r in existing
           if r["arch"] == ARCH}
    for row in rows:
        if (row["size"], str(row["batch"]), row["lr"]) in ran:
            problems += 1
            print(f"  ** {row['size']} b{row['batch']} {row['lr']}: already ran")

    # The grace period, in steps.
    for batch, want in EXPECTED_GRACE.items():
        got = grace_for(batch)
        if got != want:
            problems += 1
            print(f"  ** grace at batch {batch}: {got} against directive {want}")

    # Everything else is the cell's existing arm.
    for row in rows:
        siblings = [r for r in existing if r["arch"] == ARCH
                    and r["size"] == row["size"]
                    and int(r["batch"]) == int(row["batch"])]
        if not siblings:
            problems += 1
            print(f"  ** {row['size']} b{row['batch']}: no existing arm to copy from")
            continue
        for column in SHARED_WITH_CELL:
            want = {str(r[column]) for r in siblings}
            if len(want) != 1:
                problems += 1
                print(f"  ** {row['size']} b{row['batch']}: existing arms disagree on "
                      f"{column}: {sorted(want)}")
                continue
            if str(row[column]) != want.pop():
                problems += 1
                print(f"  ** {row['size']} b{row['batch']} {row['lr']}: {column}="
                      f"{row[column]} against the cell's existing arms")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", required=True, help="the measured per-cell table")
    ap.add_argument("--arms", required=True, help="the 68-arm inventory")
    ap.add_argument("--out", required=True, help="the TSV to write")
    args = ap.parse_args(argv)

    cells = [r for r in read_rows(args.cells)]
    existing = read_rows(args.arms)
    rows = added_arms(cells)

    print(f"=== {len(rows)} added arms from {args.cells} ===")
    print(f"{'size':<7}{'batch':>6}{'dir':>6}{'mult':>7}{'lr':>11}"
          f"{'steps':>9}{'warmup':>8}{'grace':>8}{'threshold':>11}")
    for row in rows:
        print(f"{row['size']:<7}{row['batch']:>6}{row['direction']:>6}"
              f"{row['lr_multiplier']:>7}{row['lr']:>11}{row['steps']:>9}"
              f"{row['warmup_steps']:>8}{row['degenerate_grace_steps']:>8}"
              f"{row['degenerate_threshold']:>11}")

    print("\n=== against the directive's tables and the running arms ===")
    problems = check(rows, cells, existing)
    print(f"disagreements: {problems}")

    write_tsv(check_name(args.out), ADDED_COLUMNS, rows, {
        "model": "compounds bert batch-size grid, the 2026-11-27b additions",
        "source": f"{args.cells}; {args.arms}",
        "generated": stamp("scripts/compounds_batch_sweep_added_arms.py"),
        "note": (f"lr = the 2,560 arm's rate x sqrt(batch/2560) x the column's "
                 f"multiplier, to three significant figures; the multipliers continue "
                 f"the cell's own series at ratio {RATIO}. degenerate_grace_steps = "
                 f"{GRACE_ROWS:,} blocks / batch (judgment §4); "
                 f"degenerate_threshold {DEGENERATE_THRESHOLD} is the [MASK]-position "
                 f"unigram loss {DEGENERATE_THRESHOLD_MEASURED} to four figures "
                 f"(judgment §3). Added arms only: the 68 running arms keep 2.395 and "
                 f"main.py's 5,000-step grace."),
    })
    print(f"\n{len(rows)} row(s) -> {args.out}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
