"""Derive the molecule_nat_lang batch-size grid's arms, and check the directive's tables.

The 2026-11-20 directive prints the step counts, the evaluation and save intervals, the
warmup lengths and the 68 learning rates it wants. All of them follow from rules the
same directive states:

    a fixed number of 1,024-token blocks, so the step count is blocks / batch (§2)
    intervals fixed in blocks, so every arm's points land at the same block
      position -- BERT every 122,880 blocks and saves every 614,400, GPT-2 every
      40,960 (§2)
    the rate scaled by sqrt(batch / 2560) off the existing 2,560 arm, times the
      column's multiplier (§3)
    warmup at the framework's own fraction of the arm's own length, truncated (§4)
    one GPU's block count chosen per size by measurement, with the accumulation
      count carrying the rest (§4)

So this derives them and compares the result against the printed tables, which are
written out below as expectations. A cell that disagrees is reported rather than
silently preferred: either the rule or the table is wrong, and which one it is cannot
be settled by copying one into the other.

    molnl_batch_sweep_arms.py --out rt1-...tsv [--micro-batch-set current]

One GPU's block count is the one value the directive does not fix. §4 asks for it to
be measured per size and for the measured numbers to be reported, and it constrains
the answer: ``one GPU's blocks x 4`` must divide 256, so 4, 8, 16, 32 and 64 are the
candidates. ``--micro-batch-set`` names which table is used, every row records it, and
``current`` -- what molnl's configs carry today -- is what the grid falls back to until
the measurement replaces it. The sets differ in throughput, not in the global batch.

Read-only: it computes and writes one TSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: §1. Five batches, four sizes, both frameworks.
BATCHES = [256, 512, 1024, 2048, 4096]
SIZES = ["small", "medium", "large", "xl"]

#: §2. The budget, in 1,024-token blocks. BERT is the existing 12,000 x 2,560 grid's
#: block count; GPT-2 is 30 epochs, rounded to a multiple of its evaluation interval
#: and so of every batch. The train split is 318,118 blocks (stated in the configs and
#: in the header of rt1-molnl-bert-grid-eval-loss-mask.tsv).
BUDGET_BLOCKS = {"bert": 30_720_000, "gpt2": 9_543_680}
TRAIN_BLOCKS = 318_118
SEQUENCE_LENGTH = 1024

#: §2. The intervals, in blocks rather than in steps: that is what puts the arms'
#: points at the same place on a block axis whatever their batch.
EVAL_EVERY_BLOCKS = {"bert": 122_880, "gpt2": 40_960}
SAVE_EVERY_BLOCKS = {"bert": 614_400, "gpt2": None}

#: §3. The 2,560 arm each column is scaled from. BERT's four are the best values of
#: the 2026-09-24 grid (read back from rt1-molnl-bert-grid-eval-loss-mask.tsv: small
#: 1e-3 0.0732, medium 3e-4 0.0670, large 1e-4 0.1161), except xl, which was not in
#: that grid and is extrapolated by the 1/3.16 per size the other three show. GPT-2's
#: four are the values its per-size configs carry, which are the rates the "shuffled
#: corpus" rows of rt1-molnl-gpt2-ladder-val-loss.tsv ran at.
BASE_LR_2560 = {
    "bert": {"small": 1e-3, "medium": 3e-4, "large": 1e-4, "xl": 3.16e-5},
    "gpt2": {"small": 6e-4, "medium": 3e-4, "large": 2.5e-4, "xl": 2e-4},
}
BASE_BATCH = 2560

#: §3. BERT runs several rates per batch and keeps the lowest; GPT-2 runs one. The
#: multipliers are the directive's: 0.7 and 1.0 where there are two, 0.7, 1.0 and 1.4
#: where there are three.
BERT_MULTIPLIERS = {256: [0.7, 1.0], 512: [0.7, 1.0], 1024: [0.7, 1.0],
                    2048: [0.7, 1.0, 1.4], 4096: [0.7, 1.0, 1.4]}
GPT2_MULTIPLIERS = {b: [1.0] for b in BATCHES}

#: §2. warmup is the framework's own current fraction of its own step count, truncated.
WARMUP_FRACTION = {"bert": 0.10, "gpt2": 0.02}

#: One GPU's block count. ``current`` is what molnl's configs carry today: BERT 8 at
#: every size, GPT-2 16 / 16 / 8 / 4. All of them divide 256 at 4 GPUs, so the grid can
#: be derived from them; §4 nonetheless asks for the fastest shape per size, measured.
#:
#: The one molnl measurement on record does not answer it. Jobs 75095-75097 compared
#: 8x80, 32x20 and 64x10 on medium and found 35.57 / 35.16 / 35.15 s/step -- a 1.2 %
#: spread, which is where "molnl is already compute-bound" comes from. They ran on
#: 2026-09-01; bf16 reached the BERT base configs on 2026-09-17 (7a13dc0f), and
#: production medium now runs at about 7.6 s/step. The probe is a measurement of a
#: 4.5x slower fp32 run, and what fraction of a bf16 step is spent waiting for input
#: does not follow from it.
MICRO_BATCH = {
    "current": {
        "bert": {"small": 8, "medium": 8, "large": 8, "xl": 8},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
    #: §4's measurement, 2026-11-20. Written here as a literal, the way the rna and
    #: compounds grids write theirs, so that a checkout can derive the 68 arms with
    #: nothing but the repository. It was read out of CHOICE_FILE before, and that made
    #: the derivation -- and so eighteen tests -- work only on the machine that had run
    #: the probes: the file matches .gitignore's *.json and had never been committed.
    "measured-1120": {
        "bert": {"small": 64, "medium": 64, "large": 64, "xl": 32},
        "gpt2": {"small": 64, "medium": 64, "large": 64, "xl": 32},
    },
}
#: The probe output the literal above was taken from. It carries every measured shape
#: beside the one that won, and the probe logs each came from, so it is the provenance
#: of those eight numbers and is committed with them. When it is present it is read and
#: compared, and a disagreement raises rather than being resolved one way or the other:
#: a typed number that no longer matches its measurement is the thing worth stopping on.
CHOICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "molnl_batch_shape_choice_20261120.json")


def _cross_check_choice_file():
    if not os.path.exists(CHOICE_FILE):
        return
    with open(CHOICE_FILE, encoding="utf-8") as handle:
        choice = json.load(handle)
    name = choice.get("set")
    chosen = choice.get("chosen", {})
    # A half-finished measurement is not compared against: all eight cells or nothing.
    if not all(size in chosen.get(arch, {}) for arch in ("bert", "gpt2")
               for size in SIZES):
        return
    typed = MICRO_BATCH.get(name)
    if typed is None:
        raise SystemExit(
            f"{CHOICE_FILE} names the set {name!r}, which MICRO_BATCH does not have. "
            f"Add it as a literal rather than loading it, or the derivation works only "
            f"where this file is.")
    differ = [f"{arch} {size}: module {typed[arch][size]} vs file {chosen[arch][size]}"
              for arch in ("bert", "gpt2") for size in SIZES
              if typed[arch][size] != chosen[arch][size]]
    if differ:
        raise SystemExit(f"MICRO_BATCH[{name!r}] disagrees with {CHOICE_FILE}: "
                         + "; ".join(differ))


_cross_check_choice_file()

#: §4's shape as it is actually run, 2026-11-24. The set above is the measurement, and
#: it is derived from it rather than retyped, so the two cannot drift apart: the only
#: cells that may differ are the ones named here. It is placed after the cross-check so
#: that nothing is derived from the measurement before the measurement has been checked
#: against the probe output it came from.
#:
#: The probe ranks the shapes by speed. It cannot rank them by whether a whole arm fits
#: in memory, and for one column it did not. GPT-2 xl ran the measured 32 blocks per GPU
#: and died of CUDA out of memory at batch 1,024 twice -- job 374933 at step 1,520 and
#: job 384760 at step 1,760, both in model.py's cross_entropy, on the first training
#: micro-step after a checkpoint was written. The allocation it failed on is 6.13 GiB:
#: one GPU's blocks x 1,024 positions x the vocabulary in fp32, so it halves with the
#: shape. Its four siblings at the same shape finished, so the shape is not wrong by
#: itself -- it is at the edge of a 184 GiB card, and at batch 1,024 it goes over.
#:
#: §4 fixes one shape per size, so the whole gpt2 xl column moves rather than the one
#: arm: four finished runs are re-run so that the five can be read against each other.
#: 16 blocks is the probe's runner-up, 10.851 s/step against 10.586 -- a 2.5 % cost.
ADOPTED_OVERRIDES = {("gpt2", "xl"): 16}
MICRO_BATCH["adopted-1124"] = {
    arch: {size: ADOPTED_OVERRIDES.get((arch, size), micro)
           for size, micro in by_size.items()}
    for arch, by_size in MICRO_BATCH["measured-1120"].items()
}

#: A size re-run under a different shape must not write into the directories of the
#: pass it replaces: those hold finished runs, and nanoGPT's ``init_from = "scratch"``
#: would overwrite the checkpoint rather than refuse. The suffix names the shape rather
#: than the pass, so a directory says what it is -- the unsuffixed gpt2 xl directories
#: are the 32-block pass. Only a size whose shape actually moved may carry one, which
#: is asserted in tests/unit/test_molnl_batch_sweep.py rather than left to reading.
OUT_SUFFIX = {"adopted-1124": {("gpt2", "xl"): "-m16"}}
for _set in OUT_SUFFIX:
    if _set not in MICRO_BATCH:
        raise SystemExit(f"OUT_SUFFIX names the set {_set!r}, which MICRO_BATCH "
                         f"does not have")

#: §4's constraint on any set: one GPU's blocks times the GPU count must divide 256,
#: the smallest batch of the grid.
MICRO_CANDIDATES = [4, 8, 16, 32, 64]

# ---------------------------------------------------------------------------
# The 2026-11-26 answer to the 2026-11-24 report. Opted into with --extension; with
# nothing passed, this module derives the original 68 arms and nothing else.
# ---------------------------------------------------------------------------

#: §8.2. Which multiplier won each BERT cell of the finished grid. This is a
#: measurement, not a rule: it is the lowest ``best_value`` in the cell, read off
#: rt1-molnl-bsweep-arm-summary-20261125.tsv. It is a literal here for the same reason
#: MICRO_BATCH is -- so a checkout derives the arms with nothing but the repository --
#: and scripts/molnl_grid_winners_20261125.json carries the reading it came from, which
#: is compared against this at import.
WINNING_MULTIPLIER_1125 = {
    ("bert", "small"): {256: 0.7, 512: 0.7, 1024: 0.7, 2048: 1.0, 4096: 1.0},
    ("bert", "medium"): {256: 1.0, 512: 1.0, 1024: 1.0, 2048: 1.4, 4096: 1.0},
    ("bert", "large"): {256: 1.0, 512: 1.0, 1024: 1.0, 2048: 1.4, 4096: 1.0},
    ("bert", "xl"): {256: 1.0, 512: 1.0, 1024: 1.0, 2048: 1.4, 4096: 0.7},
}
WINNERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "molnl_grid_winners_20261125.json")

#: §8.2. Three more rates above the cell's top, keyed by how many the cell already
#: tried. The step is the existing 1.4x, continued: 1.0 -> 1.4, 2.0, 2.8 for a cell
#: that ran two, and 1.4 -> 2.0, 2.8, 4.0 for one that ran three. The centre does not
#: move, so the rate is still the 2,560 value times sqrt(batch / 2560).
EXTENSION_MULTIPLIERS = {2: [1.4, 2.0, 2.8], 3: [2.0, 2.8, 4.0]}

#: §15. One BERT arm run again with nothing changed but how many blocks one GPU takes
#: at a time, to measure on the BERT side the spread the gpt2 xl pairs showed. The
#: directive names the arm: small, batch 2,048, 8.94e-4 -- the x1.0 column. The grid
#: ran it at 64 blocks; these are the other two. 32 and 16 are the probe's next two
#: measured shapes for bert small (2.725 and 2.897 s/step against 64's 2.517, so
#: +8.3 % and +15.1 %), which keeps all three on measured ground and spaces them by
#: a factor of two.
SHAPE_REPLICATE_ARM = {"arch": "bert", "size": "small", "batch": 2048,
                       "multiplier": 1.0}
SHAPE_REPLICATE_MICRO = [32, 16]


def _cross_check_winners_file():
    """The literal above against the reading it was taken from."""
    if not os.path.exists(WINNERS_FILE):
        return
    with open(WINNERS_FILE, encoding="utf-8") as handle:
        payload = json.load(handle)
    differ = []
    seen = set()
    for cell in payload.get("cells", []):
        key = (cell["arch"], cell["size"])
        batch = int(cell["batch"])
        if key not in WINNING_MULTIPLIER_1125:
            differ.append(f"{key} is in the file and not in the module")
            continue
        seen.add((key, batch))
        typed = WINNING_MULTIPLIER_1125[key].get(batch)
        read = float(cell["winning_multiplier"])
        if typed is None:
            differ.append(f"{key[0]} {key[1]} batch {batch}: missing from the module")
        elif abs(typed - read) > 1e-9:
            differ.append(f"{key[0]} {key[1]} batch {batch}: module x{typed} vs "
                          f"file x{read}")
    for key, by_batch in WINNING_MULTIPLIER_1125.items():
        for batch in by_batch:
            if (key, batch) not in seen:
                differ.append(f"{key[0]} {key[1]} batch {batch}: in the module and "
                              f"not in {os.path.basename(WINNERS_FILE)}")
    if differ:
        raise SystemExit(f"WINNING_MULTIPLIER_1125 disagrees with {WINNERS_FILE}: "
                         + "; ".join(differ))


_cross_check_winners_file()


def extension_cells():
    """§8.2's cells: the ones whose winning rate was the largest the cell tried."""
    cells = []
    for (arch, size), by_batch in WINNING_MULTIPLIER_1125.items():
        for batch in BATCHES:
            tried = BERT_MULTIPLIERS[batch] if arch == "bert" else GPT2_MULTIPLIERS[batch]
            won = by_batch[batch]
            if abs(won - max(tried)) > 1e-9:
                continue
            added = EXTENSION_MULTIPLIERS.get(len(tried))
            if added is None:
                raise SystemExit(f"{arch} {size} batch {batch} tried {len(tried)} "
                                 f"rates, and EXTENSION_MULTIPLIERS has no row for "
                                 f"that count")
            cells.append({"arch": arch, "size": size, "batch": batch,
                          "tried": list(tried), "won": won, "added": list(added)})
    cells.sort(key=lambda c: (c["arch"], SIZES.index(c["size"]), c["batch"]))
    return cells


#: §8.2's own counts, to be compared against rather than derived from.
EXPECTED_EXTENSION = {"cells": 12, "arms": 36, "shape_replicates": 2}

#: nanoGPT divides gradient_accumulation_steps by the DDP world size, so GPT-2's global
#: batch is micro x accumulation with no world-size factor (molnl's config says
#: 16 x 160 = 2,560). BERT multiplies by the GPU count as well (8 x 80 x 4 = 2,560).
#: §6 fixes the GPU count at 4 for BERT.
WORLD_SIZE = {"bert": 4, "gpt2": 1}

#: §6. One evaluation micro-batch for all 48 BERT arms. 8 is what every molnl BERT run
#: has evaluated at, xl included, so it is the largest value xl is on record as having
#: run -- and holding it there keeps the fixed mask drawing the same positions as the
#: 2026-09-24 and 24,000-step runs, which is what makes this grid's numbers readable
#: against theirs. The masked positions depend on how the rows fall into batches.
BERT_EVAL_BATCH = 8
#: §6. molnl's validation split is 1,631 blocks and is scored whole: no subset is
#: drawn, so there is no subset size and no subset seed to hold. The mask is fixed by
#: fixed_eval_mask with eval_mask_seed 42.
BERT_EVAL_BLOCKS = 1631

#: §7. What an arm is judged, kept and ranked on.
JUDGE_METRIC = {"bert": "eval_loss_mask", "gpt2": "val_loss"}

#: §8. GPT-2's floor is a tenth of its own peak and its decay length is its own run.
GPT2_MIN_LR_DIVISOR = 10
#: Not fixed by the directive. Checkpoint opportunities every five evaluation windows,
#: which puts them at the same block positions across the five batches -- the same
#: multiple BERT's own save interval is (614,400 = 5 x 122,880).
GPT2_SAVE_MULTIPLE = 5

# ---------------------------------------------------------------------------
# The tables as the directive prints them. Nothing below reads these except check().
# ---------------------------------------------------------------------------

EXPECTED_STEPS = {
    "bert": {256: 120_000, 512: 60_000, 1024: 30_000, 2048: 15_000, 4096: 7_500},
    "gpt2": {256: 37_280, 512: 18_640, 1024: 9_320, 2048: 4_660, 4096: 2_330},
}
EXPECTED_INTERVALS = {
    "bert": {256: (480, 2400), 512: (240, 1200), 1024: (120, 600),
             2048: (60, 300), 4096: (30, 150)},
    "gpt2": {256: (160, None), 512: (80, None), 1024: (40, None), 2048: (20, None),
             4096: (10, None)},
}
EXPECTED_WARMUP = {
    "bert": {256: 12_000, 512: 6_000, 1024: 3_000, 2048: 1_500, 4096: 750},
    "gpt2": {256: 745, 512: 372, 1024: 186, 2048: 93, 4096: 46},
}
#: §2's own summary line, which is a second statement of the same budget.
EXPECTED_BUDGET = {
    "bert": {"epochs": 96.57, "tokens_g": 31.46, "eval_points": 250},
    "gpt2": {"epochs": 30.00, "tokens_g": 9.77, "eval_points": 233},
}
EXPECTED_LR = {
    "bert": {
        "small": {256: [2.21e-4, 3.16e-4], 512: [3.13e-4, 4.47e-4],
                  1024: [4.43e-4, 6.32e-4], 2048: [6.26e-4, 8.94e-4, 1.25e-3],
                  4096: [8.85e-4, 1.26e-3, 1.77e-3]},
        "medium": {256: [6.64e-5, 9.49e-5], 512: [9.39e-5, 1.34e-4],
                   1024: [1.33e-4, 1.90e-4], 2048: [1.88e-4, 2.68e-4, 3.76e-4],
                   4096: [2.66e-4, 3.79e-4, 5.31e-4]},
        "large": {256: [2.21e-5, 3.16e-5], 512: [3.13e-5, 4.47e-5],
                  1024: [4.43e-5, 6.32e-5], 2048: [6.26e-5, 8.94e-5, 1.25e-4],
                  4096: [8.85e-5, 1.26e-4, 1.77e-4]},
        "xl": {256: [6.99e-6, 9.99e-6], 512: [9.89e-6, 1.41e-5],
               1024: [1.40e-5, 2.00e-5], 2048: [1.98e-5, 2.83e-5, 3.96e-5],
               4096: [2.80e-5, 4.00e-5, 5.60e-5]},
    },
    "gpt2": {
        "small": {256: [1.90e-4], 512: [2.68e-4], 1024: [3.79e-4], 2048: [5.37e-4],
                  4096: [7.59e-4]},
        "medium": {256: [9.49e-5], 512: [1.34e-4], 1024: [1.90e-4], 2048: [2.68e-4],
                   4096: [3.79e-4]},
        "large": {256: [7.91e-5], 512: [1.12e-4], 1024: [1.58e-4], 2048: [2.24e-4],
                  4096: [3.16e-4]},
        "xl": {256: [6.32e-5], 512: [8.94e-5], 1024: [1.26e-4], 2048: [1.79e-4],
               4096: [2.53e-4]},
    },
}
#: §1's count.
EXPECTED_ARMS = {"bert": 48, "gpt2": 20}

COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "lr_slug", "steps",
           "warmup_steps", "eval_interval", "save_interval", "blocks", "tokens",
           "epochs", "eval_points", "micro_batch", "grad_accum", "world_size",
           "expected_global_batch", "eval_batch", "eval_blocks", "min_lr",
           "lr_decay_iters", "judge_metric", "micro_batch_set", "origin", "config",
           "out_dir"]


def three_sig(value: float) -> float:
    """The value rounded the way the directive's tables are written."""
    if value == 0:
        return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + 2)


def lr_slug(value: float) -> str:
    """``4.47e-4`` -> ``4p47e4``, which is how the existing arms are named."""
    mantissa, _, exponent = f"{value:.2e}".partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa.replace('.', 'p')}e{abs(int(exponent))}"


def scaled_rate(arch: str, size: str, batch: int, multiplier: float) -> float:
    """§3: the 2,560 rate times sqrt(batch / 2560), times the column's multiplier."""
    base = BASE_LR_2560[arch][size]
    return three_sig(base * math.sqrt(batch / BASE_BATCH) * multiplier)


def interval(arch: str, batch: int, which: str):
    """§2: the interval in blocks, expressed in this arm's own steps."""
    blocks = (EVAL_EVERY_BLOCKS if which == "eval" else SAVE_EVERY_BLOCKS)[arch]
    if blocks is None:
        return None
    if blocks % batch:
        raise SystemExit(f"{arch} {which} interval {blocks:,} blocks is not a multiple "
                         f"of batch {batch}")
    return blocks // batch


def steps_for(arch: str, batch: int) -> int:
    blocks = BUDGET_BLOCKS[arch]
    if blocks % batch:
        raise SystemExit(f"{arch} budget {blocks:,} is not a multiple of batch {batch}")
    return blocks // batch


def warmup_for(arch: str, steps: int) -> int:
    """§2: the framework's fraction of this arm's own length, truncated."""
    return int(WARMUP_FRACTION[arch] * steps)


def accumulation(arch: str, size: str, batch: int, which_set: str):
    """(micro_batch, grad_accum, world_size) that multiply out to this batch."""
    micro = MICRO_BATCH[which_set][arch][size]
    world = WORLD_SIZE[arch]
    if micro not in MICRO_CANDIDATES:
        raise SystemExit(f"{arch} {size}: one GPU's {micro} blocks is not one of "
                         f"{MICRO_CANDIDATES} (§4)")
    gpus = 4  # §6 for BERT; GPT-2 runs on 4 as well and divides the accumulation by it
    if 256 % (micro * gpus):
        raise SystemExit(f"{arch} {size}: {micro} blocks x {gpus} GPUs cannot build "
                         f"the grid's smallest batch, 256 (§4)")
    divisor = micro * world
    if batch % divisor:
        raise SystemExit(f"{arch} {size} batch {batch} is not a multiple of "
                         f"micro {micro} x world {world}")
    return micro, batch // divisor, world


def accumulation_with_micro(arch: str, batch: int, micro: int):
    """(micro_batch, grad_accum, world_size) for one named shape, not a set.

    §15 varies one GPU's block count on purpose, so the shape comes from the arm
    rather than from the size's entry in MICRO_BATCH. The constraints are the same.
    """
    world = WORLD_SIZE[arch]
    if micro not in MICRO_CANDIDATES:
        raise SystemExit(f"{arch}: one GPU's {micro} blocks is not one of "
                         f"{MICRO_CANDIDATES} (§4)")
    divisor = micro * world
    if batch % divisor:
        raise SystemExit(f"{arch} batch {batch} is not a multiple of micro {micro} "
                         f"x world {world}")
    return micro, batch // divisor, world


def arms(which_set: str, extension: bool = False):
    rows = []
    #: §8.2's three rates per in-scope cell, and §15's two shapes of one existing arm.
    #: Both are empty unless the extension is asked for, so the 68 arms the grid ran
    #: are derived exactly as they were.
    added_rates = {}
    added_shapes = {}
    if extension:
        for cell in extension_cells():
            added_rates[(cell["arch"], cell["size"], cell["batch"])] = cell["added"]
        key = (SHAPE_REPLICATE_ARM["arch"], SHAPE_REPLICATE_ARM["size"],
               SHAPE_REPLICATE_ARM["batch"])
        added_shapes[key] = (SHAPE_REPLICATE_ARM["multiplier"], SHAPE_REPLICATE_MICRO)

    for arch in ("bert", "gpt2"):
        multipliers = BERT_MULTIPLIERS if arch == "bert" else GPT2_MULTIPLIERS
        blocks = BUDGET_BLOCKS[arch]
        for size in SIZES:
            for batch in BATCHES:
                steps = steps_for(arch, batch)
                micro, accum, world = accumulation(arch, size, batch, which_set)
                eval_interval = interval(arch, batch, "eval")
                plan = [(m, "grid-1120", None) for m in multipliers[batch]]
                plan += [(m, "extend-1126-lr", None)
                         for m in added_rates.get((arch, size, batch), [])]
                shape = added_shapes.get((arch, size, batch))
                if shape is not None:
                    plan += [(shape[0], "extend-1126-shape", m) for m in shape[1]]
                for multiplier, origin, micro_override in plan:
                    lr = scaled_rate(arch, size, batch, multiplier)
                    slug = lr_slug(lr)
                    stem = f"{arch}_{size}_b{batch}_lr{slug}"
                    out_size = "ex-large" if (arch == "gpt2" and size == "xl") else size
                    out_suffix = OUT_SUFFIX.get(which_set, {}).get((arch, size), "")
                    if micro_override is not None:
                        # The same arm at another shape: same batch, same rate, same
                        # name but for the shape. Without the suffix it would be given
                        # the directory of the 64-block run it is being compared with.
                        micro, accum, world = accumulation_with_micro(
                            arch, batch, micro_override)
                        stem = f"{stem}_m{micro_override}"
                        out_suffix = f"{out_suffix}-m{micro_override}"
                    else:
                        micro, accum, world = accumulation(arch, size, batch, which_set)
                    rows.append({
                        "arch": arch, "size": size, "batch": batch, "lr": f"{lr:.2e}",
                        "lr_multiplier": f"{multiplier:.1f}", "lr_slug": slug,
                        "steps": steps,
                        "warmup_steps": warmup_for(arch, steps),
                        "eval_interval": eval_interval,
                        "save_interval": interval(arch, batch, "save")
                                         if arch == "bert"
                                         else eval_interval * GPT2_SAVE_MULTIPLE,
                        "blocks": blocks,
                        "tokens": blocks * SEQUENCE_LENGTH,
                        "epochs": f"{blocks / TRAIN_BLOCKS:.2f}",
                        "eval_points": blocks // EVAL_EVERY_BLOCKS[arch],
                        "micro_batch": micro, "grad_accum": accum, "world_size": world,
                        "expected_global_batch": batch,
                        "eval_batch": BERT_EVAL_BATCH if arch == "bert" else None,
                        "eval_blocks": BERT_EVAL_BLOCKS if arch == "bert" else None,
                        "min_lr": (f"{three_sig(lr / GPT2_MIN_LR_DIVISOR):.2e}"
                                   if arch == "gpt2" else None),
                        "lr_decay_iters": steps if arch == "gpt2" else None,
                        "judge_metric": JUDGE_METRIC[arch],
                        "micro_batch_set": which_set,
                        "origin": origin,
                        "config": "molcrawl/tasks/pretrain/configs/molecule_nat_lang/"
                                  f"{stem}.py",
                        # The architecture is in the name because on this modality
                        # the two of them collide without it: BERT medium and GPT-2
                        # medium share the 2,560 rate 3e-4, so their x1.0 arms scale
                        # to the same rate and the same slug at all five batches.
                        # bert-output/ and gpt2-output/ keep them apart on disk, but a
                        # directory name that does not name one arm is read as if it
                        # did. compounds has the same template and got away with it
                        # because its two sets of rates happen never to meet.
                        "out_dir": f"molecule_nat_lang-{out_size}-{arch}"
                                   f"-b{batch}-lr{slug}{out_suffix}",
                    })
    return rows


def check(rows, which_set: str) -> int:
    """Compare the derivation against the directive's printed tables."""
    problems = 0

    base = [r for r in rows if r["origin"] == "grid-1120"]
    for arch in ("bert", "gpt2"):
        got = sum(1 for r in base if r["arch"] == arch)
        want = EXPECTED_ARMS[arch]
        if got != want:
            problems += 1
            print(f"  ** {arch}: {got} arms against directive {want}")

    for arch in ("bert", "gpt2"):
        blocks = BUDGET_BLOCKS[arch]
        want = EXPECTED_BUDGET[arch]
        got_epochs = round(blocks / TRAIN_BLOCKS, 2)
        got_tokens = round(blocks * SEQUENCE_LENGTH / 1e9, 2)
        got_points = blocks // EVAL_EVERY_BLOCKS[arch]
        for label, g, w in (("epochs", got_epochs, want["epochs"]),
                            ("tokens (G)", got_tokens, want["tokens_g"]),
                            ("eval points", got_points, want["eval_points"])):
            ok = g == w
            if not ok:
                problems += 1
            print(f"  {'ok' if ok else '**'} {arch:5s} {label:12s} {g} "
                  f"(directive {w})")

    for arch in ("bert", "gpt2"):
        for batch in BATCHES:
            got = steps_for(arch, batch)
            want = EXPECTED_STEPS[arch][batch]
            if got != want:
                problems += 1
            print(f"  {'ok' if got == want else '**'} {arch:5s} batch {batch:5d}  "
                  f"steps {got:>7,} (directive {want:>7,})")
            want_eval, want_save = EXPECTED_INTERVALS[arch][batch]
            for which, wanted in (("eval", want_eval), ("save", want_save)):
                if wanted is None:
                    continue
                g = interval(arch, batch, which)
                if g != wanted:
                    problems += 1
                    print(f"     ** {which} interval {g} against directive {wanted}")
            g = warmup_for(arch, got)
            w = EXPECTED_WARMUP[arch][batch]
            if g != w:
                problems += 1
                print(f"     ** warmup {g} against directive {w}")

    for arch in ("bert", "gpt2"):
        for size in SIZES:
            for batch in BATCHES:
                sel = [r for r in base if r["arch"] == arch and r["size"] == size
                       and r["batch"] == batch]
                got = sorted(float(r["lr"]) for r in sel)
                want = sorted(EXPECTED_LR[arch][size][batch])
                if [three_sig(v) for v in got] != [three_sig(v) for v in want]:
                    problems += 1
                    print(f"  ** {arch} {size} batch {batch}: derived "
                          f"{[f'{v:.2e}' for v in got]} against directive "
                          f"{[f'{v:.2e}' for v in want]}")

    added = [r for r in rows if r["origin"] == "extend-1126-lr"]
    shaped = [r for r in rows if r["origin"] == "extend-1126-shape"]
    if added or shaped:
        print("\n  --- the 2026-11-26 answer's extension ---")
        cells = extension_cells()
        for label, got, want in (
                ("cells in scope", len(cells), EXPECTED_EXTENSION["cells"]),
                ("rates added", len(added), EXPECTED_EXTENSION["arms"]),
                ("shape replicates", len(shaped),
                 EXPECTED_EXTENSION["shape_replicates"])):
            ok = got == want
            if not ok:
                problems += 1
            print(f"  {'ok' if ok else '**'} {label:18s} {got} (answer {want})")
        # Every added rate is the cell's centre times its multiplier, and is above
        # every rate the cell already ran: an extension that lands inside the tried
        # range is not an extension.
        for cell in cells:
            top = max(scaled_rate(cell["arch"], cell["size"], cell["batch"], m)
                      for m in cell["tried"])
            for multiplier in cell["added"]:
                lr = scaled_rate(cell["arch"], cell["size"], cell["batch"], multiplier)
                if not lr > top:
                    problems += 1
                    print(f"  ** {cell['arch']} {cell['size']} batch {cell['batch']}: "
                          f"added {lr:.2e} (x{multiplier}) is not above the cell's "
                          f"top {top:.2e}")
        for row in shaped:
            if int(row["micro_batch"]) == MICRO_BATCH[which_set][row["arch"]][
                    row["size"]]:
                problems += 1
                print(f"  ** {row['out_dir']}: the shape replicate carries the set's "
                      f"own block count, so it repeats an existing arm")
        print("  cells: " + ", ".join(
            f"{c['size']} b{c['batch']} x{c['won']}->{c['added']}" for c in cells))

    rates = sum(len(EXPECTED_LR[a][s][b])
                for a in EXPECTED_LR for s in SIZES for b in BATCHES)
    print(f"\n  compared: {rates} rates, {2 * len(BATCHES)} step counts, "
          f"{2 * len(BATCHES)} warmup lengths, "
          f"{len(BATCHES) * 3} intervals, 6 budget summaries")
    print(f"  one GPU's block count came from the '{which_set}' set, which the "
          f"directive leaves to measurement (§4); it is not checked against a table")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--micro-batch-set", choices=sorted(MICRO_BATCH),
                    default="current",
                    help="one GPU's block count per size; 'current' is what molnl's "
                         "configs carry today")
    ap.add_argument("--out", required=True,
                    help="the base name begins with rt1- (the column convention)")
    ap.add_argument("--extension", action="store_true",
                    help="also derive the 2026-11-26 answer's arms: three rates above "
                         "the top of every cell whose top rate won (§8.2), and one "
                         "arm at two more block counts (§15). Without it, the 68 arms "
                         "the grid ran are derived and nothing else")
    args = ap.parse_args(argv)
    check_name(args.out)

    rows = arms(args.micro_batch_set, extension=args.extension)
    by_origin = {}
    for row in rows:
        by_origin[row["origin"]] = by_origin.get(row["origin"], 0) + 1
    print(f"=== derived {len(rows)} arms "
          f"(BERT {sum(1 for r in rows if r['arch'] == 'bert')}, "
          f"GPT-2 {sum(1 for r in rows if r['arch'] == 'gpt2')}), "
          f"one GPU's blocks from '{args.micro_batch_set}' ===")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(by_origin.items())))
    print("\n=== against the directive's printed tables ===")
    problems = check(rows, args.micro_batch_set)

    print("\n=== per size and batch ===")
    for arch in ("bert", "gpt2"):
        for size in SIZES:
            for batch in BATCHES:
                sel = [r for r in rows if r["arch"] == arch and r["size"] == size
                       and r["batch"] == batch]
                one = sel[0]
                print(f"  {arch:5s} {size:6s} batch {batch:5d}  {len(sel)} arm(s)  "
                      f"steps {one['steps']:>7,}  warmup {one['warmup_steps']:>6,}  "
                      f"eval {one['eval_interval']:>4}  "
                      f"save {str(one['save_interval']):>4}  "
                      f"{one['micro_batch']} x {one['grad_accum']} x "
                      f"{one['world_size']} = {one['expected_global_batch']}")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "several (columns: arch,size,batch,lr) -- the molecule_nat_lang "
                 "batch-size grid asked for by the 2026-11-20 directive",
        "source": "derived from the directive's rules (budget in blocks; intervals in "
                  "blocks; sqrt scaling off the 2,560 arm; warmup fraction; one GPU's "
                  "block count) and compared against its printed tables",
        "generated": stamp("scripts/molnl_batch_sweep_arms.py"),
        "note": [
            f"budget: BERT {BUDGET_BLOCKS['bert']:,} blocks "
            f"({BUDGET_BLOCKS['bert'] / TRAIN_BLOCKS:.2f} epochs of the "
            f"{TRAIN_BLOCKS:,}-block train split), GPT-2 {BUDGET_BLOCKS['gpt2']:,} "
            f"({BUDGET_BLOCKS['gpt2'] / TRAIN_BLOCKS:.2f} epochs); block "
            f"{SEQUENCE_LENGTH} tokens",
            f"intervals in blocks: BERT evaluates every {EVAL_EVERY_BLOCKS['bert']:,} "
            f"and saves every {SAVE_EVERY_BLOCKS['bert']:,}, GPT-2 evaluates every "
            f"{EVAL_EVERY_BLOCKS['gpt2']:,}",
            "lr = the 2,560 arm's rate x sqrt(batch/2560) x the column's multiplier, "
            "to three significant figures; BERT's 2,560 rates are the best values of "
            "the 2026-09-24 grid, xl extrapolated; GPT-2's are its configs' own",
            f"warmup: BERT {WARMUP_FRACTION['bert']:.0%} and GPT-2 "
            f"{WARMUP_FRACTION['gpt2']:.0%} of the arm's own length, truncated",
            f"BERT: evaluation micro-batch {BERT_EVAL_BATCH} and 4 GPUs on all 48 arms, "
            f"the whole {BERT_EVAL_BLOCKS:,}-block validation split at every point, "
            f"mask fixed with seed 42; judged on {JUDGE_METRIC['bert']}. GPT-2 on "
            f"{JUDGE_METRIC['gpt2']}",
            f"one GPU's block count from the '{args.micro_batch_set}' set; BERT x "
            f"{WORLD_SIZE['bert']} GPUs, GPT-2 with no world-size factor",
            "the 68 arms with origin grid-1120 ran 2026-11-20 to 11-25 and are "
            "reported in rt1-molnl-bsweep-arm-summary-20261125.tsv; rows with "
            "origin extend-1126-lr or extend-1126-shape have not run",
            "extension (2026-11-26 answer): three rates above the top of every cell "
            "whose top rate won (§8.2, BERT only), and one arm at two more block "
            "counts (§15)" if args.extension else
            "extension: not asked for, so only the grid's own 68 arms are here",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"\n{len(rows)} arms -> {args.out}")
    print(f"disagreements with the directive: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
