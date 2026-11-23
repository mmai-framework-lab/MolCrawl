"""Derive the rna batch-size grid's arms, and check the directive's own tables.

The directive of 2026-11-20 prints the step counts, the evaluation and save intervals,
the warmup lengths and the learning rates it wants for 68 arms. All of them follow
from rules the same directive states:

    a fixed number of blocks, so the step count is blocks / batch (§2)
    intervals in blocks, so every arm's points land at the same block position (§2)
    the rate scaled by sqrt(batch / 2560) off the adopted 2,560 arm (§3)
    warmup at the framework's own fraction, truncated (§2)
    one GPU's block count held across the five batches of a size, so the
    accumulation count carries the whole difference (§4)

So this derives them and compares the result against the printed tables, which are
written out below as expectations. A cell that disagrees is reported rather than
silently preferred: either the rule or the table is wrong, and which one it is cannot
be settled by copying one into the other.

    rna_batch_sweep_arms.py --micro-batch-set current --out rt1-...tsv

One GPU's block count is the open value. §4 asks for it to be measured per size and
set to the fastest shape that can build 256, and until that measurement exists the
only honest sets are the shape the rna arms run now (``current``) and the shape the
same four model sizes measured fastest on compounds (``compounds-prior``), which is a
prior and not an rna measurement. ``--micro-batch-set`` names which one a row was
built with and the name goes into every row, so a table can never be read without
knowing which shape it describes.

Read-only: it computes and writes one TSV.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

#: §1. Five batches, four sizes, both frameworks.
BATCHES = [256, 512, 1024, 2048, 4096]
SIZES = ["small", "medium", "large", "xl"]
ARCHES = ["bert", "gpt2"]

#: §2. The budget, in blocks of 1,024 tokens. The existing grid's 120,960 x 2,560,
#: so this grid meets it at the same block position. Both frameworks spend the same.
BUDGET_BLOCKS = 309_657_600
SEQUENCE_LENGTH = 1024

#: §2. Evaluation every 1,376,256 blocks and saving at five times that, which puts
#: 225 evaluation points and 45 save points on every arm whatever its batch.
EVAL_EVERY_BLOCKS = 1_376_256
SAVE_EVERY_BLOCKS = 6_881_280

#: §3. The adopted 2,560 arm whose rate each column is scaled from.
BASE_LR_2560 = {
    "bert": {"small": 3.0e-4, "medium": 3.0e-4, "large": 1.7e-4, "xl": 2.5e-5},
    "gpt2": {"small": 1.2e-3, "medium": 1.2e-3, "large": 6.0e-4, "xl": 6.0e-4},
}
BASE_BATCH = 2560

#: §3. BERT runs several rates per batch and keeps the lowest; GPT-2 runs one.
BERT_MULTIPLIERS = {256: [0.7, 1.0], 512: [0.7, 1.0], 1024: [0.7, 1.0],
                    2048: [0.7, 1.0, 1.4], 4096: [0.7, 1.0, 1.4]}
GPT2_MULTIPLIERS = {batch: [1.0] for batch in BATCHES}

#: §2. Each framework's own current fraction, truncated.
WARMUP_FRACTION = {"bert": 0.10, "gpt2": 0.02}

#: BERT's global batch is micro x accumulation x GPU count (the rna base runs
#: 8 x 80 x 4 = 2,560); GPT-2's is micro x accumulation with no GPU factor (its base
#: says 16 x 160 = 2,560) because nanoGPT divides the accumulation count across ranks.
#: §6 fixes the GPU count at 4 for BERT, and the rna GPT-2 grid also runs on 4.
WORLD_SIZE = {"bert": 4, "gpt2": 1}
GPU_COUNT = 4

#: One GPU's block count. ``current`` is what the rna arms run today -- BERT 8 at every
#: size, GPT-2 16 / 16 / 8 / 4 -- and every one of those values can build 256, so the
#: grid is runnable under it; §4 asks for the fastest instead and says not to leave
#: BERT at 8. ``compounds-prior`` is the shape the same four model sizes measured
#: fastest on compounds at 4 GPUs and 1,024 tokens (small 3.13 s/step at 32, medium
#: 7.35 at 64, large 12.14 at 64, xl 23.70 at 32, against 8 being 1.84 to 1.97 times
#: slower). It is a prior from another corpus, not an rna measurement: rna reads
#: 1,024-token blocks out of a uint16 memmap and is latency-bound where compounds is
#: not, so the ranking has to be measured on rna before a row built from this set is
#: used for anything but planning.
MICRO_BATCH = {
    "current": {
        "bert": {"small": 8, "medium": 8, "large": 8, "xl": 8},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
    "compounds-prior": {
        "bert": {"small": 32, "medium": 64, "large": 64, "xl": 32},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
}

#: §6. One evaluation micro-batch for all 48 BERT arms. 8 is what all 25 arms of the
#: existing rna grid and all four extension arms evaluate at, xl included, so it needs
#: no untried value; it groups the fixed 10,000 eval rows into 313 batches of 32 on
#: 4 GPUs, and that grouping is what decides which positions the fixed mask hides.
BERT_EVAL_BATCH = 8

#: §7. What the arm is judged and kept on.
JUDGE_METRIC = {"bert": "eval_loss_mask", "gpt2": "val_loss"}

#: §8. GPT-2's floor is a tenth of its own peak and its decay length is its own run.
GPT2_MIN_LR_DIVISOR = 10

#: The collapse detector, measured for rna on 2026-11-18 and approved for the four
#: extension arms: the 14 arms that ended collapsed sit at a median of 9.3737 at the
#: lowest, and the unigram floor is 9.3710, so 9.0 is below both. §8 does not name it;
#: it was put to the lead with the measurement below and approved for this grid on
#: 2026-11-20, so it is on unless --no-collapse-detection is passed.
DEGENERATE_THRESHOLD = 9.0
DEGENERATE_PATIENCE = 5

#: grace is in steps and is compared against the absolute step count, so it cannot be
#: one number across batches that differ 16-fold. It is a block position divided by
#: each arm's batch. The position: of the four adopted arms, the last to fall below
#: 9.0 is xl, at 6,656,000 blocks -- 2.15 % of the budget (job 374267, measured off
#: the 2026-11-17 curve TSV). Twice that leaves the same 4.30 % of the schedule on
#: every one of the five batches, and it divides each of them exactly.
GRACE_BLOCKS = 13_312_000

COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "lr_slug", "rate_rank",
           "rates_at_this_batch", "steps", "warmup_steps", "eval_interval",
           "save_interval", "eval_points", "save_points", "blocks", "tokens",
           "micro_batch", "grad_accum", "world_size", "gpu_count",
           "expected_global_batch", "eval_batch", "min_lr", "lr_decay_iters",
           "judge_metric", "degenerate_loss_threshold", "degenerate_patience",
           "degenerate_grace_steps", "micro_batch_set", "config", "out_dir"]

# ---------------------------------------------------------------------------
# The tables as the directive prints them. Nothing below reads these except check().
# ---------------------------------------------------------------------------

EXPECTED_STEPS = {256: 1_209_600, 512: 604_800, 1024: 302_400, 2048: 151_200,
                  4096: 75_600}
EXPECTED_INTERVALS = {256: (5_376, 26_880), 512: (2_688, 13_440),
                      1024: (1_344, 6_720), 2048: (672, 3_360), 4096: (336, 1_680)}
EXPECTED_WARMUP = {
    "bert": {256: 120_960, 512: 60_480, 1024: 30_240, 2048: 15_120, 4096: 7_560},
    "gpt2": {256: 24_192, 512: 12_096, 1024: 6_048, 2048: 3_024, 4096: 1_512},
}
EXPECTED_LR = {
    "bert": {
        "small": {256: [6.64e-5, 9.49e-5], 512: [9.39e-5, 1.34e-4],
                  1024: [1.33e-4, 1.90e-4], 2048: [1.88e-4, 2.68e-4, 3.76e-4],
                  4096: [2.66e-4, 3.79e-4, 5.31e-4]},
        "medium": {256: [6.64e-5, 9.49e-5], 512: [9.39e-5, 1.34e-4],
                   1024: [1.33e-4, 1.90e-4], 2048: [1.88e-4, 2.68e-4, 3.76e-4],
                   4096: [2.66e-4, 3.79e-4, 5.31e-4]},
        "large": {256: [3.76e-5, 5.38e-5], 512: [5.32e-5, 7.60e-5],
                  1024: [7.53e-5, 1.08e-4], 2048: [1.06e-4, 1.52e-4, 2.13e-4],
                  4096: [1.51e-4, 2.15e-4, 3.01e-4]},
        "xl": {256: [5.53e-6, 7.91e-6], 512: [7.83e-6, 1.12e-5],
               1024: [1.11e-5, 1.58e-5], 2048: [1.57e-5, 2.24e-5, 3.13e-5],
               4096: [2.21e-5, 3.16e-5, 4.43e-5]},
    },
    "gpt2": {
        "small": {256: [3.79e-4], 512: [5.37e-4], 1024: [7.59e-4], 2048: [1.07e-3],
                  4096: [1.52e-3]},
        "medium": {256: [3.79e-4], 512: [5.37e-4], 1024: [7.59e-4], 2048: [1.07e-3],
                   4096: [1.52e-3]},
        "large": {256: [1.90e-4], 512: [2.68e-4], 1024: [3.79e-4], 2048: [5.37e-4],
                  4096: [7.59e-4]},
        "xl": {256: [1.90e-4], 512: [2.68e-4], 1024: [3.79e-4], 2048: [5.37e-4],
               4096: [7.59e-4]},
    },
}
#: §1's count.
EXPECTED_ARMS = {"bert": 48, "gpt2": 20}
#: §2's own statement about the point count.
EXPECTED_EVAL_POINTS = 225


def three_sig(value: float) -> float:
    """The value rounded the way the directive's tables are written."""
    if value == 0:
        return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + 2)


def lr_slug(value: float) -> str:
    """``9.49e-05`` -> ``9p49e5``, which is how the existing arms are named."""
    mantissa, _, exponent = f"{value:.2e}".partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa.replace('.', 'p')}e{abs(int(exponent))}"


def scaled_rate(arch: str, size: str, batch: int, multiplier: float) -> float:
    """§3: the adopted 2,560 rate times sqrt(batch / 2560), times the multiplier."""
    base = BASE_LR_2560[arch][size]
    return three_sig(base * math.sqrt(batch / BASE_BATCH) * multiplier)


def steps_for(batch: int) -> int:
    if BUDGET_BLOCKS % batch:
        raise SystemExit(f"budget {BUDGET_BLOCKS:,} is not a multiple of batch {batch}")
    return BUDGET_BLOCKS // batch


def interval(batch: int, every_blocks: int, what: str) -> int:
    """§2: the interval in steps that lands on the same block position."""
    if every_blocks % batch:
        raise SystemExit(f"{what} every {every_blocks:,} blocks is not a whole number"
                         f" of steps at batch {batch}")
    return every_blocks // batch


def warmup_for(arch: str, steps: int) -> int:
    """§2: the framework's fraction of this arm's own length, truncated."""
    return int(WARMUP_FRACTION[arch] * steps)


def accumulation(arch: str, size: str, batch: int, which_set: str):
    """(micro_batch, grad_accum, world_size) that multiply out to this batch."""
    micro = MICRO_BATCH[which_set][arch][size]
    world = WORLD_SIZE[arch]
    divisor = micro * world * (GPU_COUNT if arch == "gpt2" else 1)
    if batch % divisor:
        raise SystemExit(
            f"{arch} {size}: batch {batch} is not a multiple of micro {micro}"
            f" x {divisor // micro}; §4 requires one GPU's block count to divide"
            f" every batch")
    return micro, batch // (micro * world), world


def grace_steps(batch: int, grace_blocks: int) -> int:
    """The grace window in this arm's own steps, from a block position."""
    return max(1, grace_blocks // batch)


def arms(which_set: str, grace_blocks: int | None, detector: bool):
    rows = []
    for arch in ARCHES:
        multipliers = BERT_MULTIPLIERS if arch == "bert" else GPT2_MULTIPLIERS
        for size in SIZES:
            for batch in BATCHES:
                steps = steps_for(batch)
                micro, accum, world = accumulation(arch, size, batch, which_set)
                rates = [scaled_rate(arch, size, batch, m) for m in multipliers[batch]]
                for rank, (multiplier, lr) in enumerate(zip(multipliers[batch], rates), 1):
                    slug = lr_slug(lr)
                    name = f"{arch}_{size}_b{batch}_lr{slug}"
                    out = (f"rna-{size}-b{batch}-lr{slug}" if arch == "bert"
                           else f"rna-gpt2-{size}-b{batch}-lr{slug}")
                    rows.append({
                        "arch": arch, "size": size, "batch": batch,
                        "lr": f"{lr:.6g}", "lr_multiplier": f"{multiplier:g}",
                        "lr_slug": slug, "rate_rank": rank,
                        "rates_at_this_batch": len(rates),
                        "steps": steps, "warmup_steps": warmup_for(arch, steps),
                        "eval_interval": interval(batch, EVAL_EVERY_BLOCKS, "evaluation"),
                        "save_interval": (interval(batch, SAVE_EVERY_BLOCKS, "saving")
                                          if arch == "bert" else ""),
                        "eval_points": BUDGET_BLOCKS // EVAL_EVERY_BLOCKS,
                        "save_points": (BUDGET_BLOCKS // SAVE_EVERY_BLOCKS
                                        if arch == "bert" else ""),
                        "blocks": BUDGET_BLOCKS,
                        "tokens": BUDGET_BLOCKS * SEQUENCE_LENGTH,
                        "micro_batch": micro, "grad_accum": accum,
                        "world_size": world, "gpu_count": GPU_COUNT,
                        "expected_global_batch": batch,
                        "eval_batch": BERT_EVAL_BATCH if arch == "bert" else "",
                        "min_lr": (f"{lr / GPT2_MIN_LR_DIVISOR:.6g}"
                                   if arch == "gpt2" else ""),
                        "lr_decay_iters": steps if arch == "gpt2" else "",
                        "judge_metric": JUDGE_METRIC[arch],
                        "degenerate_loss_threshold": (DEGENERATE_THRESHOLD
                                                      if detector and arch == "bert"
                                                      else ""),
                        "degenerate_patience": (DEGENERATE_PATIENCE
                                                if detector and arch == "bert" else ""),
                        "degenerate_grace_steps": (
                            grace_steps(batch, grace_blocks)
                            if detector and arch == "bert" and grace_blocks else ""),
                        "micro_batch_set": which_set,
                        "config": f"molcrawl/tasks/pretrain/configs/rna/{name}.py",
                        "out_dir": out,
                    })
    return rows


def check(rows):
    """Compare the derived arms against the directive's printed tables."""
    problems = []
    counted = {arch: 0 for arch in ARCHES}
    for row in rows:
        arch, size, batch = row["arch"], row["size"], row["batch"]
        counted[arch] += 1
        where = f"{arch} {size} b{batch} lr{row['lr_slug']}"
        if row["steps"] != EXPECTED_STEPS[batch]:
            problems.append(f"{where}: steps {row['steps']:,}, table says"
                            f" {EXPECTED_STEPS[batch]:,}")
        want_eval, want_save = EXPECTED_INTERVALS[batch]
        if row["eval_interval"] != want_eval:
            problems.append(f"{where}: eval every {row['eval_interval']:,} steps,"
                            f" table says {want_eval:,}")
        if arch == "bert" and row["save_interval"] != want_save:
            problems.append(f"{where}: save every {row['save_interval']:,} steps,"
                            f" table says {want_save:,}")
        if row["warmup_steps"] != EXPECTED_WARMUP[arch][batch]:
            problems.append(f"{where}: warmup {row['warmup_steps']:,}, table says"
                            f" {EXPECTED_WARMUP[arch][batch]:,}")
        if row["eval_points"] != EXPECTED_EVAL_POINTS:
            problems.append(f"{where}: {row['eval_points']} evaluation points,"
                            f" §2 says {EXPECTED_EVAL_POINTS}")
        want = EXPECTED_LR[arch][size][batch][row["rate_rank"] - 1]
        got = float(row["lr"])
        if abs(got - want) > abs(want) * 5e-3:
            problems.append(f"{where}: rate {got:.3g}, table says {want:.3g}")
        if row["micro_batch"] * row["grad_accum"] * row["world_size"] != batch:
            problems.append(f"{where}: micro {row['micro_batch']} x accum"
                            f" {row['grad_accum']} x world {row['world_size']}"
                            f" is not {batch}")
    for arch, want in EXPECTED_ARMS.items():
        if counted[arch] != want:
            problems.append(f"{arch}: {counted[arch]} arms, §1 says {want}")
    return problems


def main() -> int:
    # The probe table module sits beside this file, which is not the
    # working directory when this runs from the repository root.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--micro-batch-set", default="current",
                    choices=sorted(set(MICRO_BATCH) | {"measured"}),
                    help="which one-GPU block count table to build with;"
                         " 'measured' reads it from --measured-from")
    ap.add_argument("--measured-from", action="append", default=[],
                    help="a probe table TSV to take the 'measured' set from"
                         " (scripts/rna_microbatch_probe_table.py --out); repeat it"
                         " when the two frameworks were measured separately, as they"
                         " were here -- BERT on 2026-11-20 and GPT-2 on 2026-11-23."
                         " A later file wins where they overlap")
    ap.add_argument("--no-collapse-detection", action="store_true",
                    help="leave the detector's values out (approved on 2026-11-20,"
                         " so it is written by default)")
    ap.add_argument("--grace-blocks", type=int, default=GRACE_BLOCKS,
                    help="block position a healthy arm is past the threshold by;"
                         " the grace window of each arm is this divided by its batch"
                         f" (default {GRACE_BLOCKS:,}, measured)")
    ap.add_argument("--out", help="TSV to write")
    args = ap.parse_args()

    if (args.micro_batch_set == "measured") != bool(args.measured_from):
        print("--micro-batch-set measured and --measured-from go together: the"
              " measured block counts are not in this source, they are read from"
              " the probe's table", file=sys.stderr)
        return 2
    if args.measured_from:
        import rna_microbatch_probe_table as probe

        measured: dict[str, dict[str, int]] = {}
        for path in args.measured_from:
            table = probe.read_table(path)
            if any(int(r.get("repeat", 1) or 1) > 1 for r in table):
                table = probe.collapse_repeats(table)
            part = probe.fastest_micro(table)
            for arch, sizes in part.items():
                measured.setdefault(arch, {}).update(sizes)
            print(f"=== {path}: {part}")
        thin = {arch: sorted(set(SIZES) - set(measured.get(arch, {})))
                for arch in ARCHES}
        if any(thin.values()):
            print(f"no block count measured for {thin}: pass another --measured-from",
                  file=sys.stderr)
            return 2
        MICRO_BATCH["measured"] = measured
        print(f"=== measured set: {measured}")

    detector = not args.no_collapse_detection
    if detector and not args.grace_blocks:
        print("the detector needs --grace-blocks: the grace window is compared"
              " against the absolute step count, so it cannot be one number across"
              " batches that differ 16-fold", file=sys.stderr)
        return 2

    rows = arms(args.micro_batch_set, args.grace_blocks, detector)
    problems = check(rows)

    print(f"=== {len(rows)} arms, micro-batch set {args.micro_batch_set!r}")
    for arch in ARCHES:
        n = sum(1 for r in rows if r["arch"] == arch)
        print(f"    {arch}: {n}")
    print(f"=== budget {BUDGET_BLOCKS:,} blocks ="
          f" {BUDGET_BLOCKS * SEQUENCE_LENGTH / 1e9:.2f} G tokens,"
          f" {BUDGET_BLOCKS // EVAL_EVERY_BLOCKS} evaluation points per arm")
    print(f"=== 指示書の表との食い違い {len(problems)} 件")
    for problem in problems:
        print(f"    - {problem}")

    for arch in ARCHES:
        print(f"\n== {arch} の 1 GPU 塊数と accumulation (§9)")
        print(f"   {'size':8s} {'micro':>6s} " +
              " ".join(f"{'b' + str(b):>9s}" for b in BATCHES))
        for size in SIZES:
            micro = MICRO_BATCH[args.micro_batch_set][arch][size]
            cells = []
            for batch in BATCHES:
                row = next(r for r in rows if r["arch"] == arch and r["size"] == size
                           and r["batch"] == batch)
                cells.append(f"{row['grad_accum']:>9,}")
            print(f"   {size:8s} {micro:>6d} " + " ".join(cells))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("#! report-tsv 1\n")
            handle.write("# model: several (columns: arch,size)\n")
            handle.write("# note: the rna batch-size grid of the 2026-11-20 directive,"
                         " derived from its rules and checked against its tables.\n")
            handle.write(f"# note: one-GPU block count set: {args.micro_batch_set}."
                         " current = what the rna arms run today; compounds-prior ="
                         " the fastest shape measured on compounds, a prior, not an rna"
                         " measurement.\n")
            handle.write("# note: out_dir is the arm's directory name. BERT resolves it"
                         " under MODEL_OUTPUT_ROOT/rna/bert-output and GPT-2 directly"
                         " under MODEL_OUTPUT_ROOT; the launch sets that root away from"
                         " the trees the 18-epoch extension writes.\n")
            handle.write("\t".join(COLUMNS) + "\n")
            for row in rows:
                handle.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        print(f"\nwrote {args.out}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
