"""Derive the compounds batch-size grid's arms, and check the directive's own tables.

The 2026-11-19c directive prints the step counts, the evaluation and save intervals,
the warmup lengths, the accumulation counts and the 56 learning rates it wants. All of
them follow from rules the same directive states:

    a fixed number of columns, so the step count is columns / batch (§2)
    intervals that put every arm's points at the same column position (§2)
    the rate scaled by sqrt(batch / 2560) off the existing 2,560 arm (§3)
    warmup at the framework's own fraction, truncated (§4)
    one GPU's column count held across the four batches of a size, so the
    accumulation count carries the whole difference (the micro-batch directive
    of 2026-11-19, which replaces 19c §8)

So this derives them from the rules and compares the result against the printed
tables, which are written out below as expectations. A cell that disagrees is
reported rather than silently preferred: either the rule or the table is wrong, and
which one it is cannot be settled by copying one into the other.

    compounds_batch_sweep_arms.py --out rt1-...tsv [--micro-batch-set micro-order]

One GPU's column count went through a correction. 19c §8 stated it twice and not the
same way: its tables named BERT medium / large / xl 8 while its sentence said each
size keeps its current value, and the adopted arms run at medium 160, large 64, xl 32
-- a value that divides none of the four batches at 4 GPUs. The directive of
2026-11-19 on this one question replaces §8 and settles it on the fastest shape that
does divide them. That is what this writes; the superseded table is kept as
``--micro-batch-set 19c-s8`` so a run started under it could be reproduced. The two
differ in throughput, not in the global batch, and which one was used is written into
every row and into the provenance.

Read-only: it computes and writes one TSV.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: §1. Four batches, four sizes, both frameworks.
BATCHES = [512, 1024, 2048, 4096]
SIZES = ["small", "medium", "large", "xl"]

#: §2. The budget, in columns (packed 1,024-token blocks). BERT is the existing
#: 45,000 x 2,560; GPT-2 is the existing 4,674 iterations rounded to a multiple of
#: every batch.
BUDGET_ROWS = {"bert": 115_200_000, "gpt2": 11_980_800}
SEQUENCE_LENGTH = 1024

#: The existing 2,560 arm whose rate each column is scaled from (§3).
BASE_LR_2560 = {
    "bert": {"small": 1e-3, "medium": 2.8e-4, "large": 1.5e-4, "xl": 4.5e-5},
    "gpt2": {"small": 8e-4, "medium": 8e-4, "large": 1.6e-3, "xl": 1.6e-3},
}
BASE_BATCH = 2560

#: §3. BERT runs several rates per batch and keeps the lowest; GPT-2 runs one.
#: The multipliers are read off the printed table: 0.7 and 1.0 where there are two,
#: 0.7, 1.0 and 1.4 where there are three.
BERT_MULTIPLIERS = {512: [0.7, 1.0], 1024: [0.7, 1.0],
                    2048: [0.7, 1.0, 1.4], 4096: [0.7, 1.0, 1.4]}
GPT2_MULTIPLIERS = {b: [1.0] for b in BATCHES}

#: §2's intervals at batch 512. Every other batch scales them by 512 / batch, which is
#: what makes the four curves carry points at the same column positions.
ANCHOR_BATCH = 512
ANCHOR_INTERVALS = {
    "bert": {"eval": 1000, "save": 5000},
    "gpt2": {"eval": 80},
}

#: §4. warmup is the framework's own current fraction of its own step count, truncated.
WARMUP_FRACTION = {"bert": 0.10, "gpt2": 0.02}

#: One GPU's column count. ``micro-order`` is the directive of 2026-11-19 on this one
#: question (compounds-batch-micro-order-2026-11-19.md §1), which replaces §8 of the
#: 19c directive: the fastest shape that divides all four batches, measured on
#: compounds at 4 GPUs, bf16, global batch 2,560 -- small 3.13 s/step at 32, medium
#: 7.35 at 64 (7.21 at 160, which divides none of the four batches), large 12.14 at 64,
#: xl 23.70 at 32. ``19c-s8`` is the superseded table, kept so a run started under it
#: can still be reproduced; it named 8 for medium, large and xl, which is 1.84 to 1.97
#: times slower (14.47, 22.35 and 43.72 s/step).
MICRO_BATCH = {
    "micro-order": {
        "bert": {"small": 32, "medium": 64, "large": 64, "xl": 32},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
    "19c-s8": {
        "bert": {"small": 32, "medium": 8, "large": 8, "xl": 8},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
}

#: GPT-2 computes its global batch as micro x accumulation with no world-size factor
#: (the existing config says 16 x 160 = 2,560); BERT multiplies by the GPU count as
#: well (32 x 20 x 4 = 2,560). §6 fixes the GPU count at 4 for BERT.
WORLD_SIZE = {"bert": 4, "gpt2": 1}

#: 19c §6. One evaluation micro-batch for all 40 BERT arms. 32 is the largest value xl is
#: on record as having run at (every adopted compounds xl arm scores at 32), so it is
#: chosen rather than a larger untried one.
BERT_EVAL_BATCH = 32

#: §7. The metric the arm is judged and kept on.
JUDGE_METRIC = {"bert": "eval_loss", "gpt2": "val_loss"}

#: §4. GPT-2's floor is a tenth of its own peak, and its decay length is its own run.
GPT2_MIN_LR_DIVISOR = 10

# ---------------------------------------------------------------------------
# The tables as the directive prints them. Nothing below reads these except check().
# ---------------------------------------------------------------------------

EXPECTED_STEPS = {
    "bert": {512: 225_000, 1024: 112_500, 2048: 56_250, 4096: 28_125},
    "gpt2": {512: 23_400, 1024: 11_700, 2048: 5_850, 4096: 2_925},
}
EXPECTED_INTERVALS = {
    "bert": {512: (1000, 5000), 1024: (500, 2500), 2048: (250, 1250), 4096: (125, 625)},
    "gpt2": {512: (80, None), 1024: (40, None), 2048: (20, None), 4096: (10, None)},
}
EXPECTED_WARMUP = {
    "bert": {512: 22_500, 1024: 11_250, 2048: 5_625, 4096: 2_812},
    "gpt2": {512: 468, 1024: 234, 2048: 117, 4096: 58},
}
#: The micro-batch directive's second table (§1), and GPT-2's from 19c §8, which that
#: directive leaves standing ("GPT-2 の 16 本は変更しない").
EXPECTED_ACCUM = {
    ("bert", "small"): {512: 4, 1024: 8, 2048: 16, 4096: 32},
    ("bert", "medium"): {512: 2, 1024: 4, 2048: 8, 4096: 16},
    ("bert", "large"): {512: 2, 1024: 4, 2048: 8, 4096: 16},
    ("bert", "xl"): {512: 4, 1024: 8, 2048: 16, 4096: 32},
    ("gpt2", "small"): {512: 32, 1024: 64, 2048: 128, 4096: 256},
    ("gpt2", "medium"): {512: 32, 1024: 64, 2048: 128, 4096: 256},
    ("gpt2", "large"): {512: 64, 1024: 128, 2048: 256, 4096: 512},
    ("gpt2", "xl"): {512: 128, 1024: 256, 2048: 512, 4096: 1024},
}
EXPECTED_LR = {
    "bert": {
        "small": {512: [3.13e-4, 4.47e-4], 1024: [4.43e-4, 6.32e-4],
                  2048: [6.26e-4, 8.94e-4, 1.25e-3], 4096: [8.85e-4, 1.26e-3, 1.77e-3]},
        "medium": {512: [8.77e-5, 1.25e-4], 1024: [1.24e-4, 1.77e-4],
                   2048: [1.75e-4, 2.50e-4, 3.51e-4], 4096: [2.48e-4, 3.54e-4, 4.96e-4]},
        "large": {512: [4.70e-5, 6.71e-5], 1024: [6.64e-5, 9.49e-5],
                  2048: [9.39e-5, 1.34e-4, 1.88e-4], 4096: [1.33e-4, 1.90e-4, 2.66e-4]},
        "xl": {512: [1.41e-5, 2.01e-5], 1024: [1.99e-5, 2.85e-5],
               2048: [2.82e-5, 4.02e-5, 5.63e-5], 4096: [3.98e-5, 5.69e-5, 7.97e-5]},
    },
    "gpt2": {
        "small": {512: [3.58e-4], 1024: [5.06e-4], 2048: [7.16e-4], 4096: [1.01e-3]},
        "medium": {512: [3.58e-4], 1024: [5.06e-4], 2048: [7.16e-4], 4096: [1.01e-3]},
        "large": {512: [7.16e-4], 1024: [1.01e-3], 2048: [1.43e-3], 4096: [2.02e-3]},
        "xl": {512: [7.16e-4], 1024: [1.01e-3], 2048: [1.43e-3], 4096: [2.02e-3]},
    },
}
#: §1's count.
EXPECTED_ARMS = {"bert": 40, "gpt2": 16}

COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "lr_slug", "steps",
           "warmup_steps", "eval_interval", "save_interval", "rows", "tokens",
           "micro_batch", "grad_accum", "world_size", "expected_global_batch",
           "eval_batch", "min_lr", "lr_decay_iters", "judge_metric",
           "micro_batch_set", "config", "out_dir"]


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
    """§2: the anchor interval scaled so the points land at the same column."""
    anchor = ANCHOR_INTERVALS[arch].get(which)
    if anchor is None:
        return None
    scaled = anchor * ANCHOR_BATCH / batch
    if scaled != int(scaled):
        raise SystemExit(f"{arch} {which} interval at batch {batch} is {scaled}, not whole")
    return int(scaled)


def steps_for(arch: str, batch: int) -> int:
    rows = BUDGET_ROWS[arch]
    if rows % batch:
        raise SystemExit(f"{arch} budget {rows:,} is not a multiple of batch {batch}")
    return rows // batch


def warmup_for(arch: str, steps: int) -> int:
    """§4: the framework's fraction of this arm's own length, truncated."""
    return int(WARMUP_FRACTION[arch] * steps)


def accumulation(arch: str, size: str, batch: int, which_set: str):
    """(micro_batch, grad_accum, world_size) that multiply out to this batch."""
    micro = MICRO_BATCH[which_set][arch][size]
    world = WORLD_SIZE[arch]
    divisor = micro * world
    if batch % divisor:
        raise SystemExit(f"{arch} {size} batch {batch} is not a multiple of "
                         f"micro {micro} x world {world}")
    return micro, batch // divisor, world


def arms(which_set: str):
    rows = []
    for arch in ("bert", "gpt2"):
        multipliers = BERT_MULTIPLIERS if arch == "bert" else GPT2_MULTIPLIERS
        for size in SIZES:
            for batch in BATCHES:
                steps = steps_for(arch, batch)
                micro, accum, world = accumulation(arch, size, batch, which_set)
                for multiplier in multipliers[batch]:
                    lr = scaled_rate(arch, size, batch, multiplier)
                    slug = lr_slug(lr)
                    stem = f"{arch}_{size}_b{batch}_lr{slug}"
                    rows.append({
                        "arch": arch, "size": size, "batch": batch, "lr": f"{lr:.2e}",
                        "lr_multiplier": f"{multiplier:.1f}", "lr_slug": slug,
                        "steps": steps,
                        "warmup_steps": warmup_for(arch, steps),
                        "eval_interval": interval(arch, batch, "eval"),
                        "save_interval": interval(arch, batch, "save"),
                        "rows": BUDGET_ROWS[arch],
                        "tokens": BUDGET_ROWS[arch] * SEQUENCE_LENGTH,
                        "micro_batch": micro, "grad_accum": accum, "world_size": world,
                        "expected_global_batch": batch,
                        "eval_batch": BERT_EVAL_BATCH if arch == "bert" else None,
                        "min_lr": (f"{three_sig(lr / GPT2_MIN_LR_DIVISOR):.2e}"
                                   if arch == "gpt2" else None),
                        "lr_decay_iters": steps if arch == "gpt2" else None,
                        "judge_metric": JUDGE_METRIC[arch],
                        "micro_batch_set": which_set,
                        "config": f"molcrawl/tasks/pretrain/configs/compounds/{stem}.py",
                        "out_dir": f"compounds-{size}-b{batch}-lr{slug}",
                    })
    return rows


def check(rows, which_set: str) -> int:
    """Compare the derivation against the directive's printed tables."""
    problems = 0

    for arch in ("bert", "gpt2"):
        got = sum(1 for r in rows if r["arch"] == arch)
        want = EXPECTED_ARMS[arch]
        if got != want:
            problems += 1
            print(f"  ** {arch}: {got} arms against directive {want}")

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
                sel = [r for r in rows if r["arch"] == arch and r["size"] == size
                       and r["batch"] == batch]
                got = sorted(float(r["lr"]) for r in sel)
                want = sorted(EXPECTED_LR[arch][size][batch])
                if [three_sig(v) for v in got] != [three_sig(v) for v in want]:
                    problems += 1
                    print(f"  ** {arch} {size} batch {batch}: derived {got} "
                          f"against directive {want}")
                accum = {r["grad_accum"] for r in sel}
                wanted = EXPECTED_ACCUM[(arch, size)][batch]
                if accum != {wanted}:
                    problems += 1
                    print(f"  ** {arch} {size} batch {batch}: accumulation "
                          f"{sorted(accum)} against directive {wanted}"
                          + ("" if which_set == "micro-order"
                             else f" (micro-batch set '{which_set}', not the table)"))

    print(f"\n  compared: {sum(len(EXPECTED_LR[a][s][b]) for a in EXPECTED_LR for s in SIZES for b in BATCHES)} rates, "
          f"{len(EXPECTED_ACCUM) * len(BATCHES)} accumulation counts, "
          f"{2 * len(BATCHES)} step counts, {2 * len(BATCHES)} warmup lengths")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--micro-batch-set", choices=sorted(MICRO_BATCH),
                    default="micro-order",
                    help="one GPU's column count: 'micro-order' is the 2026-11-19 "
                         "micro-batch directive, '19c-s8' the table it replaced")
    ap.add_argument("--out", required=True,
                    help="the base name begins with rt1- (the column convention)")
    args = ap.parse_args(argv)
    check_name(args.out)

    rows = arms(args.micro_batch_set)
    print(f"=== derived {len(rows)} arms "
          f"(BERT {sum(1 for r in rows if r['arch'] == 'bert')}, "
          f"GPT-2 {sum(1 for r in rows if r['arch'] == 'gpt2')}), "
          f"micro-batch set '{args.micro_batch_set}' ===")
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
        "model": "several (columns: arch,size,batch,lr) -- the compounds batch-size "
                 "grid asked for by the 2026-11-19c directive",
        "source": "derived from the directive's rules (budget; interval anchoring; "
                  "sqrt scaling; warmup fraction; one GPU's column count) and compared "
                  "against its printed tables; no arm has been launched",
        "generated": stamp("scripts/compounds_batch_sweep_arms.py"),
        "note": [
            f"budget: BERT {BUDGET_ROWS['bert']:,} columns, GPT-2 "
            f"{BUDGET_ROWS['gpt2']:,} columns, sequence length {SEQUENCE_LENGTH}",
            f"one GPU's column count from the '{args.micro_batch_set}' set; "
            f"BERT x {WORLD_SIZE['bert']} GPUs, GPT-2 with no world-size factor",
            "lr = the 2,560 arm's rate x sqrt(batch/2560) x the column's multiplier, "
            "to three significant figures",
            f"warmup: BERT {WARMUP_FRACTION['bert']:.0%} and GPT-2 "
            f"{WARMUP_FRACTION['gpt2']:.0%} of the arm's own length, truncated",
            f"BERT evaluation micro-batch {BERT_EVAL_BATCH} on all 40 arms; judged on "
            f"{JUDGE_METRIC['bert']}, GPT-2 on {JUDGE_METRIC['gpt2']}",
            "no arm has been launched; config and out_dir are planned names",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"\n{len(rows)} arms -> {args.out}")
    print(f"disagreements with the directive: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
