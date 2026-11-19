"""Derive the compounds batch-size grid's arms, and check the directive's own tables.

The 2026-11-19b directive prints the step counts, the evaluation and save intervals
and the learning rates it wants. All three are derivable from rules the same
directive states:

    a fixed number of columns, so the step count is columns / batch (§2)
    intervals that put every arm's points at the same column position (§2)
    the rate scaled by sqrt(batch / 2560) off the existing 2,560 arm (§3)

So this derives them from the rules and compares the result against the printed
tables, which are written out below as expectations. A cell that disagrees is
reported rather than silently preferred: either the rule or the table is wrong, and
which one it is cannot be settled by copying one into the other.

    compounds_batch_sweep_arms.py --out rt1-...tsv [--bert-micro-batch 32]

The micro-batch is an argument because §7 lists "BERT 8" among the things not to
change while the four existing compounds BERT arms run at 32 (32 x 20 x 4 GPUs =
2,560). The two readings differ in throughput, not in the global batch, so the
inventory can be produced either way and the difference is visible in the columns.

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

#: §2. The budget, in columns (sequences). BERT is the existing 45,000 x 2,560;
#: GPT-2 is the existing 4,674 x 2,560 rounded up to a multiple of every batch.
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

#: §6. warmup is 10 % of the arm's own step count.
WARMUP_FRACTION = 0.10

#: The tables as the directive prints them, to three significant figures. Nothing
#: below reads these except the comparison.
EXPECTED_STEPS = {
    "bert": {512: 225_000, 1024: 112_500, 2048: 56_250, 4096: 28_125},
    "gpt2": {512: 23_400, 1024: 11_700, 2048: 5_850, 4096: 2_925},
}
EXPECTED_INTERVALS = {
    "bert": {512: (1000, 5000), 1024: (500, 2500), 2048: (250, 1250), 4096: (125, 625)},
    "gpt2": {512: (80, None), 1024: (40, None), 2048: (20, None), 4096: (10, None)},
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

#: §7. One GPU's micro-batch. GPT-2 computes its global batch as micro x accumulation
#: with no world-size factor (the existing config says 16 x 160 = 2,560); BERT
#: multiplies by the GPU count as well (32 x 20 x 4 = 2,560).
GPT2_MICRO_BATCH = 16
BERT_WORLD_SIZE = 4

COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "lr_slug", "steps",
           "warmup_steps", "eval_interval", "save_interval", "rows", "tokens",
           "micro_batch", "grad_accum", "world_size", "expected_global_batch",
           "config", "out_dir"]


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


def accumulation(arch: str, batch: int, bert_micro: int) -> tuple[int, int, int]:
    """(micro_batch, grad_accum, world_size) that multiply out to this batch."""
    if arch == "gpt2":
        micro, world = GPT2_MICRO_BATCH, 1
    else:
        micro, world = bert_micro, BERT_WORLD_SIZE
    divisor = micro * world
    if batch % divisor:
        raise SystemExit(
            f"{arch} batch {batch} is not a multiple of micro {micro} x world {world}")
    return micro, batch // divisor, world


def arms(bert_micro: int):
    rows = []
    for arch in ("bert", "gpt2"):
        multipliers = BERT_MULTIPLIERS if arch == "bert" else GPT2_MULTIPLIERS
        for size in SIZES:
            for batch in BATCHES:
                steps = steps_for(arch, batch)
                micro, accum, world = accumulation(arch, batch, bert_micro)
                warmup = WARMUP_FRACTION * steps
                for multiplier in multipliers[batch]:
                    lr = scaled_rate(arch, size, batch, multiplier)
                    slug = lr_slug(lr)
                    stem = f"{arch}_{size}_b{batch}_lr{slug}"
                    rows.append({
                        "arch": arch, "size": size, "batch": batch, "lr": f"{lr:.2e}",
                        "lr_multiplier": f"{multiplier:.1f}", "lr_slug": slug,
                        "steps": steps,
                        # Reported as written so a non-whole warmup is visible rather
                        # than rounded out of sight.
                        "warmup_steps": (int(warmup) if warmup == int(warmup)
                                         else f"{warmup}"),
                        "eval_interval": interval(arch, batch, "eval"),
                        "save_interval": interval(arch, batch, "save"),
                        "rows": BUDGET_ROWS[arch],
                        "tokens": BUDGET_ROWS[arch] * SEQUENCE_LENGTH,
                        "micro_batch": micro, "grad_accum": accum, "world_size": world,
                        "expected_global_batch": batch,
                        "config": f"molcrawl/tasks/pretrain/configs/compounds/{stem}.py",
                        "out_dir": f"compounds-{size}-b{batch}-lr{slug}",
                    })
    return rows


def check(rows) -> int:
    """Compare the derivation against the directive's printed tables."""
    problems = 0
    for arch in ("bert", "gpt2"):
        for batch in BATCHES:
            got = steps_for(arch, batch)
            want = EXPECTED_STEPS[arch][batch]
            mark = "ok" if got == want else "**"
            if got != want:
                problems += 1
            print(f"  {mark} {arch:5s} batch {batch:5d}  steps {got:>7,} "
                  f"(directive {want:>7,})")
            want_eval, want_save = EXPECTED_INTERVALS[arch][batch]
            for which, wanted in (("eval", want_eval), ("save", want_save)):
                if wanted is None:
                    continue
                g = interval(arch, batch, which)
                if g != wanted:
                    problems += 1
                    print(f"     ** {which} interval {g} against directive {wanted}")

    for arch in ("bert", "gpt2"):
        for size in SIZES:
            for batch in BATCHES:
                got = sorted(float(r["lr"]) for r in rows
                             if r["arch"] == arch and r["size"] == size
                             and r["batch"] == batch)
                want = sorted(EXPECTED_LR[arch][size][batch])
                if [three_sig(v) for v in got] != [three_sig(v) for v in want]:
                    problems += 1
                    print(f"  ** {arch} {size} batch {batch}: derived {got} "
                          f"against directive {want}")
    print(f"\n  rates compared: {sum(len(EXPECTED_LR[a][s][b]) for a in EXPECTED_LR for s in SIZES for b in BATCHES)}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert-micro-batch", type=int, default=32,
                    help="one GPU's BERT micro-batch; 32 is what the existing "
                         "compounds arms run at, 8 is what §7 names")
    ap.add_argument("--out", required=True,
                    help="the base name begins with rt1- (the column convention)")
    args = ap.parse_args(argv)
    check_name(args.out)

    rows = arms(args.bert_micro_batch)
    print(f"=== derived {len(rows)} arms "
          f"(BERT {sum(1 for r in rows if r['arch'] == 'bert')}, "
          f"GPT-2 {sum(1 for r in rows if r['arch'] == 'gpt2')}) ===")
    print("\n=== against the directive's printed tables ===")
    problems = check(rows)

    print("\n=== per batch ===")
    for arch in ("bert", "gpt2"):
        for batch in BATCHES:
            sel = [r for r in rows if r["arch"] == arch and r["batch"] == batch]
            one = sel[0]
            print(f"  {arch:5s} batch {batch:5d}  {len(sel):2d} arms  "
                  f"steps {one['steps']:>7,}  warmup {one['warmup_steps']:>8}  "
                  f"eval {one['eval_interval']:>5}  save {str(one['save_interval']):>5}  "
                  f"{one['micro_batch']} x {one['grad_accum']} x {one['world_size']}")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "several (columns: arch,size,batch,lr) -- the compounds batch-size "
                 "grid planned by the 2026-11-19b directive",
        "source": "derived from the directive's rules (budget, interval anchoring, "
                  "sqrt scaling) and compared against its printed tables; no run has "
                  "been launched",
        "generated": stamp("scripts/compounds_batch_sweep_arms.py"),
        "note": [
            f"budget: BERT {BUDGET_ROWS['bert']:,} columns, GPT-2 "
            f"{BUDGET_ROWS['gpt2']:,} columns, sequence length {SEQUENCE_LENGTH}",
            f"BERT micro-batch {args.bert_micro_batch} x world size {BERT_WORLD_SIZE}; "
            f"GPT-2 micro-batch {GPT2_MICRO_BATCH} with no world-size factor",
            "lr = the 2,560 arm's rate x sqrt(batch/2560) x the column's multiplier, "
            "to three significant figures",
            "no arm has been launched; config and out_dir are planned names",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"\n{len(rows)} arms -> {args.out}")
    print(f"disagreements with the directive: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
