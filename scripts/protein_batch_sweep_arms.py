"""Derive the protein batch-size grid's 59 arms and check the directive's own tables.

The 2026-11-20 protein batch-sweep directive (protein-batch-sweep-order-2026-11-20b)
mirrors the compounds grid. Everything follows from rules the directive states:

    one budget in columns for both frameworks, so steps = budget / batch (section 2)
    eval every 380,928 columns and save every 1,904,640, so the five curves carry
      points at the same column positions (section 2)
    the rate scaled by sqrt(batch/2560) off the adopted 2,560 arm, fanned DOWN only
      -- 0.7, 1.0 where there are two rates, 0.5, 0.7, 1.0 where there are three
      (section 3: protein collapses 1.3-1.4x above the best, so no rate goes above centre)
    warmup at the framework's own fraction, truncated (section 2 table)
    one GPU's column count MEASURED (section 4): BERT all 32 (this session, jobs
      374634-374637), GPT-2 from its own probe; the accumulation count carries the rest

Grid = 56 arms (BERT small/medium/large x 12, GPT-2 four sizes x 5). The BERT xl
diagnostic of section 1.1 adds 3 explicit arms at batch 256 (rates given, not scaled),
for 59 in all.

    protein_batch_sweep_arms.py --out rt1-...tsv [--gpt2-micro adopted|measured]

Read-only: computes and writes one report-tsv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math

BATCHES = [256, 512, 1024, 2048, 4096]
BERT_SIZES = ["small", "medium", "large"]
GPT2_SIZES = ["small", "medium", "large", "xl"]

# section 2: one budget in columns (packed 1,024-token blocks), both frameworks.
# 85,708,800 = 33,531 x 2,560 rounded to a multiple of all five batches (8.986 epochs).
BUDGET_ROWS = 85_708_800
SEQUENCE_LENGTH = 1024
EVAL_EVERY = 380_928        # columns between evaluations (225 points over the budget)
SAVE_EVERY = 1_904_640      # = 5 x EVAL_EVERY

WARMUP_FRACTION = {"bert": 0.10, "gpt2": 0.02}

# section 3: the adopted 2,560 arm whose rate each column scales from.
#   BERT: the seed-42 adopted arms (rt1-protein-bert-arms-20261120.tsv).
#   GPT-2: schedule.learning_rate from each adopted -rerun run_manifest.json.
BASE_LR_2560 = {
    "bert": {"small": 3.0e-4, "medium": 1.0e-4, "large": 3.5e-5},
    "gpt2": {"small": 1.2e-3, "medium": 6.0e-4, "large": 6.0e-4, "xl": 6.0e-4},
}
BASE_BATCH = 2560

# section 3: BERT runs several rates per batch and keeps the lowest; GPT-2 runs one.
# Protein fans DOWN from the centre (never above it): 2 rates -> 0.7, 1.0; 3 -> 0.5, 0.7, 1.0.
BERT_MULTIPLIERS = {256: [0.7, 1.0], 512: [0.7, 1.0], 1024: [0.7, 1.0],
                    2048: [0.5, 0.7, 1.0], 4096: [0.5, 0.7, 1.0]}
GPT2_MULTIPLIERS = {b: [1.0] for b in BATCHES}

# section 4: one GPU's column count, MEASURED this session.
#   BERT (jobs 374634-374637, bf16, 4 GPUs): every size fastest at micro 32 (micro 64
#     OOMs at xl); report/rt1-protein-bert-microbatch-20261120.tsv.
#   GPT-2 'measured' is filled from its own probe; 'adopted' is the -rerun arms' micro
#     (small/medium 16, large 8, xl 4) used as a provisional until the probe returns.
MICRO_BATCH = {
    "measured": {
        "bert": {"small": 32, "medium": 32, "large": 32, "xl": 32},
        # GPT-2 measured 2026-11-23 (jobs 384796/384802/384803/384804, bf16, 4 GPUs):
        # fastest tok/s at small 32, medium 64, large 32, xl 32 (micro 64 OOMs at xl and
        # is anomalously slow at small/large). Differs from the adopted 16/16/8/4.
        "gpt2": {"small": 32, "medium": 64, "large": 32, "xl": 32},
    },
    "adopted": {
        "bert": {"small": 32, "medium": 32, "large": 32, "xl": 32},
        "gpt2": {"small": 16, "medium": 16, "large": 8, "xl": 4},
    },
}

# BERT effective = micro x accum x GPUs; GPT-2 effective = micro x accum (nanoGPT
# divides the declared accum by world size). section 6 fixes 4 GPUs for BERT.
WORLD_SIZE = {"bert": 4, "gpt2": 1}

# section 6: one evaluation micro-batch for every BERT arm (grid + the xl diagnostic).
# 32 is the training micro xl is measured to fit (micro 64 OOMs); eval is forward-only,
# so it fits too. Held identical so the fixed-mask grouping is the same on every arm.
BERT_EVAL_BATCH = 32

# section 7.
JUDGE_METRIC = {"bert": "eval_loss_mask", "gpt2": "val_loss"}
GPT2_MIN_LR_DIVISOR = 10  # section 8

# section 1.1: the BERT xl diagnostic. Batch 256 only, three EXPLICIT rates (not scaled):
# 1.2e-5 extrapolates the adopted-arm rates (3e-4 -> 1e-4 -> 3.5e-5, ~1/3 per size) to xl;
# 3.79e-6 is that put to batch 256 by the sqrt rule; 7.59e-6 is twice it.
XL_DIAG_BATCH = 256
XL_DIAG_RATES = [3.79e-6, 7.59e-6, 1.20e-5]

# --- the directive's printed tables; read only by check() -------------------
EXPECTED_STEPS = {256: 334_800, 512: 167_400, 1024: 83_700, 2048: 41_850, 4096: 20_925}
EXPECTED_EVAL = {256: 1488, 512: 744, 1024: 372, 2048: 186, 4096: 93}
EXPECTED_SAVE = {256: 7440, 512: 3720, 1024: 1860, 2048: 930, 4096: 465}
EXPECTED_WARMUP = {
    "bert": {256: 33_480, 512: 16_740, 1024: 8_370, 2048: 4_185, 4096: 2_092},
    "gpt2": {256: 6_696, 512: 3_348, 1024: 1_674, 2048: 837, 4096: 418},
}
EXPECTED_BERT_LR = {
    "small": {256: [6.64e-5, 9.49e-5], 512: [9.39e-5, 1.34e-4], 1024: [1.33e-4, 1.90e-4],
              2048: [1.34e-4, 1.88e-4, 2.68e-4], 4096: [1.90e-4, 2.66e-4, 3.79e-4]},
    "medium": {256: [2.21e-5, 3.16e-5], 512: [3.13e-5, 4.47e-5], 1024: [4.43e-5, 6.32e-5],
               2048: [4.47e-5, 6.26e-5, 8.94e-5], 4096: [6.32e-5, 8.85e-5, 1.26e-4]},
    "large": {256: [7.75e-6, 1.11e-5], 512: [1.10e-5, 1.57e-5], 1024: [1.55e-5, 2.21e-5],
              2048: [1.57e-5, 2.19e-5, 3.13e-5], 4096: [2.21e-5, 3.10e-5, 4.43e-5]},
}
EXPECTED_COUNTS = {"bert_grid": 36, "gpt2": 20, "bert_xl_diag": 3}

COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "lr_slug", "steps",
           "warmup_steps", "eval_interval", "save_interval", "rows", "tokens",
           "micro_batch", "grad_accum", "world_size", "expected_global_batch",
           "eval_batch", "min_lr", "lr_decay_iters", "judge_metric", "role",
           "config", "out_dir"]


def three_sig(v: float) -> float:
    if v == 0:
        return 0.0
    return round(v, -int(math.floor(math.log10(abs(v)))) + 2)


def lr_slug(v: float) -> str:
    mant, _, exp = f"{v:.2e}".partition("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant.replace('.', 'p')}e{abs(int(exp))}"


def steps_for(batch: int) -> int:
    if BUDGET_ROWS % batch:
        raise SystemExit(f"budget {BUDGET_ROWS:,} not a multiple of batch {batch}")
    return BUDGET_ROWS // batch


def interval(every: int, batch: int) -> int:
    if every % batch:
        raise SystemExit(f"interval {every} not a multiple of batch {batch}")
    return every // batch


def accum(arch: str, micro: int, batch: int):
    world = WORLD_SIZE[arch]
    divisor = micro * world
    if batch % divisor:
        raise SystemExit(f"{arch} batch {batch} not a multiple of micro {micro} x world {world}")
    return batch // divisor, world


def row_for(arch, size, batch, lr, multiplier, micro, role):
    steps = steps_for(batch)
    ga, world = accum(arch, micro, batch)
    slug = lr_slug(lr)
    stem = f"{arch}_{size}_b{batch}_lr{slug}"
    return {
        "arch": arch, "size": size, "batch": batch, "lr": f"{lr:.2e}",
        "lr_multiplier": f"{multiplier:.1f}" if multiplier is not None else "",
        "lr_slug": slug, "steps": steps,
        "warmup_steps": int(WARMUP_FRACTION[arch] * steps),
        "eval_interval": interval(EVAL_EVERY, batch),
        "save_interval": interval(SAVE_EVERY, batch),
        "rows": BUDGET_ROWS, "tokens": BUDGET_ROWS * SEQUENCE_LENGTH,
        "micro_batch": micro, "grad_accum": ga, "world_size": world,
        "expected_global_batch": batch,
        "eval_batch": BERT_EVAL_BATCH if arch == "bert" else "",
        "min_lr": f"{three_sig(lr / GPT2_MIN_LR_DIVISOR):.2e}" if arch == "gpt2" else "",
        "lr_decay_iters": steps if arch == "gpt2" else "",
        "judge_metric": JUDGE_METRIC[arch], "role": role,
        "config": f"molcrawl/tasks/pretrain/configs/protein_sequence/{stem}.py",
        "out_dir": f"protein_sequence-{arch}-{size}-b{batch}-lr{slug}",
    }


def build(micro_set: str):
    mb = MICRO_BATCH[micro_set]
    rows = []
    # BERT grid (small/medium/large)
    for size in BERT_SIZES:
        micro = mb["bert"][size]
        for batch in BATCHES:
            for m in BERT_MULTIPLIERS[batch]:
                lr = three_sig(BASE_LR_2560["bert"][size] * math.sqrt(batch / BASE_BATCH) * m)
                rows.append(row_for("bert", size, batch, lr, m, micro, "grid"))
    # GPT-2 grid (four sizes)
    for size in GPT2_SIZES:
        micro = mb["gpt2"][size]
        if micro is None:
            continue  # measured micro not yet available
        for batch in BATCHES:
            lr = three_sig(BASE_LR_2560["gpt2"][size] * math.sqrt(batch / BASE_BATCH))
            rows.append(row_for("gpt2", size, batch, lr, 1.0, micro, "grid"))
    # BERT xl diagnostic (section 1.1): batch 256, explicit rates
    for lr in XL_DIAG_RATES:
        rows.append(row_for("bert", "xl", XL_DIAG_BATCH, lr, None, mb["bert"]["xl"], "xl-diagnostic"))
    return rows


def check(rows) -> int:
    problems = 0
    n_bert_grid = sum(1 for r in rows if r["arch"] == "bert" and r["role"] == "grid")
    n_gpt2 = sum(1 for r in rows if r["arch"] == "gpt2")
    n_xl = sum(1 for r in rows if r["role"] == "xl-diagnostic")
    for label, got, want in (("bert_grid", n_bert_grid, EXPECTED_COUNTS["bert_grid"]),
                             ("gpt2", n_gpt2, EXPECTED_COUNTS["gpt2"]),
                             ("bert_xl_diag", n_xl, EXPECTED_COUNTS["bert_xl_diag"])):
        ok = got == want or (label == "gpt2" and got == 0)  # gpt2 may be deferred
        print(f"  {'ok' if ok else '**'} count {label}: {got} (directive {want})")
        if not ok:
            problems += 1
    for batch in BATCHES:
        for label, got, want in (("steps", steps_for(batch), EXPECTED_STEPS[batch]),
                                 ("eval", interval(EVAL_EVERY, batch), EXPECTED_EVAL[batch]),
                                 ("save", interval(SAVE_EVERY, batch), EXPECTED_SAVE[batch])):
            if got != want:
                problems += 1
                print(f"  ** batch {batch} {label}: {got} against directive {want}")
        for arch in ("bert", "gpt2"):
            gw = int(WARMUP_FRACTION[arch] * steps_for(batch))
            if gw != EXPECTED_WARMUP[arch][batch]:
                problems += 1
                print(f"  ** batch {batch} {arch} warmup: {gw} against {EXPECTED_WARMUP[arch][batch]}")
    for size in BERT_SIZES:
        for batch in BATCHES:
            sel = sorted(float(r["lr"]) for r in rows
                         if r["arch"] == "bert" and r["size"] == size
                         and r["batch"] == batch and r["role"] == "grid")
            want = sorted(EXPECTED_BERT_LR[size][batch])
            if [three_sig(v) for v in sel] != [three_sig(v) for v in want]:
                problems += 1
                print(f"  ** bert {size} batch {batch} lr: {sel} against directive {want}")
    print(f"  compared: {sum(len(EXPECTED_BERT_LR[s][b]) for s in BERT_SIZES for b in BATCHES)} "
          f"BERT rates, {len(BATCHES)} step counts, {2*len(BATCHES)} warmups, "
          f"{2*len(BATCHES)} intervals")
    return problems


def write_tsv(path, rows, micro_set):
    now = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
    with open(path, "w", newline="") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size,batch,lr) -- protein batch-size grid "
                 "(2026-11-20 directive), one row per arm\n")
        fh.write("# source: derived from the directive's rules (budget 85,708,800 columns both "
                 "frameworks; eval/save anchoring; sqrt scaling fanned down; warmup fraction) "
                 "and compared against its printed tables\n")
        fh.write(f"# generated: {now} by scripts/protein_batch_sweep_arms.py (micro set '{micro_set}')\n")
        fh.write("# note: BERT micro-batch 32 all sizes, MEASURED this session (jobs 374634-374637); "
                 "GPT-2 micro from its own probe ('measured'), else the adopted -rerun micro ('adopted', provisional)\n")
        fh.write("# note: BERT effective = micro x grad_accum x 4 GPUs; GPT-2 effective = micro x grad_accum "
                 "(nanoGPT divides declared accum by world size). expected_global_batch = batch\n")
        fh.write("# note: role=grid is the 56-arm grid; role=xl-diagnostic is the section-1.1 BERT xl probe "
                 "(batch 256, explicit rates 3.79e-6/7.59e-6/1.20e-5)\n")
        fh.write("# note: BERT judged on eval_loss_mask (section 7), GPT-2 on val_loss. eval micro-batch 32 "
                 "on every BERT arm (section 6)\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt2-micro", choices=["measured", "adopted"], default="adopted",
                    help="'measured' uses the GPT-2 probe's micros (None until filled); "
                         "'adopted' uses the -rerun arms' micros as a provisional")
    a = ap.parse_args(argv)
    rows = build(a.gpt2_micro)
    nb = sum(1 for r in rows if r["arch"] == "bert")
    ng = sum(1 for r in rows if r["arch"] == "gpt2")
    print(f"=== derived {len(rows)} arms (BERT {nb}, GPT-2 {ng}), gpt2-micro '{a.gpt2_micro}' ===")
    print("\n=== against the directive's printed tables ===")
    problems = check(rows)
    write_tsv(a.out, rows, a.gpt2_micro)
    print(f"\n{len(rows)} arms -> {a.out}")
    print(f"disagreements with the directive: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
