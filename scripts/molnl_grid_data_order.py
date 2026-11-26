#!/usr/bin/env python3
"""The three tables the 2026-11-26 data order asks for, off the molnl batch-size grid.

Item 2  one row per arm: identity, the adopted checkpoint and how many 1,024-token
        blocks it had read, the judged metric's minimum, its value at the budget, and
        the collapse flag exactly as the current rule sets it (the order says the rule
        is not to be changed).
Item 3  one row per arm and evaluated step: blocks and tokens read, the training loss,
        eval_loss, and the judged metric (eval_loss_mask on BERT, val_loss on GPT-2).
        Every evaluation point; nothing is thinned.
Item 4  one row per arm: the values read back from each arm's run_manifest.json.

Items 2 and 3 are read off the runs themselves, through the same readers the 2026-11-25
tables used (scripts/molnl_batch_sweep_curves.py), and the per-arm numbers are then
checked against that day's summary: a disagreement stops the script rather than writing
a table that differs from the one already sent.

The adopted checkpoint is read, not inferred from the minimum:
  BERT   best_model_checkpoint in the last checkpoint's trainer_state.json, and the
         directory it names must exist.
  GPT-2  nanoGPT writes a checkpoint when val improves and records best_val_loss in
         every checkpoint's training_args.json. The adopted one is the checkpoint whose
         own val_loss equals the last checkpoint's best_val_loss.
Either way the script reports whether that step is the minimum of the judged series.

Item 4 is the 2026-11-25 manifest table reshaped: that table was read from every arm's
run_manifest.json after the last arm finished and compared value by value against the
plan (1,424 values, 0 disagreements). A manifest is written when a run starts, so
nothing has changed since.

    python scripts/molnl_grid_data_order.py --arms <arm inventory tsv> \\
        --root <MODEL_OUTPUT_ROOT> --summary-1125 <tsv> --manifests-1125 <tsv> \\
        --out-dir <report dir> --stamp 20261126
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molnl_batch_sweep_curves import (  # noqa: E402
    SEQUENCE_LENGTH,
    SEED,
    bert_series,
    collapse_at,
    gpt2_series,
    judged,
    read_arms,
    run_dir,
)
from report_tsv import stamp, write_tsv  # noqa: E402

SCRIPT = "scripts/molnl_grid_data_order.py"
ARCH_LABEL = {"bert": "BERT", "gpt2": "GPT-2"}

ARM_COLUMNS = ["arm", "arch", "size", "batch", "lr", "lr_multiplier", "seed",
               "adopted_step", "adopted_rows", "judged_metric", "best_value",
               "budget_value", "diverged"]

CURVE_COLUMNS = ["arm", "step", "rows_seen", "tokens_seen", "train_loss", "eval_loss",
                 "eval_loss_mask_or_val_loss"]

# (output column, column of the 2026-11-25 manifest table). Order follows item 4.
MANIFEST_COLUMNS = [
    ("arm", "arm"), ("arch", "arch"),
    ("seed", "seed"),
    ("fixed_eval_mask", "fixed_eval_mask"), ("eval_mask_seed", "eval_mask_seed"),
    ("judge_on", "judge_on"), ("metric_for_best_model", "metric_for_best_model"),
    ("retention_metric", "retention_metric"),
    ("per_device_eval_batch_size", "per_device_eval_batch_size"), ("gpus", "gpus"),
    ("eval_rows", "eval_rows"), ("eval_subset_random", "eval_subset_random"),
    ("eval_subset_seed", "eval_subset_seed"),
    ("eval_sequences", "eval_sequences"),
    ("deterministic_val_eval", "deterministic_val_eval"),
    ("mask_rate", "mask_rate"),
    ("batch_size", "batch_size"),
    ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ("expected_global_batch", "expected_global_batch"),
    ("effective_global_batch", "effective_global_batch"),
    ("max_steps_or_max_iters", "max_steps"), ("warmup", "warmup_steps"),
    ("eval_interval", "eval_interval"), ("save_interval", "save_interval"),
    ("lr_decay_iters", "lr_decay_iters"), ("min_lr", "min_lr"),
    ("learning_rate", "learning_rate"),
    ("value_sources", "value_sources"), ("manifest_path", "manifest_path"),
]


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def key_of(row):
    return (row["arch"], row["size"], str(int(row["batch"])), row["lr"])


def step_of(path):
    return int(os.path.basename(path.rstrip("/")).rsplit("-", 1)[1])


def bert_adopted(path):
    cks = sorted(glob.glob(os.path.join(path, "checkpoint-*")), key=step_of)
    with open(os.path.join(cks[-1], "trainer_state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    best = state.get("best_model_checkpoint")
    if not best:
        raise SystemExit(f"{cks[-1]}: trainer_state.json names no best_model_checkpoint")
    local = os.path.join(path, os.path.basename(best.rstrip("/")))
    if not os.path.isdir(local):
        raise SystemExit(f"{path}: best_model_checkpoint {best} is not on disk")
    return step_of(local), float(state["best_metric"]), os.path.basename(local)


def gpt2_adopted(path):
    cks = sorted(glob.glob(os.path.join(path, "checkpoint-*")), key=step_of)
    args = {}
    for one in cks:
        with open(os.path.join(one, "training_args.json"), encoding="utf-8") as handle:
            args[step_of(one)] = json.load(handle)
    target = args[max(args)]["best_val_loss"]
    hits = [s for s, a in args.items() if a.get("val_loss") == target]
    if len(hits) != 1:
        raise SystemExit(f"{path}: {len(hits)} checkpoints hold val_loss == best_val_loss "
                         f"{target}; expected exactly one")
    return hits[0], float(target), f"checkpoint-{hits[0]}"


def same(a, b, places=6):
    return abs(float(a) - float(b)) <= 0.5 * 10 ** -places + 1e-12


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--root", required=True,
                    help="the PARENT of molecule_nat_lang/, as the 2026-11-25 job used")
    ap.add_argument("--summary-1125", required=True)
    ap.add_argument("--manifests-1125", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stamp", required=True)
    args = ap.parse_args(argv)

    arms = read_arms(args.arms, "both")
    sent = {key_of(r): r for r in read_tsv(args.summary_1125)}
    manifests = {key_of(r): r for r in read_tsv(args.manifests_1125)}
    if len(arms) != 68 or len(sent) != 68 or len(manifests) != 68:
        raise SystemExit(f"expected 68 arms everywhere; inventory {len(arms)}, "
                         f"2026-11-25 summary {len(sent)}, manifests {len(manifests)}")

    arm_rows, curve_rows, manifest_rows, checks = [], [], [], []
    for arm in arms:
        key = key_of(arm)
        path = run_dir(args.root, arm)
        name = arm["out_dir"]
        points, source = (bert_series if arm["arch"] == "bert" else gpt2_series)(path)
        metric = arm["judge_metric"]
        series = judged(points, metric)
        batch = int(arm["batch"])
        planned = int(arm["steps"])

        best_value, best_step = min((v, s) for s, v in series)
        final_step, final_value = series[-1]
        if final_step != planned:
            raise SystemExit(f"{name}: last evaluated step {final_step}, planned {planned}")
        hit = collapse_at(series, None)
        adopted_step, adopted_metric, adopted_dir = (
            bert_adopted(path) if arm["arch"] == "bert" else gpt2_adopted(path))

        old = sent[key]
        mismatch = [field for field, mine in (
            ("best_value", f"{best_value:.6f}"), ("final_value", f"{final_value:.6f}"))
            if not same(mine, old[field])]
        if int(old["best_step"]) != best_step:
            mismatch.append("best_step")
        if old["collapsed"] != ("yes" if hit is not None else "no"):
            mismatch.append("collapsed")
        if mismatch:
            raise SystemExit(f"{name}: differs from the 2026-11-25 summary in {mismatch}")

        checks.append((name, adopted_step, best_step, adopted_metric, best_value,
                       adopted_dir))
        arm_rows.append({
            "arm": name, "arch": ARCH_LABEL[arm["arch"]], "size": arm["size"],
            "batch": batch, "lr": arm["lr"], "lr_multiplier": arm["lr_multiplier"],
            "seed": SEED, "adopted_step": adopted_step,
            "adopted_rows": adopted_step * batch, "judged_metric": metric,
            "best_value": f"{best_value:.6f}", "budget_value": f"{final_value:.6f}",
            "diverged": "yes" if hit is not None else "no",
        })

        mine = manifests[key]
        if mine["arm"] != name:
            raise SystemExit(f"manifest row for {key} is {mine['arm']}, inventory {name}")
        if str(mine["seed"]) != str(SEED):
            raise SystemExit(f"{name}: manifest seed {mine['seed']}")
        manifest_rows.append({out: (mine[src] if mine[src] != "" else None)
                              for out, src in MANIFEST_COLUMNS})
        manifest_rows[-1]["arch"] = ARCH_LABEL[arm["arch"]]

        judged_name = "eval_loss_mask" if arm["arch"] == "bert" else "val_loss"
        train_name = "train_loss_mask" if arm["arch"] == "bert" else "train_loss"
        for step, values in sorted(points.items()):
            if judged_name not in values:
                continue
            curve_rows.append({
                "arm": name, "step": step, "rows_seen": step * batch,
                "tokens_seen": step * batch * SEQUENCE_LENGTH,
                "train_loss": (f"{values[train_name]:.6f}"
                               if train_name in values else None),
                "eval_loss": (f"{values['eval_loss']:.6f}"
                              if "eval_loss" in values else None),
                "eval_loss_mask_or_val_loss": f"{values[judged_name]:.6f}",
            })
        for value in [v for s, v in series]:
            if not math.isfinite(value):
                print(f"NOTE {name}: non-finite judged value")

    off = [c for c in checks if c[1] != c[2]]
    bert_off = [c for c in checks if c[0].find("-bert-") >= 0
                and not same(f"{c[3]:.6f}", f"{c[4]:.6f}")]
    print(f"adopted checkpoint != step of the judged minimum: {len(off)} of {len(checks)}")
    for c in off:
        print(f"  {c[0]}: adopted {c[1]} ({c[5]}), minimum at {c[2]}, "
              f"adopted metric {c[3]:.6f}, minimum {c[4]:.6f}")
    print(f"BERT best_metric differing from the series minimum: {len(bert_off)}")

    now = stamp(SCRIPT)
    model = "several -- the 68 arms of the molnl batch-size grid (2026-11-20 directive)"
    root = os.path.join(args.root, "molecule_nat_lang")
    n_arm = write_tsv(
        os.path.join(args.out_dir, f"rt1-molnl-grid-arms-{args.stamp}.tsv"),
        ARM_COLUMNS, arm_rows,
        {"model": model,
         "source": f"each arm's run under {root}: the last checkpoint's trainer_state.json "
                   "(BERT) or logging_*.csv and checkpoint-*/training_args.json (GPT-2)",
         "generated": now,
         "note": [
             "item 2 of the 2026-11-26 data order; one row per arm, all 68, the arms "
             "that diverged included",
             "adopted_step: BERT = best_model_checkpoint in trainer_state.json; GPT-2 = "
             "the checkpoint whose own val_loss equals best_val_loss (nanoGPT saves on "
             "improvement); adopted_rows = adopted_step x batch, in 1,024-token blocks",
             "best_value = lowest value of judged_metric over the arm's evaluation "
             "points; budget_value = its value at the planned last step, which every "
             "arm reached",
             "diverged is the current rule unchanged: yes on a non-finite value, or on a "
             "value after the minimum at or above the arm's own first evaluation point; "
             "no floor is applied",
             f"cross-checked against {os.path.basename(args.summary_1125)}: best_value, "
             "best_step, final value and the collapse flag agree on all 68 arms"]})
    n_curve = write_tsv(
        os.path.join(args.out_dir, f"rt1-molnl-grid-curves-{args.stamp}.tsv"),
        CURVE_COLUMNS, curve_rows,
        {"model": model,
         "source": "the same reads as the arm table, every evaluated step",
         "generated": now,
         "note": [
             "item 3 of the 2026-11-26 data order; every evaluation point, none thinned",
             "rows_seen = step x batch (1,024-token blocks); tokens_seen = rows_seen x 1024",
             "BERT train_loss is HF Trainer's logged 'loss': cross-entropy over all "
             "positions chosen for prediction (20 % of tokens; 80 % [MASK], 10 % random, "
             "10 % unchanged), averaged over the 480 optimizer steps since the previous "
             "log. The 2026-11-25 curve file called this train_loss_mask; it is not "
             "[MASK]-only",
             "BERT eval_loss is the same mixture on the valid split; the last column is "
             "eval_loss_mask on BERT ([MASK] positions only, fixed mask) and val_loss "
             "on GPT-2",
             "GPT-2 train_loss is nanoGPT's estimate_loss on the train split at the "
             "evaluation step (50 batches of 64 randomly drawn sequences, eval mode), "
             "not the optimizer-step loss; GPT-2 has no eval_loss, so that cell is empty"]})
    n_man = write_tsv(
        os.path.join(args.out_dir, f"rt1-molnl-grid-manifests-{args.stamp}.tsv"),
        [c for c, _ in MANIFEST_COLUMNS], manifest_rows,
        {"model": model,
         "source": f"{os.path.basename(args.manifests_1125)}, itself read from each "
                   "arm's run_manifest.json; reshaped to item 4's columns",
         "generated": now,
         "note": [
             "item 4 of the 2026-11-26 data order; one row per arm",
             "an empty cell is a value the manifest does not record (value_sources "
             "names the one value read elsewhere: per_device_eval_batch_size on BERT, "
             "from the config)",
             "GPT-2 records its evaluation set as eval_sequences and "
             "deterministic_val_eval instead of the BERT fields",
             "max_steps_or_max_iters is max_steps on BERT and max_iters on GPT-2; "
             "warmup is warmup_steps / warmup_iters"]})
    print(f"wrote {n_arm} arm rows, {n_curve} curve rows, {n_man} manifest rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
