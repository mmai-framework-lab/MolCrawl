"""protein BERT (seed 42) loss curves + arm inventory, per the 2026-11-20 directive
(protein-curve-tsv-order-2026-11-20.md).

Two report-tsv v1 files, mirroring the rna rt1-rna-curves shape with the directive's
added `blocks` and `tokens` columns:

  curves TSV (one row per point):
    arch size lr seed step blocks tokens metric value
    metric in {train_loss, eval_loss, eval_loss_mask, eval_loss_copy, eval_loss_random}
      - train_loss        = HF trainer training loss (log_history "loss")
      - eval_loss         = full validation loss
      - eval_loss_mask    = loss on masked positions (the judged metric)
      - eval_loss_copy    = loss on kept-in-place positions  (据え置き)
      - eval_loss_random  = loss on replaced-with-random positions (置き換え)

  arms TSV (one row per arm): size lr adopted batch_size gradient_accumulation_steps
    world_size expected_global_batch max_steps warmup_steps eval_steps judge_on
    fixed_eval_mask mlm_probability reached_step min_value min_step collapsed

Every value is read from the run's run_manifest.json (NOT the config) and the newest
checkpoint's trainer_state.json (HF restores the full log_history on resume, so it
spans all segments). No model, no GPU -- small JSON reads only.

Arm set = the seed-42 LR grid, one arm per (size, lr), 4 sizes, all learning rates,
collapsed arms kept. The adopted large arm is the -rerun (what the official ProteinGym
eval scored); the superseded original large-lr3p5e5 and the seed-variance (-s1/-s17)
and diagnostic (-dgtest/-dgtrace) runs are out of the grid and listed in the report's
provenance section, not here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os

# (size, lr_tag, dirname, adopted) -- explicit so the inclusion rule is auditable.
# adopted per size from workflows/submit-protein-proteingym-official-fullassay.sh:
#   small -> small-lr3e4/ckpt-32000 ; medium -> medium-lr1e4/ckpt-32900 ;
#   large -> large-lr3p5e5-rerun/ckpt-32000 ; xl -> none (did not learn; downstream dropped).
ARMS = [
    ("small", "lr1e3",   "protein_sequence-small-lr1e3",   False),
    ("small", "lr1e4",   "protein_sequence-small-lr1e4",   False),
    ("small", "lr3e4",   "protein_sequence-small-lr3e4",   True),
    ("small", "lr4p1e4", "protein_sequence-small-lr4p1e4", False),
    ("small", "lr5p5e4", "protein_sequence-small-lr5p5e4", False),
    ("small", "lr7p4e4", "protein_sequence-small-lr7p4e4", False),
    ("medium", "lr1e3",   "protein_sequence-medium-lr1e3",   False),
    ("medium", "lr1e4",   "protein_sequence-medium-lr1e4",   True),
    ("medium", "lr1p3e4", "protein_sequence-medium-lr1p3e4", False),
    ("medium", "lr1p7e4", "protein_sequence-medium-lr1p7e4", False),
    ("medium", "lr2p3e4", "protein_sequence-medium-lr2p3e4", False),
    ("medium", "lr3e4",   "protein_sequence-medium-lr3e4",   False),
    ("large", "lr1e3",   "protein_sequence-large-lr1e3",         False),
    ("large", "lr1e4",   "protein_sequence-large-lr1e4",         False),
    ("large", "lr1p2e5", "protein_sequence-large-lr1p2e5",       False),
    ("large", "lr2p5e5", "protein_sequence-large-lr2p5e5",       False),
    ("large", "lr3e4",   "protein_sequence-large-lr3e4",         False),
    ("large", "lr3p5e5", "protein_sequence-large-lr3p5e5-rerun", True),   # adopted = the rerun
    ("large", "lr5e5",   "protein_sequence-large-lr5e5",         False),
    ("large", "lr7e5",   "protein_sequence-large-lr7e5",         False),
    ("xl", "lr1p2e5", "protein_sequence-xl-lr1p2e5", False),
    ("xl", "lr1p7e5", "protein_sequence-xl-lr1p7e5", False),
    ("xl", "lr2p4e5", "protein_sequence-xl-lr2p4e5", False),
    ("xl", "lr3p4e5", "protein_sequence-xl-lr3p4e5", False),
    ("xl", "lr5p5e6", "protein_sequence-xl-lr5p5e6", False),
    ("xl", "lr8e6",   "protein_sequence-xl-lr8e6",   False),
]

METRICS = ["train_loss", "eval_loss", "eval_loss_mask", "eval_loss_copy", "eval_loss_random"]
# key in trainer_state log_history for each reported metric
LOGKEY = {
    "train_loss": "loss",
    "eval_loss": "eval_loss",
    "eval_loss_mask": "eval_loss_mask",
    "eval_loss_copy": "eval_loss_copy",
    "eval_loss_random": "eval_loss_random",
}
COLLAPSE_MARGIN = 0.10  # collapsed := final eval_loss_mask exceeds the arm's own best by > this


def newest_trainer_state(run_dir):
    cks = [int(p.split("-")[-1]) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))]
    if not cks:
        return None
    p = os.path.join(run_dir, f"checkpoint-{max(cks)}", "trainer_state.json")
    return json.load(open(p)) if os.path.exists(p) else None


def collect_points(ts):
    """metric -> sorted list of (step, value), de-duplicated by step (last wins)."""
    out = {}
    for rep, key in LOGKEY.items():
        d = {}
        for e in ts.get("log_history", []):
            if key in e and "step" in e:
                # a train "loss" entry also has learning_rate/grad_norm; an eval entry does not.
                if rep == "train_loss" and "eval_loss" in e:
                    continue
                d[int(e["step"])] = float(e[key])
        out[rep] = sorted(d.items())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bert-root", required=True, help="…/protein_sequence/bert-output")
    ap.add_argument("--curves-out", required=True)
    ap.add_argument("--arms-out", required=True)
    a = ap.parse_args()

    now = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
    gen = f"# generated: {now} by scripts/protein_bert_seed42_curves.py"

    curve_rows = []
    arm_rows = []
    summary = []
    for size, _lrtag, dirname, adopted in ARMS:
        run_dir = os.path.join(a.bert_root, dirname)
        man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
        ts = newest_trainer_state(run_dir)
        if ts is None:
            raise SystemExit(f"no trainer_state for {dirname}")

        seed = man["seed"]["seed"]
        lr = man["schedule"]["learning_rate"]
        gb = man["batch"]["effective_global_batch"]
        tps = man["batch"]["tokens_per_step"]
        lrs = f"{lr:.2e}"

        pts = collect_points(ts)
        for metric in METRICS:
            for step, val in pts[metric]:
                curve_rows.append([
                    "bert", size, lrs, seed, step,
                    step * gb, step * tps, metric, f"{val:.6f}",
                ])

        emask = dict(pts["eval_loss_mask"])
        reached = ts.get("global_step")
        min_step = min(emask, key=emask.get) if emask else None
        min_val = emask[min_step] if emask else None
        final_step = max(emask) if emask else None
        final_val = emask[final_step] if emask else None
        collapsed = bool(emask) and (final_val - min_val) > COLLAPSE_MARGIN

        arm_rows.append([
            size, lrs, "yes" if adopted else "no",
            man["batch"]["per_device_train_batch_size"],
            man["batch"]["gradient_accumulation_steps"],
            man["batch"]["world_size"],
            man["batch"]["expected_global_batch"],
            man["schedule"]["max_steps"],
            man["schedule"]["warmup_steps"],
            man["selection"]["eval_steps"],
            man["selection"]["metric_for_best_model"],
            man.get("eval", {}).get("fixed_mask", {}).get("enabled"),
            man.get("objective", {}).get("mlm_probability"),
            reached,
            f"{min_val:.6f}" if min_val is not None else "",
            min_step if min_step is not None else "",
            "yes" if collapsed else "no",
        ])
        summary.append((size, lrs, adopted, min_val, min_step, final_val, final_step, reached, collapsed))

    # ---- curves TSV ----
    with open(a.curves_out, "w", newline="") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size,lr,seed) -- protein BERT seed-42 LR grid\n")
        fh.write("# source: each run's newest checkpoint trainer_state.json (HF restores the full log_history on resume)\n")
        fh.write(f"{gen}\n")
        fh.write("# note: one row per step and metric. blocks = step x effective_global_batch; tokens = step x tokens_per_step (both from run_manifest.json)\n")
        fh.write("# note: metric names a measured loss so a reader selects rather than mixes. train_loss is the HF training loss; the four eval_* are on a fixed 10,000-row validation subset (seed 42), mask redrawn each evaluation\n")
        fh.write("# note: eval_loss_copy = kept-in-place positions (据え置き); eval_loss_random = replaced-with-random positions (置き換え)\n")
        fh.write("# note: seed 42 on every arm. The adopted large arm is large-lr3p5e5-rerun; see the report for the superseded/variance/diagnostic runs left out\n")
        fh.write("# note: arms that stopped short of max_steps end where they stopped; see the arms TSV reached_step/collapsed columns\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["arch", "size", "lr", "seed", "step", "blocks", "tokens", "metric", "value"])
        w.writerows(curve_rows)
    print(f"wrote {len(curve_rows):,} rows -> {a.curves_out}")

    # ---- arms TSV ----
    with open(a.arms_out, "w", newline="") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: size,lr) -- protein BERT seed-42 LR grid, one row per arm\n")
        fh.write("# source: each run's run_manifest.json (values as they actually ran) + newest trainer_state.json for reached_step/min_value\n")
        fh.write(f"{gen}\n")
        fh.write("# note: adopted = the arm the official ProteinGym eval scored (small-lr3e4, medium-lr1e4, large-lr3p5e5-rerun); xl has no adopted arm (did not learn)\n")
        fh.write("# note: min_value/min_step are the lowest eval_loss_mask and its step. collapsed := final eval_loss_mask exceeds the arm's own best by > 0.10\n")
        fh.write("# note: fixed_eval_mask blank = older manifest did not record eval.fixed_mask.enabled (behaviour is mask-redrawn, subset_random false)\n")
        fh.write("# note: mlm_probability blank/null = DocumentMaskingCollator (document-level masking), no scalar rate recorded in the manifest\n")
        fh.write("# note: all learning rates come from schedule.learning_rate with sources.learning_rate.from = overridden (config file; env/argv indistinguishable per manifest)\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow([
            "size", "lr", "adopted", "batch_size", "gradient_accumulation_steps",
            "world_size", "expected_global_batch", "max_steps", "warmup_steps",
            "eval_steps", "judge_on", "fixed_eval_mask", "mlm_probability",
            "reached_step", "min_value", "min_step", "collapsed",
        ])
        w.writerows(arm_rows)
    print(f"wrote {len(arm_rows)} arms -> {a.arms_out}")

    print("\n=== per-arm eval_loss_mask: min (step) | final (step) | reached | collapsed ===")
    for size, lrs, adopted, mn, ms, fv, fs, reached, col in summary:
        tag = " *ADOPTED" if adopted else ""
        print(f"{size:6s} {lrs:>9s}: min {mn:.4f}@{ms:<6d} final {fv:.4f}@{fs:<6d} reached {reached:<6d} "
              f"{'COLLAPSED' if col else 'ok':9s}{tag}")


if __name__ == "__main__":
    main()
