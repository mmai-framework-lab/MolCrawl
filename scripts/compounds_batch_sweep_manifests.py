"""Read every arm's run_manifest.json and compare it, value by value, with the plan.

The 2026-11-20 directive's §3 asks for one thing: that each arm's settings be read out
of what the run itself recorded, listed, and compared one at a time against the
directive, with the number of disagreements stated. That is what this does. The
expectation is not typed in again here -- it comes from the derived arm inventory,
which ``compounds_batch_sweep_arms.py`` has already checked against the directives'
printed tables -- so a disagreement means the run and the plan differ, not that two
hand-copied tables differ.

    compounds_batch_sweep_manifests.py --arms rt1-...tsv --runs-root <runs root> \
        --out rt1-...tsv [--configs <configs/compounds>]

One row per arm and key. ``checked`` says whether the row carries an expectation:
a few values are read out and reported without one, because the directive asks for
them but fixes no value (the evaluation rows GPT-2 scores over, for instance).

Why one key is read from the config rather than the manifest: the BERT manifest does
not record ``per_device_eval_batch_size``. §3 asks for it, and the honest place to get
it until the manifest records it is the config the run was launched with, so that row
names the config as its source. Adding it to the manifest writer would change what
every modality records and would not reach the runs already going, so it is proposed
rather than done here.

Read-only: it opens manifests and configs and writes one TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: §9's seed, §6's mask seed and subset seed, and §6's evaluation rows: one value each
#: across the grid, so they are stated here rather than carried per arm.
SEED = 42
EVAL_ROWS = 10_000
MASK_RATE = 0.2
GPT2_EVAL_SEQUENCES = 3200
WORLD_SIZE = {"bert": 4, "gpt2": 4}

#: The size's name in the output tree. nanoGPT's compounds xl writes "ex-large".
GPT2_DIR_SIZE = {"small": "small", "medium": "medium", "large": "large",
                 "xl": "ex-large"}

COLUMNS = ["arch", "size", "batch", "lr", "arm", "key", "source", "expected",
           "observed", "checked", "agrees", "note"]


def dig(blob, path):
    """``a/b/c`` out of nested dicts; a missing step returns the marker, not None,
    so 'the manifest does not record it' is distinguishable from 'it recorded None'."""
    node = blob
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            return KeyError
        node = node[part]
    return node


def same(expected, observed) -> bool:
    if expected is None:
        return True
    if isinstance(expected, float) or isinstance(observed, float):
        try:
            return math.isclose(float(expected), float(observed), rel_tol=1e-9,
                                abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, bool) or isinstance(observed, bool):
        return bool(expected) == bool(observed)
    return str(expected) == str(observed)


def cell(value) -> str:
    """R10: a value the manifest does not carry is an empty cell, not a marker."""
    if value is KeyError or value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def read_arms(path):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: no arm rows")
    return rows


def run_dir(arm, runs_root):
    slug = f"-b{arm['batch']}-lr{arm['lr_slug']}"
    if arm["arch"] == "bert":
        return os.path.join(runs_root, "compounds", "bert-output",
                            f"compounds-{arm['size']}{slug}")
    return os.path.join(runs_root, "compounds", "gpt2-output",
                        f"compounds-{GPT2_DIR_SIZE[arm['size']]}{slug}")


def eval_batch_from_config(arm, configs_dir):
    """per_device_eval_batch_size, which the BERT manifest does not record."""
    path = os.path.join(configs_dir, os.path.basename(arm["config"]))
    if not os.path.exists(path):
        return KeyError, path
    for line in open(path, encoding="utf-8"):
        if line.startswith("per_device_eval_batch_size"):
            return int(line.split("=", 1)[1].split("#")[0].strip()), path
    return KeyError, path


def bert_checks(arm, configs_dir):
    """(key, manifest path, expected, note) for one BERT arm. None = read out only."""
    judged = arm["judge_metric"]
    eval_batch, cfg_path = eval_batch_from_config(arm, configs_dir)
    return [
        ("seed", "seed/seed", SEED, "§9"),
        ("fixed_eval_mask", "eval/fixed_mask/enabled", True, "§6"),
        ("eval_mask_seed", "eval/fixed_mask/seed", SEED, "§6"),
        ("judge_on", "selection/judge_on", judged, "2026-11-20 §2"),
        ("metric_for_best_model", "selection/metric_for_best_model", judged,
         "2026-11-20 §2"),
        ("save_on_improve_metric", "introduced/save_on_improve_metric", judged,
         "2026-11-20 §2 (the retention side)"),
        ("checkpoint_metric", "introduced/checkpoint_metric", judged,
         "2026-11-20 §2 (the retention side)"),
        ("retention_metric", "selection/retention/metric", judged,
         "2026-11-20 §2 (what the keeper ranks on)"),
        ("gpus", "placement/world_size", WORLD_SIZE["bert"], "§6"),
        ("eval_rows", "eval/rows", EVAL_ROWS, "§6"),
        ("eval_subset_random", "eval/subset_random", True, "§6"),
        ("eval_subset_seed", "eval/subset_seed", SEED,
         "§6; from EVAL_SUBSET_SEED in models/bert/main.py, not from a config"),
        ("mask_rate", "objective/mlm_probability", MASK_RATE,
         "hardcoded in models/bert/main.py"),
        ("batch_size", "batch/per_device_train_batch_size", int(arm["micro_batch"]),
         "micro-batch directive §1"),
        ("gradient_accumulation_steps", "batch/gradient_accumulation_steps",
         int(arm["grad_accum"]), "micro-batch directive §1"),
        ("expected_global_batch", "batch/expected_global_batch", int(arm["batch"]),
         "§5"),
        ("effective_global_batch", "batch/effective_global_batch", int(arm["batch"]),
         "§5; what the run actually built"),
        ("max_steps", "schedule/max_steps", int(arm["steps"]), "§2"),
        ("warmup_steps", "schedule/warmup_steps", int(arm["warmup_steps"]), "§4"),
        ("eval_steps", "selection/eval_steps", int(arm["eval_interval"]), "§2"),
        ("save_steps", "selection/save_steps", int(arm["save_interval"]), "§2"),
        ("learning_rate", "schedule/learning_rate", float(arm["lr"]), "§3"),
        ("per_device_eval_batch_size", f"config:{os.path.basename(cfg_path)}",
         int(arm["eval_batch"]), "§6; the manifest does not record it",
         eval_batch),
    ]


def gpt2_checks(arm):
    return [
        ("seed", "seed/seed", SEED, "§9"),
        ("judge_on", "eval/judge_on", None,
         "19c §7: val loss; the manifest writes it as a description"),
        ("gpus", "placement/world_size", WORLD_SIZE["gpt2"],
         "not fixed by the directive; nanoGPT's global batch is GPU-independent"),
        ("batch_size", "batch/batch_size", int(arm["micro_batch"]), "19c §8"),
        ("gradient_accumulation_steps", "batch/gradient_accumulation_steps_configured",
         int(arm["grad_accum"]), "19c §8"),
        ("effective_global_batch", "batch/effective_global_batch", int(arm["batch"]),
         "§5; the GPT-2 manifest records no declared/observed pair -- train.py "
         "refuses to start on a mismatch, so the run existing is that check"),
        ("max_iters", "schedule/max_iters", int(arm["steps"]), "§2"),
        ("warmup_iters", "schedule/warmup_iters", int(arm["warmup_steps"]), "§4"),
        ("lr_decay_iters", "schedule/lr_decay_iters", int(arm["lr_decay_iters"]),
         "§4"),
        ("min_lr", "schedule/min_lr", float(arm["min_lr"]), "§4"),
        ("learning_rate", "schedule/learning_rate", float(arm["lr"]), "§3"),
        ("eval_interval", "eval/eval_interval", int(arm["eval_interval"]), "§2"),
        ("eval_sequences", "eval/eval_sequences", GPT2_EVAL_SEQUENCES, "§6"),
        ("deterministic_val_eval", "eval/deterministic_val_eval", True, "§6"),
        ("eval_rows", "data/rows/eval", None,
         "read out only: GPT-2 scores its own val split, not §6's 10,000 rows"),
        ("mask_rate", "", None, "read out only: not applicable to a causal LM"),
    ]


def rows_for(arms, runs_root, configs_dir):
    rows, missing, problems = [], [], 0
    for arm in arms:
        where = run_dir(arm, runs_root)
        manifest = os.path.join(where, "run_manifest.json")
        name = os.path.basename(where)
        if not os.path.exists(manifest):
            missing.append((arm, manifest))
            continue
        blob = json.load(open(manifest, encoding="utf-8"))
        checks = (bert_checks(arm, configs_dir) if arm["arch"] == "bert"
                  else gpt2_checks(arm))
        for check in checks:
            key, path, expected, note = check[:4]
            observed = check[4] if len(check) > 4 else (
                dig(blob, path) if path else KeyError)
            checked = expected is not None
            agrees = same(expected, observed) if checked else None
            if checked and not agrees:
                problems += 1
            rows.append({
                "arch": arm["arch"], "size": arm["size"], "batch": int(arm["batch"]),
                "lr": arm["lr"], "arm": name, "key": key,
                "source": path or "n/a",
                "expected": cell(expected), "observed": cell(observed),
                "checked": "yes" if checked else "no",
                "agrees": "" if agrees is None else ("yes" if agrees else "NO"),
                "note": note,
            })
    return rows, missing, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--configs", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "molcrawl", "tasks", "pretrain", "configs", "compounds"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    check_name(args.out)

    arms = read_arms(args.arms)
    rows, missing, problems = rows_for(arms, args.runs_root, args.configs)
    read = len({r["arm"] for r in rows})
    print(f"=== {len(arms)} arms; {read} with a manifest, {len(missing)} without ===")
    for arm, path in missing:
        print(f"  -- no manifest yet for {arm['arch']} {arm['size']} "
              f"batch {arm['batch']} lr {arm['lr']}  ({path})")

    for arch in ("bert", "gpt2"):
        sel = [r for r in rows if r["arch"] == arch]
        if not sel:
            continue
        done = [r for r in sel if r["checked"] == "yes"]
        bad = [r for r in done if r["agrees"] == "NO"]
        print(f"\n  {arch}: {len({r['arm'] for r in sel})} arms, {len(done)} values "
              f"compared, {len(bad)} disagreement(s)")
        for r in bad:
            print(f"    ** {r['arm']}  {r['key']}: manifest {r['observed']!r} "
                  f"against the plan {r['expected']!r}")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "several (columns: arch,size,batch,lr) -- every arm of the compounds "
                 "batch-size grid that has a manifest",
        "source": "each arm's own run_manifest.json, except the "
                  "per_device_eval_batch_size row, whose source column names the "
                  "config it was read from (the BERT manifest does not record it)",
        "generated": stamp("scripts/compounds_batch_sweep_manifests.py"),
        "note": [
            "one row per arm and key (2026-11-20 §3); expected comes from the derived "
            "arm inventory, which was itself checked against the directives' tables",
            "checked=no is a value the directive asks to see but fixes no value for",
            f"{sum(1 for r in rows if r['checked'] == 'yes')} values compared, "
            f"{problems} disagreement(s)",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"disagreements: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
