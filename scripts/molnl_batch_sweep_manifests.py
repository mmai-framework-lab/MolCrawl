"""Read every arm's run_manifest.json back and compare it with the plan, value by value.

§11 of the 2026-11-20 directive asks for one row per arm, carrying the settings the run
resolved to and the absolute path of the file each was read from, and for the values to
be compared against the directive one at a time with the number of disagreements
stated. That is what this writes.

The expectation is not typed in again here: it comes from the derived arm inventory,
which ``molnl_batch_sweep_arms.py`` has already checked against the directive's printed
tables. So a disagreement means the run and the plan differ, not that two hand-copied
tables differ.

    molnl_batch_sweep_manifests.py --arms rt1-...tsv --runs-root <output root> \
        --out rt1-...tsv [--configs <configs/molecule_nat_lang>] \
        [--submit-record <file>]

§11 says to read the values from the manifest and not from the config. Two cannot be:
``per_device_eval_batch_size``, which the BERT manifest does not record, and the mask
rate, which is hardcoded in models/bert/main.py -- the manifest does record that one as
``objective/mlm_probability``, so only the evaluation micro-batch falls back to the
config. Its own column names where it came from, and the fallback is written into the
provenance rather than left for the reader to discover. Adding it to the manifest
writer would change what five modalities record and would not reach runs already going,
so it is proposed rather than done here.

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

#: One value each across the grid, so they are stated here rather than carried per arm.
SEED = 42
MASK_RATE = 0.2
GPT2_EVAL_SEQUENCES = 3200
#: §6. molnl scores its whole validation split; main.py's 10,000-row cap never bites.
EVAL_ROWS = 1631
EVAL_SUBSET_RANDOM = False
GPUS = 4

#: (column, manifest path, inventory column or literal, note). A literal that is None
#: means the value is read out and reported without an expectation, because §11 asks to
#: see it but fixes no value.
BERT_KEYS = [
    ("seed", "seed/seed", SEED, "§8"),
    ("fixed_eval_mask", "eval/fixed_mask/enabled", True, "§6"),
    ("eval_mask_seed", "eval/fixed_mask/seed", SEED, "§6"),
    ("judge_on", "selection/judge_on", "@judge_metric", "§7"),
    ("metric_for_best_model", "selection/metric_for_best_model", "@judge_metric", "§7"),
    ("retention_metric", "selection/retention/metric", "@judge_metric",
     "§7: the metric the keeper ranks on"),
    ("gpus", "placement/world_size", GPUS, "§6"),
    ("eval_rows", "data/rows/eval", EVAL_ROWS, "§6: the whole validation split"),
    ("eval_subset_random", "eval/subset_random", EVAL_SUBSET_RANDOM,
     "§6: no subset is drawn; the split is shorter than the 10,000-row cap"),
    ("eval_subset_seed", "eval/subset_seed", SEED,
     "§6: EVAL_SUBSET_SEED in models/bert/main.py, not from a config"),
    ("mask_rate", "objective/mlm_probability", MASK_RATE,
     "§8: hardcoded in models/bert/main.py"),
    ("seq_len", "batch/seq_len", 1024, "§8"),
    ("batch_size", "batch/per_device_train_batch_size", "@micro_batch", "§4"),
    ("gradient_accumulation_steps", "batch/gradient_accumulation_steps", "@grad_accum",
     "§4"),
    ("expected_global_batch", "batch/expected_global_batch", "@batch", "§5"),
    ("effective_global_batch", "batch/effective_global_batch", "@batch",
     "§5: what the run actually built"),
    ("max_steps", "schedule/max_steps", "@steps", "§2"),
    ("warmup_steps", "schedule/warmup_steps", "@warmup_steps", "§2"),
    ("eval_interval", "selection/eval_steps", "@eval_interval", "§2"),
    ("save_interval", "selection/save_steps", "@save_interval", "§2"),
    ("learning_rate", "schedule/learning_rate", "@lr", "§3"),
    ("train_rows", "data/rows/train", 318_118, "§2: the block count the budget is of"),
]
GPT2_KEYS = [
    ("seed", "seed/seed", SEED, "§8"),
    ("judge_on", "eval/judge_on", None,
     "§7: val_loss; the manifest writes it as a description"),
    ("gpus", "placement/world_size", GPUS,
     "not fixed by the directive; nanoGPT's global batch is GPU-count-independent"),
    ("eval_sequences", "eval/eval_sequences", GPT2_EVAL_SEQUENCES, "§6"),
    ("deterministic_val_eval", "eval/deterministic_val_eval", True,
     "§6 as answered: molnl's GPT-2 default resamples, and the 2026-10-14 order §2 "
     "fixed it"),
    ("eval_rows", "data/rows/eval", EVAL_ROWS, "read out: GPT-2 scores its own split"),
    ("seq_len", "batch/seq_len", 1024, "§8"),
    ("batch_size", "batch/batch_size", "@micro_batch", "§4"),
    ("gradient_accumulation_steps", "batch/gradient_accumulation_steps_configured",
     "@grad_accum", "§4"),
    ("effective_global_batch", "batch/effective_global_batch", "@batch",
     "§5: train.py refuses to start on a mismatch, so the run existing is that check"),
    ("max_steps", "schedule/max_iters", "@steps", "§2"),
    ("warmup_steps", "schedule/warmup_iters", "@warmup_steps", "§2"),
    ("eval_interval", "eval/eval_interval", "@eval_interval", "§2"),
    ("save_interval", "", None, "read out: nanoGPT records no save interval"),
    ("lr_decay_iters", "schedule/lr_decay_iters", "@lr_decay_iters", "§8"),
    ("min_lr", "schedule/min_lr", "@min_lr", "§8"),
    ("learning_rate", "schedule/learning_rate", "@lr", "§3"),
    ("train_rows", "data/rows/train", 318_118, "§2: the block count the budget is of"),
]

#: §11's location columns, all absolute.
PLACE_COLUMNS = ["config_path", "manifest_path", "out_dir", "job_id", "submit_record",
                 "commit"]
VALUE_COLUMNS = [k[0] for k in BERT_KEYS]
#: GPT-2 adds two the BERT rows leave empty.
VALUE_COLUMNS += [k[0] for k in GPT2_KEYS if k[0] not in VALUE_COLUMNS]
#: The one value with no manifest path of its own, so it is in neither key table: the
#: BERT manifest does not record the evaluation micro-batch and it is read from the
#: config instead. It still needs a column, or the row carries a field the header does
#: not and nothing is written.
VALUE_COLUMNS += ["per_device_eval_batch_size"]
COLUMNS = (["arch", "size", "batch", "lr", "arm"] + VALUE_COLUMNS
           + ["disagreements", "value_sources"] + PLACE_COLUMNS)

MISSING = object()


def dig(blob, path):
    """``a/b/c`` out of nested dicts; a missing step returns MISSING, so 'not recorded'
    stays distinguishable from 'recorded as null'."""
    node = blob
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def same(expected, observed) -> bool:
    if observed is MISSING:
        return False
    if isinstance(expected, bool) or isinstance(observed, bool):
        return bool(expected) == bool(observed)
    if isinstance(expected, float) or isinstance(observed, float):
        try:
            return math.isclose(float(expected), float(observed), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(observed)


def cell(value) -> str:
    """R10: a value the manifest does not carry is an empty cell, not a marker."""
    if value is MISSING or value is None:
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
    """Where the arm wrote, taken from the inventory rather than composed again here.

    This spelled the directory itself once -- size, batch and rate -- and so missed the
    architecture, which was added to the name after molnl's BERT medium and GPT-2 medium
    turned out to share a rate and to produce one name for two arms. Composing the name
    in a second place is how that happens: every arm then read as "not started yet" and
    §11 reported 0 arms with 0 disagreements, which looks like a pass. The inventory's
    out_dir is what the configs were generated from, so it is the only spelling.
    """
    try:
        name = arm["out_dir"]
    except KeyError:
        raise SystemExit("the arm inventory has no out_dir column; regenerate it with "
                         "scripts/molnl_batch_sweep_arms.py") from None
    if not name:
        raise SystemExit(f"{arm['arch']} {arm['size']} batch {arm['batch']}: "
                         f"the inventory's out_dir is empty")
    tree = "bert-output" if arm["arch"] == "bert" else "gpt2-output"
    return os.path.join(runs_root, "molecule_nat_lang", tree, name)


def eval_batch_from_config(arm, configs_dir):
    """per_device_eval_batch_size: the one value the BERT manifest does not record."""
    path = os.path.join(configs_dir, os.path.basename(arm["config"]))
    if not os.path.exists(path):
        return MISSING, path
    for line in open(path, encoding="utf-8"):
        if line.startswith("per_device_eval_batch_size"):
            return int(line.split("=", 1)[1].split("#")[0].strip()), path
    return MISSING, path


def expectation(spec, arm):
    """A literal, or the inventory column named by '@column'.

    A column is read as a number when it reads as one and left as text otherwise. Not
    every expectation is numeric: the three selection keys are compared against
    judge_metric, which is the name eval_loss_mask, and forcing that through float()
    raised on the first arm that had a manifest to read. ``same`` already compares
    non-numbers as strings.
    """
    if not isinstance(spec, str) or not spec.startswith("@"):
        return spec
    raw = arm[spec[1:]]
    if raw in ("", None):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except (TypeError, ValueError):
            continue
    return raw


def row_for(arm, runs_root, configs_dir, submit_record):
    where = run_dir(arm, runs_root)
    manifest_path = os.path.join(where, "run_manifest.json")
    if not os.path.exists(manifest_path):
        return None, manifest_path, []
    blob = json.load(open(manifest_path, encoding="utf-8"))
    keys = BERT_KEYS if arm["arch"] == "bert" else GPT2_KEYS
    config_path = os.path.join(configs_dir, os.path.basename(arm["config"]))

    row = {c: "" for c in COLUMNS}
    row.update({"arch": arm["arch"], "size": arm["size"], "batch": int(arm["batch"]),
                "lr": arm["lr"], "arm": os.path.basename(where)})
    disagreements = []
    sources = []
    for column, path, spec, _note in keys:
        observed = dig(blob, path) if path else MISSING
        want = expectation(spec, arm)
        row[column] = cell(observed)
        if want is not None and not same(want, observed):
            disagreements.append(f"{column}: manifest {cell(observed)!r} vs plan {want!r}")

    if arm["arch"] == "bert":
        observed, cfg = eval_batch_from_config(arm, configs_dir)
        row["per_device_eval_batch_size"] = cell(observed)
        want = expectation("@eval_batch", arm)
        if want is not None and not same(want, observed):
            disagreements.append(
                f"per_device_eval_batch_size: config {cell(observed)!r} vs plan {want!r}")
        sources.append(f"per_device_eval_batch_size<-{cfg}")

    row["disagreements"] = "; ".join(disagreements)
    row["value_sources"] = ("run_manifest.json"
                            + ("; " + "; ".join(sources) if sources else ""))
    row["config_path"] = os.path.abspath(config_path)
    row["manifest_path"] = os.path.abspath(manifest_path)
    row["out_dir"] = os.path.abspath(where)
    row["job_id"] = cell(dig(blob, "run/job_id"))
    row["commit"] = cell(dig(blob, "run/git/commit"))
    row["submit_record"] = os.path.abspath(submit_record) if submit_record else ""
    return row, manifest_path, disagreements


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--runs-root", required=True,
                    help="the MODEL_OUTPUT_ROOT the arms were launched with")
    ap.add_argument("--configs", default=os.path.join(
        here, "molcrawl", "tasks", "pretrain", "configs", "molecule_nat_lang"))
    ap.add_argument("--submit-record", default="",
                    help="the submission record file, for §11's location column")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    check_name(args.out)

    arms = read_arms(args.arms)
    rows, missing, problems = [], [], 0
    for arm in arms:
        row, manifest_path, bad = row_for(arm, args.runs_root, args.configs,
                                          args.submit_record)
        if row is None:
            missing.append((arm, manifest_path))
            continue
        rows.append(row)
        problems += len(bad)

    print(f"=== {len(arms)} arms; {len(rows)} with a manifest, {len(missing)} without ===")
    for arm, path in missing:
        print(f"  -- not started yet: {arm['arch']} {arm['size']} batch {arm['batch']} "
              f"lr {arm['lr']}  ({path})")
    for row in rows:
        if row["disagreements"]:
            print(f"  ** {row['arm']}")
            for item in row["disagreements"].split("; "):
                print(f"       {item}")
    compared = sum(
        sum(1 for k in (BERT_KEYS if r["arch"] == "bert" else GPT2_KEYS)
            if k[2] is not None)
        + (1 if r["arch"] == "bert" else 0)
        for r in rows)
    print(f"\n  {compared} values compared over {len(rows)} arms")
    print(f"disagreements: {problems}")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "several (columns: arch,size,batch,lr) -- every arm of the molnl "
                 "batch-size grid that has started",
        "source": "each arm's own run_manifest.json; the value_sources column names "
                  "any value read elsewhere, which is per_device_eval_batch_size alone "
                  "(the BERT manifest does not record it, so it comes from the config)",
        "generated": stamp("scripts/molnl_batch_sweep_manifests.py"),
        "note": [
            "one row per arm (§11); the expected values come from the derived arm "
            "inventory, which was itself checked against the directive's tables",
            f"{compared} values compared over {len(rows)} arms, {problems} "
            f"disagreement(s); a disagreeing arm names them in its own column",
            f"{len(missing)} of {len(arms)} arms have no manifest yet and have no row",
            "an empty value cell is one the manifest does not record",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"{len(rows)} rows -> {args.out}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
