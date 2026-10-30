#!/usr/bin/env python3
"""Check whether each BERT run's saved best checkpoint is the one its judged metric picked.

Directive `all-checkpoint-metric-order-2026-10-28-v2.md`:

* §3/§4/§6 -- for every BERT run, compare the step in ``best_model_checkpoint``
  against the step where ``eval_loss_mask`` reaches its minimum over the whole
  ``log_history``, and say whether the checkpoint at that step still exists.
* §7 -- the conditions each run was taken under: validation set, seed, epochs
  configured against epochs actually run.

Reads `trainer_state.json` and `run_manifest.json` only. Nothing is loaded onto a
GPU, no weight file is opened and nothing is written inside a run directory.

Why `log_history` and not `best_metric`: `MlmBreakdownMixin.evaluate` adds
`eval_loss_mask` through `self.log()` after `super().evaluate()` has returned, so the
mask metric lands in its own `log_history` row rather than in the eval row beside
`eval_loss`. Both are scanned. `best_metric` is whatever the Trainer was tracking,
which is the point of the comparison and so is never used as the answer.

GPT-2 runs are reported too, for §7, and are marked `arch=gpt2`: they carry a
`run_manifest.json` but no `trainer_state.json`, because nanoGPT keeps its own
selection (see §2). Their val curve is read from `logging_*.csv`.

Usage:
    python scripts/audit_checkpoint_metric_match.py \
        --root compounds=<dir> --root genome_sequence=<dir> \
        --report <out.json> --tsv <out.tsv>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# The metric every BERT run in this project is judged on. Taken from the manifest
# per run as well; this is the fallback when a run has no manifest.
JUDGED = "eval_loss_mask"
# A checkpoint directory counts as present only if the weights are in it. The
# aborted compounds runs left directories holding rng_state_*.pth and nothing else
# (see the 2026-10-19 rank-divergence failures), and those cannot be evaluated.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def checkpoint_steps(run_dir):
    """Every checkpoint step in a run directory, and which of them hold weights."""
    steps = {}
    try:
        entries = os.listdir(run_dir)
    except OSError:
        return steps
    for name in entries:
        m = CHECKPOINT_RE.match(name)
        if not m:
            continue
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            inner = set(os.listdir(path))
        except OSError:
            inner = set()
        steps[int(m.group(1))] = {
            "has_weights": any(f in inner for f in WEIGHT_FILES),
            "has_trainer_state": "trainer_state.json" in inner,
        }
    return steps


def newest_trainer_state(run_dir, steps):
    """The trainer_state.json covering the most steps.

    A resume reloads the state and keeps appending, so the highest checkpoint holds
    the longest history. A run-root trainer_state.json (written by save_state at the
    end) is preferred when it is there, since it covers the final step as well.
    """
    root_state = os.path.join(run_dir, "trainer_state.json")
    candidates = []
    if os.path.isfile(root_state):
        candidates.append((None, root_state))
    for step in sorted(steps, reverse=True):
        if steps[step]["has_trainer_state"]:
            candidates.append((step, os.path.join(run_dir, f"checkpoint-{step}",
                                                  "trainer_state.json")))
    best = None
    for step, path in candidates:
        try:
            with open(path) as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            continue
        reach = int(state.get("global_step") or 0)
        if best is None or reach > best[0]:
            best = (reach, path, step, state)
    return best


def metric_minimum(log_history, key):
    """Smallest value of `key` over the whole history, with its step and any ties.

    Ties are counted rather than broken silently: the CSV the curves are drawn from
    rounds to four decimals, and four of 24 apparent mismatches in the 2026-09-15
    audit were ties at that rounding where the run itself compared in full precision.
    """
    rows = [r for r in log_history if key in r and r.get(key) is not None]
    if not rows:
        return None
    values = [(float(r[key]), int(r.get("step") or 0)) for r in rows]
    best_value = min(v for v, _ in values)
    tied = sorted(s for v, s in values if v == best_value)
    return {
        "value": best_value,
        "step": tied[0],
        "points": len(values),
        "first_step": min(s for _, s in values),
        "last_step": max(s for _, s in values),
        "tied_steps": tied if len(tied) > 1 else [],
    }


# The size tokens the run directories use, longest first so `ex-large` is not read
# as `large`. Only consulted when the manifest has no `introduced.model_size`,
# which is the case for every nanoGPT run and for the genome BERT campaign.
SIZE_RE = re.compile(r"(?<![a-z])(ex-large|xlarge|xl|large|medium|small|tiny)(?![a-z])")


def model_size(run_name, manifest):
    """The size, from the manifest when it recorded one, else from the run's name."""
    recorded = dig(manifest or {}, "introduced.model_size")
    if recorded:
        return str(recorded), "manifest"
    m = SIZE_RE.search(run_name)
    if m:
        return m.group(1), "run name"
    return None, None


def dig(mapping, path, default=None):
    """Read a dotted path out of nested dicts."""
    node = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_manifest(run_dir):
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def gpt2_val_curve(run_dir):
    """Lowest val loss and its iteration, over every logging CSV in the run.

    A resume opens a new timestamped CSV, so the segments are read together.
    """
    rows = []
    try:
        names = sorted(f for f in os.listdir(run_dir)
                       if f.startswith("logging_") and f.endswith(".csv"))
    except OSError:
        return None
    for name in names:
        try:
            with open(os.path.join(run_dir, name)) as fh:
                for row in csv.DictReader(fh, skipinitialspace=True):
                    try:
                        it = int(row["iter"])
                        val = float(row["val_loss"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append((val, it))
        except OSError:
            continue
    if not rows:
        return None
    best_value = min(v for v, _ in rows)
    tied = sorted(i for v, i in rows if v == best_value)
    return {
        "segments": len(names),
        "points": len(rows),
        "min_value": best_value,
        "min_iter": tied[0],
        "tied_iters": tied if len(tied) > 1 else [],
        "last_iter": max(i for _, i in rows),
    }


def epochs(train_rows, global_batch, steps_run, max_steps):
    """Steps per epoch, and the epochs the configured and the completed step counts are.

    §7 asks for the actual value as completed steps divided by steps per epoch, so
    it is computed that way here rather than taken from the Trainer's own `epoch`
    field, which counts whole dataloader passes and so reads slightly lower.
    """
    if not train_rows or not global_batch:
        return {}
    per_epoch = train_rows / global_batch
    out = {"steps_per_epoch": per_epoch}
    if max_steps:
        out["epochs_configured_from_max_steps"] = max_steps / per_epoch
    if steps_run:
        out["epochs_run"] = steps_run / per_epoch
    return out


def audit_bert(run_dir, modality, manifest, steps):
    found = newest_trainer_state(run_dir, steps)
    if found is None:
        return None
    reach, state_path, from_step, state = found
    history = state.get("log_history") or []

    judged = str(dig(manifest or {}, "eval.judge_on") or JUDGED)
    tracked = str(dig(manifest or {}, "selection.metric_for_best_model") or "")

    best_ckpt = state.get("best_model_checkpoint") or ""
    best_step = None
    if best_ckpt:
        m = CHECKPOINT_RE.match(os.path.basename(best_ckpt.rstrip("/")))
        if m:
            best_step = int(m.group(1))

    judged_min = metric_minimum(history, judged)
    loss_min = metric_minimum(history, "eval_loss")

    steps_run = int(state.get("global_step") or 0)
    eval_steps = int(state.get("eval_steps") or 0) or int(
        dig(manifest or {}, "selection.eval_steps") or 0)
    # Whether the history in this file actually covers the run. A resume that lost
    # its state, or a trainer_state read from an early checkpoint, would make the
    # minimum a minimum over part of the run only, and that must not be reported
    # as the run's.
    expected_points = (steps_run // eval_steps) if eval_steps else None
    coverage = None
    if expected_points and judged_min:
        coverage = judged_min["points"] / expected_points

    at_min = None
    if judged_min is not None:
        at_min = steps.get(judged_min["step"])

    row = {
        "modality": modality,
        "arch": "bert",
        "run": os.path.basename(run_dir),
        "run_dir": run_dir,
        "size": model_size(os.path.basename(run_dir), manifest)[0],
        "size_from": model_size(os.path.basename(run_dir), manifest)[1],
        "learning_rate": dig(manifest or {}, "schedule.learning_rate"),
        # Two records of the seed, and they disagree on some runs. `seed.seed` is
        # getattr(args, "seed") -- the value transformers actually seeded with, since
        # Trainer.__init__ calls set_seed(self.args.seed). `sources.seed.value` is
        # what the config asked for. `seed=seed` stopped being passed to
        # TrainingArguments in 0f3e7067 (2026-10-02), so from then on the config's
        # seed reached only data_seed, which 4.45.1 never reads. Both are carried
        # here rather than one being chosen, because §7 asks for the value used.
        "seed": dig(manifest or {}, "seed.seed"),
        "seed_configured": dig(manifest or {}, "sources.seed.value"),
        "seed_from": dig(manifest or {}, "sources.seed.from"),
        "data_seed": dig(manifest or {}, "seed.data_seed"),
        "judged_on": judged,
        "metric_for_best_model": tracked,
        "save_on_improve": dig(manifest or {}, "sources.save_on_improve.value"),
        "trainer_state_from": "run root" if from_step is None else f"checkpoint-{from_step}",
        "steps_run": steps_run,
        "max_steps": int(state.get("max_steps") or 0) or dig(manifest or {}, "schedule.max_steps"),
        "eval_steps": eval_steps or None,
        "save_steps": int(state.get("save_steps") or 0) or None,
        "best_metric_recorded": state.get("best_metric"),
        "best_model_checkpoint_step": best_step,
        "judged_min": judged_min,
        "eval_loss_min": loss_min,
        "history_points_expected": expected_points,
        "history_coverage": coverage,
        "checkpoint_at_judged_min": at_min,
        "checkpoints_present": sorted(steps),
        "epoch_field": state.get("epoch"),
        "num_train_epochs_field": state.get("num_train_epochs"),
        "eval": {
            "rows": dig(manifest or {}, "eval.rows"),
            "subset_random": dig(manifest or {}, "eval.subset_random"),
            "subset_seed": dig(manifest or {}, "eval.subset_seed"),
            "fixed_mask": dig(manifest or {}, "eval.fixed_mask.enabled"),
            "fixed_mask_seed": dig(manifest or {}, "eval.fixed_mask.seed"),
        },
        "job_id": dig(manifest or {}, "run.job_id"),
    }
    row.update(epochs(
        dig(manifest or {}, "data.rows.train"),
        dig(manifest or {}, "batch.effective_global_batch"),
        steps_run,
        row["max_steps"],
    ))
    row["train_rows"] = dig(manifest or {}, "data.rows.train")
    row["global_batch"] = dig(manifest or {}, "batch.effective_global_batch")

    if judged_min is None:
        row["match"] = "no judged metric in history"
    elif best_step is None:
        row["match"] = "no best_model_checkpoint"
    else:
        row["match"] = "yes" if best_step == judged_min["step"] else "no"
    return row


def audit_gpt2(run_dir, modality, manifest, steps):
    curve = gpt2_val_curve(run_dir)
    steps_run = max(steps) if steps else (curve or {}).get("last_iter")
    row = {
        "modality": dig(manifest, "data.modality") or modality,
        "arch": "gpt2",
        "run": os.path.basename(run_dir),
        "run_dir": run_dir,
        "size": model_size(os.path.basename(run_dir), manifest)[0],
        "size_from": model_size(os.path.basename(run_dir), manifest)[1],
        "learning_rate": dig(manifest, "schedule.learning_rate"),
        # nanoGPT seeds with torch.manual_seed(seed + ddp_rank) in train.py, so the
        # config's value does reach the run here -- unlike the BERT path.
        "seed": dig(manifest, "seed.seed"),
        "seed_configured": dig(manifest, "sources.seed.value"),
        "seed_from": dig(manifest, "sources.seed.from"),
        "seed_per_rank_offset": dig(manifest, "seed.per_rank_offset"),
        "judged_on": dig(manifest, "eval.judge_on"),
        "metric_for_best_model": "n/a (nanoGPT keeps its own best_val_loss)",
        "steps_run": steps_run,
        "max_steps": dig(manifest, "schedule.max_iters"),
        "eval_steps": dig(manifest, "eval.eval_interval"),
        "val_curve": curve,
        "checkpoints_present": sorted(steps),
        "eval": {
            "rows": dig(manifest, "eval.eval_sequences"),
            "eval_iters": dig(manifest, "eval.eval_iters"),
            "deterministic_val_eval": dig(manifest, "eval.deterministic_val_eval"),
            "always_save_checkpoint": dig(manifest, "selection.always_save_checkpoint"),
            "max_checkpoints": dig(manifest, "selection.max_checkpoints"),
        },
        "job_id": dig(manifest, "run.job_id"),
        "match": "n/a (§2: not through this decision)",
    }
    row.update(epochs(
        dig(manifest, "data.rows.train"),
        dig(manifest, "batch.effective_global_batch"),
        steps_run,
        row["max_steps"],
    ))
    row["train_rows"] = dig(manifest, "data.rows.train")
    row["global_batch"] = dig(manifest, "batch.effective_global_batch")
    return row


def walk_root(root, modality):
    """Every run directory under `root`, with the checkpoint dirs not descended into."""
    rows = []
    skipped = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Never walk into a checkpoint: its contents are listed directly when needed.
        dirnames[:] = [d for d in dirnames if not CHECKPOINT_RE.match(d)]
        steps = checkpoint_steps(dirpath)
        manifest = read_manifest(dirpath)
        is_run = bool(steps) or manifest is not None
        if not is_run:
            continue
        framework = str(dig(manifest or {}, "framework") or "")
        has_hf_state = any(v["has_trainer_state"] for v in steps.values()) or \
            os.path.isfile(os.path.join(dirpath, "trainer_state.json"))
        if framework == "nanoGPT" or (not has_hf_state and
                                      any(f.startswith("logging_")
                                          for f in filenames)):
            rows.append(audit_gpt2(dirpath, modality, manifest or {}, steps))
            continue
        row = audit_bert(dirpath, modality, manifest, steps)
        if row is None:
            skipped.append({"modality": modality, "run_dir": dirpath,
                            "reason": "no readable trainer_state.json",
                            "checkpoints": sorted(steps),
                            "has_manifest": manifest is not None})
            continue
        rows.append(row)
    return rows, skipped


# Diagnostic, throughput and smoke runs, named by the convention the run names
# follow. They are measured and reported like any other run; this rule only marks
# them, because §7 asks for the conditions a *result* was taken under and these
# runs produced none. Lived in build_run_conditions_deck.py until 2026-10-30; it is
# here so the deck, the TSV and the §1 extract cannot disagree about what counts.
DIAGNOSTIC = re.compile(
    r"diag|smoke|tput|microbatch|trial|aborted|pathcheck|envprobe|dsvl|lrprobe"
    r"|lrpilot|reprobe|demoH4|8x80|effbatch|nodocmask|docmask-P|warmup2k|mlm015"
    r"|esmopt|bertswap|stopped|rpb-|sat9|bf16$|fp32$",
    re.I,
)


def is_production(row):
    """Whether this run's numbers stand behind a reported result.

    Three ways out: the DeepSpeed comparison tree (`other_ds`), a name matching
    DIAGNOSTIC, and a run with no job_id -- the generation that predates
    run_manifest.json, whose conditions cannot be stated.
    """
    if row.get("modality") == "other_ds":
        return False
    if DIAGNOSTIC.search(row.get("run") or ""):
        return False
    return bool(row.get("job_id"))


TSV_COLUMNS = [
    # §6 names the columns this TSV must carry; the rest are what §4 asks to be
    # able to say without opening the JSON.
    ("modality", lambda r: r["modality"]),
    ("arch", lambda r: r["arch"]),
    ("size", lambda r: r.get("size") or ""),
    ("learning_rate", lambda r: r.get("learning_rate") or ""),
    ("run", lambda r: r["run"]),
    ("seed", lambda r: r.get("seed") if r.get("seed") is not None else ""),
    ("steps_run", lambda r: r.get("steps_run") or ""),
    ("best_model_checkpoint_step", lambda r: r.get("best_model_checkpoint_step")
     if r.get("best_model_checkpoint_step") is not None else ""),
    ("judged_metric", lambda r: r.get("judged_on") or ""),
    ("judged_min_step", lambda r: (r.get("judged_min") or {}).get("step", "")),
    ("best_metric_recorded", lambda r: _num(r.get("best_metric_recorded"))),
    ("judged_min_value", lambda r: _num((r.get("judged_min") or {}).get("value"))),
    ("match", lambda r: r.get("match") or ""),
    ("checkpoint_at_judged_min", lambda r: _ckpt(r.get("checkpoint_at_judged_min"))),
    ("eval_loss_min_step", lambda r: (r.get("eval_loss_min") or {}).get("step", "")),
    ("eval_loss_min_value", lambda r: _num((r.get("eval_loss_min") or {}).get("value"))),
    ("eval_points", lambda r: (r.get("judged_min") or {}).get("points", "")),
    ("history_coverage", lambda r: _num(r.get("history_coverage"), 3)),
    ("job_id", lambda r: r.get("job_id") or ""),
    ("production", lambda r: "yes" if r.get("production") else "no"),
]


def eval_set_description(row):
    """§7's validation-set column: the same set every time, or redrawn.

    BERT reads a fixed leading or seeded-random subset of the split and reads the
    same rows at every eval point. GPT-2 reads a fixed prefix of one constant
    permutation only when deterministic_val_eval is on; with it off, every eval
    resamples with replacement.
    """
    ev = row.get("eval") or {}
    if row["arch"] == "bert":
        rows = ev.get("rows")
        if rows is None:
            return "不明（manifest 無し）"
        how = "無作為抽出" if ev.get("subset_random") else "先頭取り"
        seed = ev.get("subset_seed")
        text = f"固定 {int(rows):,} 行（{how}"
        if ev.get("subset_random") and seed is not None:
            text += f"、seed {seed}"
        text += "）"
        if ev.get("fixed_mask"):
            text += f" + mask 固定 seed {ev.get('fixed_mask_seed')}"
        return text
    det = ev.get("deterministic_val_eval")
    seqs = ev.get("rows")
    if det is None:
        return "不明（manifest 無し）"
    if det:
        return f"固定 {int(seqs):,} 系列（定数 seed の並べ替えの先頭）" if seqs else "固定（定数 seed の並べ替えの先頭）"
    return f"引き直し（毎回 {int(seqs):,} 系列を復元抽出）" if seqs else "引き直し（毎回復元抽出）"


def seed_description(row):
    """§7's seed column: the value the run used, and the value it was asked for.

    They differ on the BERT runs started after 0f3e7067 stopped passing `seed=` to
    TrainingArguments: transformers seeded with its own default 42 while the config
    named something else.
    """
    used, asked = row.get("seed"), row.get("seed_configured")
    if used is None:
        return "不明（manifest 無し）"
    if asked is None or int(asked) == int(used):
        return str(used)
    return f"{used}（config は {asked} を指定。効いていない）"


CONDITION_COLUMNS = [
    ("modality", lambda r: r["modality"]),
    ("arch", lambda r: r["arch"]),
    ("size", lambda r: r.get("size") or ""),
    ("learning_rate", lambda r: r.get("learning_rate") or ""),
    ("run", lambda r: r["run"]),
    ("検証集合", eval_set_description),
    ("seed", seed_description),
    ("steps_per_epoch", lambda r: _num(r.get("steps_per_epoch"), 2)),
    ("epochs_設定", lambda r: _num(r.get("epochs_configured_from_max_steps"), 2)),
    ("epochs_実際", lambda r: _num(r.get("epochs_run"), 2)),
    ("max_steps_設定", lambda r: r.get("max_steps") or ""),
    ("steps_完走", lambda r: r.get("steps_run") or ""),
    ("train_rows", lambda r: r.get("train_rows") or ""),
    ("global_batch", lambda r: r.get("global_batch") or ""),
    ("eval_interval", lambda r: r.get("eval_steps") or ""),
    ("job_id", lambda r: r.get("job_id") or ""),
]


def _num(value, places=6):
    return "" if value is None else f"{float(value):.{places}f}"


def _ckpt(entry):
    if entry is None:
        return "absent"
    return "present" if entry.get("has_weights") else "no weights"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, metavar="MODALITY=DIR")
    ap.add_argument("--report", required=True)
    ap.add_argument("--tsv", help="§6: one row per run, the metric comparison")
    ap.add_argument("--tsv-conditions", help="§7: one row per run, the conditions")
    ap.add_argument("--tsv-mismatches",
                    help="§1 of the 2026-10-30 judgment: the same columns as "
                         "--tsv, carrying the production BERT runs whose saved "
                         "best checkpoint is not the judged minimum")
    args = ap.parse_args(argv)

    rows, skipped = [], []
    # Several of these trees reach the same run through a symlink -- the rna build
    # is mounted under two names -- so a run is kept once, by its real path.
    seen = {}
    for item in args.root:
        if "=" not in item:
            raise SystemExit(f"--root wants MODALITY=DIR, got {item!r}")
        modality, root = item.split("=", 1)
        if not os.path.isdir(root):
            raise SystemExit(f"{modality}: not a directory: {root}")
        print(f"=== {modality}: {root}", flush=True)
        got, miss = walk_root(root, modality)
        fresh = []
        for row in got:
            real = os.path.realpath(row["run_dir"])
            if real in seen:
                continue
            seen[real] = row["run_dir"]
            fresh.append(row)
        if len(fresh) != len(got):
            print(f"  {len(got) - len(fresh)} already counted under another root",
                  flush=True)
        got = fresh
        rows.extend(got)
        skipped.extend(miss)
        bert = [r for r in got if r["arch"] == "bert"]
        mismatched = [r for r in bert if r["match"] == "no"]
        print(f"  {len(got)} runs ({len(bert)} bert), "
              f"{len(mismatched)} mismatched, {len(miss)} unreadable", flush=True)
        for r in mismatched:
            jm = r["judged_min"]
            print(f"    MISMATCH {r['run']}: best={r['best_model_checkpoint_step']} "
                  f"{r['judged_on']} min={jm['step']} ({jm['value']:.6f}) "
                  f"{_ckpt(r['checkpoint_at_judged_min'])}", flush=True)

    rows.sort(key=lambda r: (r["modality"], r["arch"], str(r.get("size")), r["run"]))
    for row in rows:
        row["production"] = is_production(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump({"runs": rows, "unreadable": skipped}, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}", flush=True)

    mismatched_production = [r for r in rows if r["arch"] == "bert"
                             and r["match"] == "no" and r["production"]]
    for path, columns, subset in ((args.tsv, TSV_COLUMNS, rows),
                                  (args.tsv_conditions, CONDITION_COLUMNS, rows),
                                  (args.tsv_mismatches, TSV_COLUMNS,
                                   mismatched_production)):
        if not path:
            continue
        # A BOM, because these are opened in Excel and read in Slack, where a
        # UTF-8 file without one is taken for binary (2026-08-29).
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow([name for name, _ in columns])
            for row in subset:
                writer.writerow([fn(row) for _, fn in columns])
        print(f"wrote {path} ({len(subset)} rows)", flush=True)

    bert = [r for r in rows if r["arch"] == "bert"]
    print(f"\n{len(rows)} runs, {len(bert)} bert, "
          f"{sum(1 for r in bert if r['match'] == 'yes')} match, "
          f"{sum(1 for r in bert if r['match'] == 'no')} mismatch, "
          f"{sum(1 for r in bert if r['match'] not in ('yes', 'no'))} undecided")
    prod = [r for r in bert if r["production"]]
    print(f"production bert {len(prod)}, "
          f"{sum(1 for r in prod if r['match'] == 'yes')} match, "
          f"{len(mismatched_production)} mismatch")
    by_modality = {}
    for r in mismatched_production:
        by_modality[r["modality"]] = by_modality.get(r["modality"], 0) + 1
    for modality, count in sorted(by_modality.items(), key=lambda kv: -kv[1]):
        print(f"  {modality}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
