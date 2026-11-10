#!/usr/bin/env python3
"""Is the checkpoint that was scored the one the trainer resolved as best?

The 2026-11-10 judgment asks this in 5.11. Two of the three BERT models were scored at
checkpoint-24000, which is also the last step of a 24,000-step schedule, so the table
cannot say whether the best was taken or the end was taken. The third was scored at
23,800, which reads as a resolved best -- but reading it that way is a guess until the
run's own record is opened. The same confusion produced a wrong answer for protein, where
ProteinGym turned out to have scored small and medium at their terminal checkpoints.

For each run this prints what the run recorded, not what it ought to have recorded:

  best_model_checkpoint   the path the trainer settled on, and its best_metric
  the metric it compared  ``metric_for_best_model`` from the run manifest beside the
                          series actually present in the trainer's log_history. These
                          runs predate c792b7e7 (2026-10-16), where the saving decision
                          was found to be following a different series from the one the
                          run was judged on, so the two are printed side by side rather
                          than assumed equal.
  the argmin on disk      the lowest value in the log_history and the step it sits at,
                          which says whether the terminus happens to be the best
  scored                  whether the checkpoint handed to the retrieval scoring is that
                          path, and whether it is the terminus

    python scripts/molnl_bert_adopted_checkpoint.py --run <dir>=<scored checkpoint> ...
        --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import re


def latest_state(run_dir):
    """The trainer_state.json of the highest-numbered checkpoint in the run."""
    best, best_step = None, -1
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        path = os.path.join(run_dir, name, "trainer_state.json")
        if os.path.exists(path) and int(m.group(1)) > best_step:
            best, best_step = path, int(m.group(1))
    return best, best_step


def series_of(log_history):
    """Every eval series in the log, as {name: [(step, value)]}."""
    out = {}
    for entry in log_history:
        step = entry.get("step")
        for key, value in entry.items():
            if not key.startswith("eval_") or not isinstance(value, (int, float)):
                continue
            if key.endswith(("_runtime", "_samples_per_second", "_steps_per_second")):
                continue
            out.setdefault(key, []).append((step, float(value)))
    return out


def answer(run_dir, scored):
    state_path, last_ckpt = latest_state(run_dir)
    if not state_path:
        return {"run": run_dir, "error": "no checkpoint with a trainer_state.json"}
    state = json.load(open(state_path, encoding="utf-8"))
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) \
        if os.path.exists(manifest_path) else {}
    selection = manifest.get("selection", {})
    series = series_of(state.get("log_history", []))
    best = state.get("best_model_checkpoint")
    out = {
        "run": run_dir,
        "trainer_state_read_from": state_path,
        "global_step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "best_model_checkpoint": best,
        "best_metric": state.get("best_metric"),
        "metric_for_best_model": selection.get("metric_for_best_model"),
        "judge_on": selection.get("judge_on"),
        "eval_series_in_log": {k: len(v) for k, v in sorted(series.items())},
        "scored_checkpoint": scored,
        "scored_is_best_model_checkpoint": (
            bool(best) and os.path.realpath(scored) == os.path.realpath(best)),
        "scored_is_the_terminus": (
            os.path.basename(scored.rstrip("/")) == f"checkpoint-{state.get('max_steps')}"),
        "checkpoints_on_disk": sorted(
            int(m.group(1)) for m in
            (re.fullmatch(r"checkpoint-(\d+)", n) for n in os.listdir(run_dir)) if m),
        "last_checkpoint_on_disk": last_ckpt,
    }
    for name, points in series.items():
        step, value = min(points, key=lambda t: t[1])
        out[f"argmin_{name}"] = {"step": step, "value": value,
                                 "is_the_terminus": step == state.get("max_steps")}
        out[f"last_{name}"] = {"step": points[-1][0], "value": points[-1][1]}
    metric = selection.get("metric_for_best_model")
    out["metric_for_best_model_is_in_the_log"] = bool(metric) and metric in series
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, metavar="DIR=CHECKPOINT",
                    help="a run directory and the checkpoint the scoring was given")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    answers = []
    for item in a.run:
        run_dir, _, scored = item.partition("=")
        if not scored:
            raise SystemExit(f"--run takes DIR=CHECKPOINT, got {item!r}")
        answers.append(answer(run_dir, scored))
    for row in answers:
        print(f"  {os.path.basename(row['run'])}", flush=True)
        for key, value in row.items():
            if key == "run":
                continue
            print(f"      {key}: {value}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(answers, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
