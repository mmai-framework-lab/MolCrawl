"""Does each run still hold the checkpoint at its best evaluation point?

A reported loss is only usable downstream if the weights behind it exist. They do
not always: a callback that could not see the judged metric selected saves on a
different one, so a run's best step can have no checkpoint while worse steps do.
Found by hand on compounds BERT medium (job 151323) -- this makes the same check
over every run.

Three figures per run, all read from the run's own files:

  the minimum of ``judge_on`` over the whole log history, and its step
  ``best_metric`` / ``best_model_checkpoint``, which the Trainer picked from the
      checkpoints that happened to exist
  whether a checkpoint directory exists at the minimum's step

The first two differ exactly when the minimum was never saved, so a run where they
disagree is one whose reported best cannot be loaded. The nearest surviving
checkpoint and how much worse it is are reported so a substitute can be judged.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def newest_state(run_dir):
    """The trainer_state.json of the furthest checkpoint; it holds the whole history."""
    cks = [d for d in glob.glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(d)]
    if not cks:
        return None
    cks.sort(key=lambda d: int(os.path.basename(d).split("-")[1]))
    cand = os.path.join(cks[-1], "trainer_state.json")
    return cand if os.path.exists(cand) else None


def manifest_value(run_dir, *keys):
    """A dotted key out of run_manifest.json, or None."""
    p = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:                               # noqa: BLE001
        return None
    flat = {}

    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}.{k}" if pre else k)
        else:
            flat[pre] = o

    walk(d)
    for k in keys:
        if k in flat:
            return flat[k]
    return None


def inspect(run_dir):
    state_path = newest_state(run_dir)
    row = {"run": os.path.basename(run_dir), "dir": os.path.abspath(run_dir)}
    if not state_path:
        row["note"] = "no checkpoint"
        return row
    with open(state_path, encoding="utf-8") as fh:
        state = json.load(fh)

    metric = manifest_value(run_dir, "selection.judge_on", "eval.judge_on") or "eval_loss_mask"
    row["judge_on"] = metric
    row["save_on_improve"] = manifest_value(run_dir, "sources.save_on_improve.value")

    series = {e["step"]: e[metric] for e in state.get("log_history", [])
              if "step" in e and metric in e}
    if not series:
        row["note"] = f"{metric} not in log_history"
        return row

    have = sorted(int(os.path.basename(d).split("-")[1])
                  for d in glob.glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(d))
    best_step = min(series, key=series.get)
    row.update({
        "eval_points": len(series),
        "checkpoints": len(have),
        "best_step": best_step,
        "best_value": round(series[best_step], 6),
        "best_step_has_checkpoint": best_step in have,
        "trainer_best_metric": state.get("best_metric"),
        "trainer_best_checkpoint": os.path.basename(str(state.get("best_model_checkpoint") or "")),
    })
    if best_step not in have and have:
        near = min(have, key=lambda h: abs(h - best_step))
        row["nearest_kept_step"] = near
        if near in series:
            row["nearest_kept_value"] = round(series[near], 6)
            row["penalty"] = round(series[near] - series[best_step], 6)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", action="append", required=True,
                    help="shell glob of run directories; repeat for several")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    dirs = sorted({d for g in args.glob for d in glob.glob(g) if os.path.isdir(d)})
    print(f"  run ディレクトリ {len(dirs)} 件", flush=True)
    rows = [inspect(d) for d in dirs]

    cols = ["run", "judge_on", "save_on_improve", "eval_points", "checkpoints",
            "best_step", "best_value", "best_step_has_checkpoint",
            "trainer_best_metric", "trainer_best_checkpoint",
            "nearest_kept_step", "nearest_kept_value", "penalty", "note", "dir"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if r.get("best_step_has_checkpoint") is True]
    lost = [r for r in rows if r.get("best_step_has_checkpoint") is False]
    other = [r for r in rows if "best_step_has_checkpoint" not in r]
    print(f"\n  最良に checkpoint あり : {len(usable)}")
    print(f"  最良が失われている     : {len(lost)}")
    print(f"  判定できない           : {len(other)}")
    for r in lost:
        print(f"    {r['run'][:46]:46s} 最良 {r['best_value']} @ {r['best_step']} "
              f"→ 代替 {r.get('nearest_kept_step')} ({r.get('penalty'):+})"
              if r.get("penalty") is not None else
              f"    {r['run'][:46]:46s} 最良 @ {r['best_step']}、代替なし")
    print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
