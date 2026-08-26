"""Aggregate the genome BERT production runs into the CSVs under _results/.

Reads each run's newest checkpoint/trainer_state.json -- that file carries the
whole log_history, so nothing has to be scraped out of SLURM logs.

Two files come out:

  bert21-per-run.csv     one row per subset: the adopted checkpoint, the one HF
                         marked, all four eval metrics at both, and whether the
                         adopted one survived because HF protected it or because
                         it happened to land in the last save_total_limit slots.

  bert21-eval-series.csv every eval point of every run, with all four metrics
                         side by side.

All four metrics are kept everywhere. eval_loss alone is diluted by the copy
positions and hid a run that had learned nothing; eval_loss_copy falls to ~0.23
even in a model that has not learned, so it is not evidence on its own; and
eval_loss_random is where the differences show up most sharply. They are only
readable together.
"""

import csv
import glob
import json
import os

RUNS = "learning_source_genome_runs"
OUT = f"{RUNS}/_results"
METRICS = ("eval_loss", "eval_loss_mask", "eval_loss_copy", "eval_loss_random")
SAVE_TOTAL_LIMIT = 5


def load(run_dir):
    """Return (state, per-step metric dict, sorted saved checkpoint steps)."""
    steps = sorted(int(p.rsplit("-", 1)[1]) for p in glob.glob(f"{run_dir}/checkpoint-*"))
    state = json.load(open(f"{run_dir}/checkpoint-{steps[-1]}/trainer_state.json"))
    at = {}
    for entry in state["log_history"]:
        for m in METRICS:
            if m in entry:
                at.setdefault(entry["step"], {})[m] = entry[m]
    return state, at, steps


def main():
    os.makedirs(OUT, exist_ok=True)
    per_run, series = [], []

    for run_dir in sorted(glob.glob(f"{RUNS}/bert-small-*")):
        if not os.path.exists(f"{run_dir}/.run_complete"):
            continue
        subset = os.path.basename(run_dir).replace("bert-small-", "")
        state, at, saved = load(run_dir)

        hf_step = int(state["best_model_checkpoint"].rsplit("-", 1)[1])
        # Adopted = lowest eval_loss_mask among checkpoints that still exist.
        # Ties go to the earlier step.
        on_disk = {s: v for s, v in at.items() if s in set(saved) and "eval_loss_mask" in v}
        adopted = min(sorted(on_disk), key=lambda s: on_disk[s]["eval_loss_mask"])

        # The best eval_loss_mask over ALL evals, saved or not -- the gap shows
        # what the 1,000-step save interval costs.
        every = {s: v["eval_loss_mask"] for s, v in at.items() if "eval_loss_mask" in v}
        best_any = min(sorted(every), key=lambda s: every[s])
        final_step = max(every)

        # Why the adopted checkpoint still exists. Only the metric named in
        # metric_for_best_model is protected from save_total_limit's eviction;
        # anything else survives only by falling in the last few slots.
        if adopted == hf_step:
            survived = "protected"
        elif adopted in saved[-SAVE_TOTAL_LIMIT:]:
            survived = "recent-window"
        else:
            survived = "unexplained"

        row = {
            "subset": subset,
            "max_steps": state["max_steps"],
            "adopted_step": adopted,
            "hf_best_step": hf_step,
            "adopted_vs_hf_step": adopted - hf_step,
            "survived_by": survived,
            "ckpt_on_disk": int(os.path.isdir(f"{run_dir}/checkpoint-{adopted}")),
            "best_mask_any_step": best_any,
            "save_interval_cost": round(on_disk[adopted]["eval_loss_mask"] - every[best_any], 6),
            "final_step": final_step,
            "reversal": round(every[final_step] - every[best_any], 6),
        }
        for m in METRICS:
            row[f"adopted_{m}"] = round(on_disk[adopted].get(m, float("nan")), 6)
            row[f"hf_{m}"] = round(at.get(hf_step, {}).get(m, float("nan")), 6)
        per_run.append(row)

        for step in sorted(at):
            series.append({"subset": subset, "step": step,
                           **{m: at[step].get(m, "") for m in METRICS}})

    with open(f"{OUT}/bert21-per-run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)
    with open(f"{OUT}/bert21-eval-series.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subset", "step", *METRICS])
        w.writeheader()
        w.writerows(series)

    print(f"  {len(per_run)} runs -> {OUT}/bert21-per-run.csv")
    print(f"  {len(series)} eval points -> {OUT}/bert21-eval-series.csv")
    by = {}
    for r in per_run:
        by[r["survived_by"]] = by.get(r["survived_by"], 0) + 1
    print(f"  adopted checkpoint survived by: {by}")
    print(f"  all adopted checkpoints on disk: {all(r['ckpt_on_disk'] for r in per_run)}")


if __name__ == "__main__":
    main()
