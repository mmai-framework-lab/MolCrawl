"""The batch-size grid's loss curves and per-arm summary, read from what the runs wrote.

2026-11-19c §10 asks, for every arm: the lowest validation loss and the column count it
sits at, the value at the budget, the step reached, and whether the arm collapsed. §12
asks for one curve file holding all the arms, with the batch and the columns and tokens
read so far added to the columns the 2026-11-02 curve file carried.

Both come from one pass over what the runs wrote, so the summary cannot disagree with
the curve it is read off. The arm's identity comes from the inventory rather than from
the log, and the log is found by the job name the submitter built out of that same
inventory, so an arm cannot be mislabelled by a header line.

    compounds_batch_sweep_curves.py --arms rt1-...tsv --logs <dir> \
        --curves-out rt1-...tsv --summary-out rt1-...tsv [--arch gpt2]

The GPT-2 reader takes nanoGPT's own evaluation line out of the run's log. The BERT
reader takes the evaluation history out of the newest checkpoint's trainer_state.json,
which is where the HF Trainer keeps it: the file in the newest checkpoint carries the
whole history, including the part written before a resume, so one file per arm is the
whole series. The log is not used for BERT, because the breakdown line it prints does
not carry eval_loss and a resumed arm has its history split over several logs.

The Trainer writes one evaluation entry per evaluation and the breakdown callback adds
a second at the same step, so the two are merged by step before anything is counted --
counting entries would report twice the evaluation points.

Read-only over the runs: it writes two TSVs and launches nothing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: nanoGPT's evaluation line.
STEP = re.compile(r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)")

SEQUENCE_LENGTH = 1024
SEED = 42

#: The degenerate floor measured for this corpus, in nats per token of next-token
#: prediction. An arm whose validation loss ends above it has not learned context; it
#: is reported rather than left to be noticed. It is a GPT-2 figure and says nothing
#: about a masked-token loss, so the BERT rows leave `collapsed` empty rather than
#: compare against a floor measured for another objective -- see BERT_FLOOR below.
DEGENERATE_FLOOR = 2.395

#: What a BERT arm is judged on, and the series the 2026-11-20 directive asks to be
#: shown beside it. eval_loss averages over every position, including the ~10 per cent
#: whose answer is visible in the input; eval_loss_mask is the [MASK] positions alone.
BERT_JUDGED = "eval_loss_mask"
BERT_BESIDE = "eval_loss"

#: The masked-token floor: the unigram cross-entropy at the [MASK] positions, which is
#: what eval_loss_mask reads when a run has learned nothing beyond the token marginal.
#: 2.4733, measured on this packed build on 2026-11-02 by scripts/mlm_lookup_baselines.py
#: (`report/mlm-baselines-20261102/full/compounds.json`: valid split, 10,000 rows,
#: 1,595,521 masked positions, tables fitted on 200,000 train rows, mlm_probability 0.2)
#: -- the same split, row count and masking probability the grid's arms evaluate on.
#: The companion blended figure 2.2242 belongs beside eval_loss, which is scored over
#: the copyable positions too, and is not the line for the judged metric.
#:
#: The 2.395 in the arms' configs is the August derivation of that blended figure. It
#: sits just under this floor, so an arm that collapses to the unigram solution lands
#: above it and the collapse detector stops the run.
BERT_FLOOR = 2.4733

#: The line the collapse detector prints when it ends a run, and the warning that
#: carries the value it ended on. A collapsed arm stops improving, so nothing is saved
#: after it turns: the newest checkpoint's history can end before the collapse, and
#: reading only that file reports the arm as healthy. The log is the record of how the
#: run ended, so it is what decides.
COLLAPSE_STOP = "STOPPING: the run has collapsed"
ABOVE_THRESHOLD = re.compile(
    r"eval_loss_mask=([\d.]+) is above the degenerate threshold ([\d.]+) \((\d+)/(\d+)\)")

CURVE_COLUMNS = ["arch", "size", "batch", "lr", "seed", "step", "metric", "value",
                 "rows_seen", "tokens_seen", "source"]
SUMMARY_COLUMNS = ["arch", "size", "batch", "lr", "steps_planned", "steps_reached",
                   "eval_points", "judged_metric", "best_value", "best_step",
                   "best_rows", "best_tokens", "best_fraction_of_budget",
                   "final_value", "final_step", "final_minus_best",
                   "beside_metric", "beside_at_best", "beside_final",
                   "collapsed", "ended", "stopped_at_value", "source"]


def read_arms(path, arch):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines, delimiter="\t") if r["arch"] == arch]
    if not rows:
        raise SystemExit(f"{path}: no {arch} arm")
    return rows


def log_for(row, log_dir):
    """The arm's log, found by the job name the submitter built from the inventory."""
    stem = (f"mc-cmp-{row['arch']}-{row['size']}-b{row['batch']}"
            f"-lr{row['lr_slug']}")
    found = sorted(glob.glob(os.path.join(log_dir, f"{stem}-*.out")))
    return found


def series(paths):
    """Every evaluation point in these logs, in step order, later wins on a repeat."""
    points = {}
    for path in paths:
        with open(path, errors="ignore") as handle:
            for step, train, val in STEP.findall(handle.read()):
                points[int(step)] = (float(train), float(val))
    return [(s, *points[s]) for s in sorted(points)]


def rows_for(arms, log_dir):
    curves, summary, missing = [], [], []
    for arm in arms:
        paths = log_for(arm, log_dir)
        if not paths:
            missing.append(arm)
            continue
        points = series(paths)
        if not points:
            missing.append(arm)
            continue
        batch = int(arm["batch"])
        source = " ".join(os.path.basename(p) for p in paths)
        for step, train, val in points:
            for metric, value in (("train_loss", train), ("val_loss", val)):
                curves.append({
                    "arch": arm["arch"], "size": arm["size"], "batch": batch,
                    "lr": arm["lr"], "seed": SEED, "step": step,
                    "metric": metric, "value": f"{value:.6f}",
                    "rows_seen": step * batch,
                    "tokens_seen": step * batch * SEQUENCE_LENGTH,
                    "source": source,
                })
        best_val, best_step = min((v, s) for s, _, v in points)
        final_step, _, final_val = points[-1]
        summary.append({
            "arch": arm["arch"], "size": arm["size"], "batch": batch, "lr": arm["lr"],
            "steps_planned": int(arm["steps"]), "steps_reached": final_step,
            "eval_points": len(points),
            "judged_metric": arm["judge_metric"],
            "best_value": f"{best_val:.6f}", "best_step": best_step,
            "best_rows": best_step * batch,
            "best_tokens": best_step * batch * SEQUENCE_LENGTH,
            # 2026-11-20 §5: where the minimum sits as a fraction of the budget, and
            # how far the arm has come back up by the time the budget runs out. Both
            # are read off the same series as the two values above, so neither can
            # disagree with them.
            "best_fraction_of_budget": f"{best_step / int(arm['steps']):.4f}",
            "final_value": f"{final_val:.6f}", "final_step": final_step,
            "final_minus_best": f"{final_val - best_val:+.6f}",
            # The second series is a BERT column: nothing is judged beside val_loss
            # here, and a missing value is an empty cell (column convention R10).
            "beside_metric": "", "beside_at_best": "", "beside_final": "",
            "collapsed": "yes" if final_val >= DEGENERATE_FLOOR else "no",
            "ended": "budget" if final_step >= int(arm["steps"]) else "running",
            "stopped_at_value": "",
            "source": source,
        })
    return curves, summary, missing


def newest_state(run_dir):
    """The newest checkpoint's trainer_state.json, which holds the whole history.

    Retention keeps the newest checkpoint as well as the best ones, so the file with
    the largest step exists for a running arm and for a finished one alike. Picking it
    by step rather than by modification time matters: retention rewrites nothing, but
    a resumed arm touches older directories.
    """
    best = None
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        match = re.search(r"checkpoint-(\d+)$", path)
        state = os.path.join(path, "trainer_state.json")
        if match and os.path.exists(state):
            step = int(match.group(1))
            if best is None or step > best[0]:
                best = (step, state)
    return best[1] if best else None


def bert_series(state_path):
    """The evaluation history in that file, merged by step.

    The Trainer writes one entry per evaluation and the breakdown callback adds a
    second at the same step carrying the masked-position losses, so an entry is not an
    evaluation point. Merging by step is what makes eval_loss and eval_loss_mask two
    columns of one row instead of two rows.
    """
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    points = {}
    for entry in state.get("log_history", []):
        metrics = {k: v for k, v in entry.items()
                   if k.startswith("eval_loss") and isinstance(v, (int, float))}
        if not metrics or "step" not in entry:
            continue
        points.setdefault(int(entry["step"]), {}).update(metrics)
    return state, points


def collapse_stop(log_dir, arm):
    """Did the collapse detector end this arm, and on what value?

    Returns ``None`` when no log of this arm carries the stop line, otherwise the
    value of the last warning before it. The newest checkpoint cannot answer this:
    save-on-improve writes nothing once a run turns, so the history in that file can
    end before the collapse it was stopped for.
    """
    if not log_dir:
        return None
    stop = None
    for path in sorted(glob.glob(os.path.join(log_dir, f"{job_stem(arm)}-s*.out"))):
        with open(path, errors="ignore") as handle:
            text = handle.read()
        if COLLAPSE_STOP not in text:
            continue
        hits = ABOVE_THRESHOLD.findall(text)
        stop = {"log": os.path.basename(path),
                "value": float(hits[-1][0]) if hits else None,
                "threshold": float(hits[-1][1]) if hits else None}
    return stop


def job_stem(arm):
    """The job name the submitter built out of the inventory's own columns."""
    return f"mc-cmp-{arm['arch']}-{arm['size']}-b{arm['batch']}-lr{arm['lr_slug']}"


def rows_for_bert(arms, runs_dir, log_dir=None):
    """One curve row per step and metric, and one summary row per arm."""
    curves, summary, missing = [], [], []
    for arm in arms:
        run_dir = os.path.join(runs_dir, arm["out_dir"])
        state_path = newest_state(run_dir)
        if not state_path:
            missing.append(arm)
            continue
        state, points = bert_series(state_path)
        judged = {s: m[BERT_JUDGED] for s, m in points.items() if BERT_JUDGED in m}
        if not judged:
            missing.append(arm)
            continue
        batch = int(arm["batch"])
        source = os.path.relpath(state_path, runs_dir)
        for step in sorted(points):
            for metric, value in sorted(points[step].items()):
                curves.append({
                    "arch": arm["arch"], "size": arm["size"], "batch": batch,
                    "lr": arm["lr"], "seed": SEED, "step": step,
                    "metric": metric, "value": f"{value:.6f}",
                    "rows_seen": step * batch,
                    "tokens_seen": step * batch * SEQUENCE_LENGTH,
                    "source": source,
                })
        best_val, best_step = min((v, s) for s, v in judged.items())
        final_step = max(judged)
        final_val = judged[final_step]
        beside = points.get(best_step, {}).get(BERT_BESIDE)
        beside_final = points.get(final_step, {}).get(BERT_BESIDE)
        stopped = collapse_stop(log_dir, arm)
        summary.append({
            "arch": arm["arch"], "size": arm["size"], "batch": batch, "lr": arm["lr"],
            "steps_planned": int(arm["steps"]),
            # The loop is ahead of its last evaluation, so the step the run has
            # reached and the step the series ends at are two different numbers.
            "steps_reached": int(state.get("global_step") or final_step),
            "eval_points": len(judged),
            "judged_metric": arm["judge_metric"],
            "best_value": f"{best_val:.6f}", "best_step": best_step,
            "best_rows": best_step * batch,
            "best_tokens": best_step * batch * SEQUENCE_LENGTH,
            "best_fraction_of_budget": f"{best_step / int(arm['steps']):.4f}",
            "final_value": f"{final_val:.6f}", "final_step": final_step,
            "final_minus_best": f"{final_val - best_val:+.6f}",
            "beside_metric": BERT_BESIDE,
            "beside_at_best": "" if beside is None else f"{beside:.6f}",
            "beside_final": "" if beside_final is None else f"{beside_final:.6f}",
            # Either the arm's own last value is at or above the floor, or the
            # detector already ended it for staying above the configured line. The
            # second is the case the checkpoint history cannot show.
            "collapsed": ("yes" if (stopped or (BERT_FLOOR is not None
                                                and final_val >= BERT_FLOOR))
                          else ("" if BERT_FLOOR is None else "no")),
            "ended": ("collapse-stop" if stopped else
                      ("budget" if int(state.get("global_step") or final_step)
                       >= int(arm["steps"]) else "running")),
            "stopped_at_value": ("" if not stopped or stopped["value"] is None
                                 else f"{stopped['value']:.6f}"),
            "source": source if not stopped else f"{source}; {stopped['log']}",
        })
    return curves, summary, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--logs", help="the directory holding the run logs (gpt2)")
    ap.add_argument("--runs", help="the directory holding the run output directories "
                                   "(bert)")
    ap.add_argument("--arch", default="gpt2", choices=["gpt2", "bert"],
                    help="gpt2 reads the run logs, bert reads trainer_state.json")
    ap.add_argument("--curves-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args(argv)
    check_name(args.curves_out)
    check_name(args.summary_out)
    # Each architecture is read from the place that architecture writes, and asking
    # for one with the other's input is a mistake to refuse rather than to half-run.
    if args.arch == "gpt2" and not args.logs:
        ap.error("--logs is required for gpt2")
    if args.arch == "bert" and not args.runs:
        ap.error("--runs is required for bert")
    # For BERT the logs are optional but wanted: without them an arm the collapse
    # detector ended cannot be told from one still running, so say so rather than
    # report the weaker reading silently.
    if args.arch == "bert" and not args.logs:
        print("** no --logs: a collapse-stop will not be visible, only the last "
              "value the newest checkpoint happens to carry")

    arms = read_arms(args.arms, args.arch)
    if args.arch == "bert":
        curves, summary, missing = rows_for_bert(arms, args.runs, args.logs)
        where = "without a checkpoint to read"
    else:
        curves, summary, missing = rows_for(arms, args.logs)
        where = "without a log"
    print(f"=== {len(arms)} {args.arch} arm(s); {len(summary)} read, "
          f"{len(missing)} {where} ===")
    for arm in missing:
        print(f"  ** no evaluation point for {arm['size']} batch {arm['batch']} "
              f"lr {arm['lr']}")

    beside = "" if args.arch != "bert" else f" {BERT_BESIDE:>10s}"
    print(f"\n{'size':8s} {'batch':>5s} {'lr':>9s} {'reached':>8s} {'points':>6s} "
          f"{'best':>9s} {'at step':>8s} {'of budget':>9s} {'final':>9s} "
          f"{'final-best':>10s}{beside}")
    for row in summary:
        print(f"{row['size']:8s} {row['batch']:>5d} {row['lr']:>9s} "
              f"{row['steps_reached']:>8,} {row['eval_points']:>6d} "
              f"{row['best_value']:>9s} {row['best_step']:>8,} "
              f"{float(row['best_fraction_of_budget']) * 100:>8.1f}% "
              f"{row['final_value']:>9s} {row['final_minus_best']:>10s}"
              + (f" {row['beside_at_best']:>10s}" if args.arch == "bert" else "")
              + ("   COLLAPSED" if row["collapsed"] == "yes" else "")
              + ("  (collapse-stop at "
                 f"{row['stopped_at_value'] or '?'})" if row.get("ended") ==
                 "collapse-stop" else ""))

    # While arms are still running they are at different fractions of their budget,
    # and the lowest value so far is then a statement about who has run longest.
    if any(int(r["steps_reached"]) < int(r["steps_planned"]) for r in summary):
        print("\n** some arms have not reached the budget; the values above are "
              "where each arm has got to, not its result")

    print("\n=== lowest batch per size, with the gap to the runner-up ===")
    for size in ("small", "medium", "large", "xl"):
        # One batch can hold several rates (BERT runs two or three), and the batch's
        # value is its best arm. Ranking arms instead of batches printed the same
        # batch twice as its own runner-up.
        per_batch = {}
        for row in summary:
            if row["size"] != size:
                continue
            value = float(row["best_value"])
            if row["batch"] not in per_batch or value < per_batch[row["batch"]]:
                per_batch[row["batch"]] = value
        ranked = sorted((v, b) for b, v in per_batch.items())
        if len(ranked) < 2:
            continue
        (first, b1), (second, b2) = ranked[0], ranked[1]
        print(f"  {size:8s} batch {b1:>5d} at {first:.6f}, then batch {b2:>5d} at "
              f"{second:.6f} -- gap {second - first:.6f}")

    shared = {
        "generated": stamp("scripts/compounds_batch_sweep_curves.py"),
        "job": os.environ.get("SLURM_JOB_ID") or None,
    }
    read_from = ("the evaluation line each run wrote in its own log"
                 if args.arch == "gpt2" else
                 "the evaluation history in each run's newest checkpoint "
                 "(trainer_state.json), merged by step")
    write_tsv(args.curves_out, CURVE_COLUMNS, curves, dict(shared, **{
        "model": f"several (columns: size,batch,lr) -- the compounds batch-size grid's "
                 f"{args.arch} arms",
        "source": f"{read_from}; the arm's "
                  "identity comes from the derived inventory, not from the log",
        "note": [
            "one row per step and metric; rows_seen = step x batch and tokens_seen = "
            f"rows_seen x {SEQUENCE_LENGTH}",
            "the 2026-11-02 curve file's split column is replaced by a metric column: "
            "a value column needs one (column convention R12)",
            f"seed {SEED} on every arm",
        ],
    }))
    write_tsv(args.summary_out, SUMMARY_COLUMNS, summary, dict(shared, **{
        "model": f"several (columns: size,batch,lr) -- one row per {args.arch} arm",
        "source": "read off the same evaluation series as the curve file, in one pass",
        "note": [
            "best_value is the lowest value of the judged metric over the arm's own "
            "evaluation points; best_rows and best_tokens are where it sits",
            "final_value is the last evaluation point, which is the value at the "
            "budget only for an arm whose steps_reached equals steps_planned",
            "best_fraction_of_budget = best_step / steps_planned, and "
            "final_minus_best = final_value - best_value (2026-11-20 §5)",
            (f"collapsed = yes when the final value is at or above "
             f"{DEGENERATE_FLOOR}, the degenerate floor measured for this corpus"
             if args.arch == "gpt2" else
             f"collapsed = yes when the judged metric's last value is at or above "
             f"{BERT_FLOOR}, the unigram cross-entropy at the [MASK] positions "
             "measured on this build (2026-11-02, mlm_lookup_baselines.py)"),
            ("beside_metric is the series reported next to the judged one "
             f"({BERT_BESIDE} beside {BERT_JUDGED}, 2026-11-20 §2); beside_at_best "
             "and beside_final are its values at best_step and final_step"
             if args.arch == "bert" else
             "beside_metric is empty: nothing is reported beside val_loss here"),
            ("steps_reached is the newest checkpoint's global_step, so it lags the "
             "running loop; the loop's own position is in the progress TSV written "
             "by scripts/compounds_batch_sweep_progress.py"
             if args.arch == "bert" else
             "steps_reached is the last evaluation step in the run's log"),
        ],
    }))
    print(f"\n{len(curves)} curve rows -> {args.curves_out}")
    print(f"{len(summary)} arm rows -> {args.summary_out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
