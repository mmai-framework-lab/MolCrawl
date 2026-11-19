"""Choose the checkpoints the existing 2,560 arms are measured again at, and collect them.

The 2026-11-19c directive §11 asks for five points spread along each adopted compounds
BERT arm's curve, each carrying three numbers: the value the existing run recorded
(positions redrawn at every evaluation), and the fixed-mask eval_loss and
eval_loss_mask measured again under the grid's evaluation.

Which five is a choice, so the choice is a stated rule rather than a hand-picked set:

    the earliest checkpoint still on disk
    the one nearest step 15,000   -- where the first segment ended
    the one nearest step 30,000   -- where the second segment ended
    the adopted checkpoint        -- the one the reported number came from
    the last, step 45,000

An arm whose rule produces fewer than five distinct points is filled from the midpoint
of its widest remaining gap, which keeps the five spread rather than clustered.

An adopted arm is one run in three segments, so the checkpoints are gathered from all
three directories; retention has kept about ten in each.

    compounds_reval_points.py --runs-root <...> --out rt1-...tsv
    compounds_reval_points.py --runs-root <...> --out rt1-...tsv --collect <json dir>

Read-only over the runs: it opens each surviving checkpoint's trainer_state.json and
writes one TSV. It launches nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: The adopted arm of each size, as the three directories it was run in.
ARMS = {
    "small": "compounds-small-lr1e3-seed42",
    "medium": "compounds-medium-lr2p8e4-seed42",
    "large": "compounds-large-lr1p5e4-seed42",
    "xl": "compounds-xl-lr4p5e5-seed42",
}
SEGMENTS = ["", "-ext30k", "-ext45k"]

#: The metric the existing arms were judged and kept on. The new grid judges on
#: eval_loss (§7); both are reported here, so the two series can be compared on either
#: without the metric and the evaluation changing at the same time.
ADOPTED_METRIC = "eval_loss_mask"
METRICS = ["eval_loss", "eval_loss_mask"]

#: The anchors the rule names.
ANCHORS = [15_000, 30_000]
POINTS = 5

COLUMNS = ["arch", "size", "run", "segment", "step", "chosen_as",
           "existing_eval_loss", "existing_eval_loss_mask",
           "fixed_eval_loss", "fixed_eval_loss_mask",
           "checkpoint", "source"]


def checkpoints(runs_root: str, size: str):
    """Every surviving checkpoint of this arm, as (step, segment, path)."""
    found = []
    for segment in SEGMENTS:
        directory = os.path.join(runs_root, ARMS[size] + segment)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            stem, _, tail = name.partition("-")
            if stem != "checkpoint" or not tail.isdigit():
                continue
            found.append((int(tail), segment or "-base", os.path.join(directory, name)))
    return sorted(found)


def recorded(path: str, step: int):
    """The evaluation this checkpoint's own state recorded at its own step."""
    state = os.path.join(path, "trainer_state.json")
    if not os.path.exists(state):
        return {}
    with open(state, encoding="utf-8") as handle:
        history = json.load(handle).get("log_history", [])
    values = {}
    for entry in history:
        if entry.get("step") != step:
            continue
        for metric in METRICS:
            if metric in entry:
                values[metric] = entry[metric]
    return values


def choose(found, judged):
    """The five points the rule names, as {step: reason}."""
    steps = [step for step, _, _ in found]
    chosen = {}

    def take(step, reason):
        chosen.setdefault(step, reason)

    take(steps[0], "earliest surviving")
    for anchor in ANCHORS:
        take(min(steps, key=lambda s: (abs(s - anchor), s)), f"nearest {anchor:,}")
    ranked = [(value, step) for step, value in judged.items() if value is not None]
    if ranked:
        take(min(ranked)[1], f"adopted (lowest {ADOPTED_METRIC})")
    take(steps[-1], "last")

    # Fill from the widest gap, so five points stay spread rather than clustered.
    while len(chosen) < POINTS and len(chosen) < len(steps):
        ordered = sorted(chosen)
        gaps = [(ordered[i + 1] - ordered[i], ordered[i], ordered[i + 1])
                for i in range(len(ordered) - 1)]
        if not gaps:
            break
        # Two gaps of the same width are the ordinary case here -- the segments are
        # 15,000 steps each -- so the tie goes to the earlier one and the choice does
        # not depend on how the list happened to be built.
        _, low, high = max(gaps, key=lambda gap: (gap[0], -gap[1]))
        middle = (low + high) / 2
        candidates = [s for s in steps if s not in chosen and low < s < high]
        if not candidates:
            remaining = [s for s in steps if s not in chosen]
            if not remaining:
                break
            take(remaining[0], "filling the spread")
            continue
        take(min(candidates, key=lambda s: (abs(s - middle), s)), "widest gap")
    return chosen


def collected(directory: str):
    """The fixed-mask measurements already written, keyed by checkpoint path."""
    out = {}
    if not directory or not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as handle:
            record = json.load(handle)
        key = os.path.realpath(str(record.get("checkpoint", "")))
        metrics = record.get("metrics", {})
        out[key] = {m: metrics.get(m) for m in METRICS}
    return out


def rows_for(runs_root: str, measured) -> list:
    rows = []
    for size in ARMS:
        found = checkpoints(runs_root, size)
        if not found:
            print(f"  ** {size}: no checkpoint found under {runs_root}")
            continue
        existing = {step: recorded(path, step) for step, _, path in found}
        judged = {step: values.get(ADOPTED_METRIC) for step, values in existing.items()}
        chosen = choose(found, judged)
        print(f"  {size:6s}: {len(found):2d} surviving "
              f"({found[0][0]:,}-{found[-1][0]:,}), chose "
              f"{', '.join(f'{s:,}' for s in sorted(chosen))}")
        for step, segment, path in found:
            if step not in chosen:
                continue
            fixed = measured.get(os.path.realpath(path), {})
            rows.append({
                "arch": "bert", "size": size, "run": ARMS[size],
                "segment": segment, "step": step, "chosen_as": chosen[step],
                "existing_eval_loss": existing[step].get("eval_loss"),
                "existing_eval_loss_mask": existing[step].get("eval_loss_mask"),
                "fixed_eval_loss": fixed.get("eval_loss"),
                "fixed_eval_loss_mask": fixed.get("eval_loss_mask"),
                "checkpoint": os.path.join(ARMS[size] + (
                    "" if segment == "-base" else segment),
                    os.path.basename(path)),
                "source": os.path.join(os.path.basename(path), "trainer_state.json"),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True,
                    help="the directory holding the arms' output directories")
    ap.add_argument("--collect", default="",
                    help="a directory of eval-only JSON records to fill the "
                         "fixed-mask columns from")
    ap.add_argument("--out", required=True,
                    help="the base name begins with rt1- (the column convention)")
    args = ap.parse_args(argv)
    check_name(args.out)

    measured = collected(args.collect)
    if args.collect:
        print(f"=== {len(measured)} fixed-mask record(s) from {args.collect} ===")

    print("=== the five points per size ===")
    rows = rows_for(args.runs_root, measured)
    if not rows:
        raise SystemExit("no checkpoint was found; check --runs-root")

    filled = sum(1 for r in rows if r["fixed_eval_loss"] is not None)
    print(f"\n{len(rows)} points, {filled} measured under the fixed mask, "
          f"{len(rows) - filled} still to measure")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "several (columns: size) -- the adopted compounds BERT arms, "
                 "re-measured for the 2026-11-19c directive §11",
        "source": "each point's own checkpoint: trainer_state.json for the existing "
                  "value, and the eval-only record for the fixed-mask values",
        "generated": stamp("scripts/compounds_reval_points.py"),
        "note": [
            "five points per size: earliest surviving, nearest 15,000, nearest "
            "30,000, the adopted checkpoint, and the last; a gap is filled from its "
            "midpoint when the rule yields fewer than five",
            f"the adopted checkpoint is the lowest {ADOPTED_METRIC} among the "
            f"surviving ones, which is what the existing arms were judged on",
            "existing_* were measured with the masked positions redrawn at every "
            "evaluation; fixed_* under fixed_eval_mask with seed 42, 4 processes, "
            "evaluation micro-batch 32, 10,000 subset rows drawn with seed 42",
            "an empty fixed_* cell is a point not yet measured",
        ],
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"{len(rows)} points -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
