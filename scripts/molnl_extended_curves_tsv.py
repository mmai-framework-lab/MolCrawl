#!/usr/bin/env python3
"""The seven mol_nl runs as one curve each, original schedule and extension together.

The 2026-11-17 order §2 asks for one file holding all seven, each as a single curve from
step 1 through the end of its extension, with the training side and the validation side
both in it, the rate that was in effect at each step, and a column saying which part of
the run a row belongs to.

Three things are worth stating about the columns.

``lr`` is per step, not per run. The order asks for the rate actually in effect, and a
run's two parts ran at different rates: the original schedule decays from the arm's peak,
the extension holds a tenth of it. BERT's rates are read from the log the run wrote.
GPT-2's are computed with train.py's own ``get_lr`` from the schedule the run recorded in
its manifest, because the nanoGPT CSV has no rate column; the provenance says so, and the
same formula covers both parts because ``lr_decay_iters`` did not move.

``segment`` is ``original`` for the rows the original run measured and ``extension`` for
the rows the continuation measured. For GPT-2 that leaves the resume step appearing twice
-- the original run evaluated at 1,500 and the continuation evaluated there again before
training on -- which is two measurements of one step by two runs, not a duplicate row.
BERT has no such pair: its continuation's first evaluation is at 24,100.

``value`` with a ``metric`` column is the exception the column convention allows: the
metric differs from row to row (BERT's eval_loss_mask over the fifth of positions it
masked, GPT-2's val_loss over every token it predicted, and the two training-side
counterparts).

``--cross-check`` reads the original run's own files and fails if the original segment
disagrees with them anywhere. That is what makes the restored history in the extension's
trainer_state.json answerable: it has to equal what the original run wrote.

    python scripts/molnl_extended_curves_tsv.py --bert-root <dir> --gpt2-root <dir> \
        --out <tsv> [--cross-check] [--allow-running]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime

COLUMNS = ("arch", "size", "seed", "step", "segment", "split", "value", "metric", "lr", "source")
#: arch, size, the run being continued, the continuation, the step it resumed from
RUNS = (
    ("bert", "small", "bert_small_24k_lr1e3", "bert_small_24k_lr1e3_ext", 24000),
    ("bert", "medium", "bert_medium_24k_lr3e4", "bert_medium_24k_lr3e4_ext", 24000),
    ("bert", "large", "bert_large_24k_lr1p7e4", "bert_large_24k_lr1p7e4_ext", 24000),
    ("gpt2", "small", "gpt2_small_1500_lr1p2e3", "gpt2_small_1500_lr1p2e3_ext", 1500),
    ("gpt2", "medium", "gpt2_medium_1500_lr6e4", "gpt2_medium_1500_lr6e4_ext", 1500),
    ("gpt2", "large", "gpt2_large_1500_lr3e4", "gpt2_large_1500_lr3e4_ext", 1500),
    ("gpt2", "xl", "gpt2_xl_1500_lr3e4", "gpt2_xl_1500_lr3e4_ext", 1500),
)
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}
ARCH_ORDER = {"bert": 0, "gpt2": 1}
SPLIT_ORDER = {"train": 0, "valid": 1}


def manifest(run_dir):
    path = os.path.join(run_dir, "run_manifest.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def seed_of(run_dir):
    seed = manifest(run_dir).get("seed")
    return seed.get("seed") if isinstance(seed, dict) else seed


def newest_state(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        raise SystemExit(f"{run_dir}: no checkpoint to read trainer_state.json from")
    path = os.path.join(cks[-1], "trainer_state.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: missing -- the newest checkpoint was not finished")
    return json.load(open(path, encoding="utf-8")), os.path.basename(cks[-1])


def bert_rows(root, size, original, extension, resume, cross_check):
    run = os.path.join(root, extension)
    state, newest = newest_state(run)
    source = f"{extension}/{newest}/trainer_state.json"
    seed = seed_of(run)
    rates = {e["step"]: e["learning_rate"] for e in state["log_history"] if "learning_rate" in e}
    rows, seen = [], {}
    for entry in state["log_history"]:
        step = entry.get("step")
        segment = "original" if step <= resume else "extension"
        rate = rates.get(step)
        rate_cell = "" if rate is None else f"{rate:.10g}"
        if "loss" in entry:
            rows.append(("bert", size, seed, step, segment, "train",
                         f"{entry['loss']:.6f}", "train_loss_mask", rate_cell, source))
        if "eval_loss_mask" in entry:
            rows.append(("bert", size, seed, step, segment, "valid",
                         f"{entry['eval_loss_mask']:.6f}", "eval_loss_mask", rate_cell, source))
            seen[step] = entry["eval_loss_mask"]
    problems = []
    if cross_check:
        before, _ = newest_state(os.path.join(root, original))
        for entry in before["log_history"]:
            if "eval_loss_mask" not in entry:
                continue
            step, want = entry["step"], entry["eval_loss_mask"]
            got = seen.get(step)
            if got is None:
                problems.append(f"{extension}: the original run's step {step} is missing")
            elif abs(got - want) > 1e-12:
                problems.append(f"{extension}: step {step} reads {got!r}, the original run wrote {want!r}")
    # best_model_checkpoint is an absolute path, and for an arm the extension has not
    # improved on it still points into the ORIGINAL run's directory -- large's step 23,800.
    # Name the directory in that case, because the two are different places.
    best_path = (state.get("best_model_checkpoint") or "").rstrip("/")
    best = os.path.basename(best_path) or newest
    if best_path and os.path.basename(os.path.dirname(best_path)) != extension:
        best = f"{os.path.basename(os.path.dirname(best_path))}/{best}"
    return rows, best, state["global_step"], state["max_steps"], problems


def get_lr(step, warmup, decay_iters, peak, floor):
    """train.py's own schedule, so the column cannot drift from what the run ran at."""
    if step < warmup:
        return peak * step / warmup
    if step > decay_iters:
        return floor
    ratio = (step - warmup) / (decay_iters - warmup)
    return floor + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (peak - floor)


def read_csv(path):
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8"), skipinitialspace=True):
        if row.get("iter", "").strip().isdigit():
            out[int(row["iter"])] = (float(row["train_loss"]), float(row["val_loss"]))
    return out


def gpt2_rows(root, size, original, extension, resume, cross_check):
    run = os.path.join(root, extension)
    before_dir = os.path.join(root, original)
    copied = {os.path.basename(p) for p in glob.glob(os.path.join(before_dir, "logging_*.csv"))}
    schedule = manifest(run).get("schedule", {})
    for key in ("learning_rate", "min_lr", "warmup_iters", "lr_decay_iters"):
        if schedule.get(key) is None:
            raise SystemExit(f"{run}: run_manifest.json has no schedule.{key}, so the rate "
                             f"at each step cannot be stated")
    seed = seed_of(run)
    rows, problems, by_segment = [], [], {}
    for path in sorted(glob.glob(os.path.join(run, "logging_*.csv"))):
        name = os.path.basename(path)
        segment = "original" if name in copied else "extension"
        series = read_csv(path)
        by_segment.setdefault(segment, {}).update(series)
        for step, (train, val) in sorted(series.items()):
            rate = get_lr(step, schedule["warmup_iters"], schedule["lr_decay_iters"],
                          schedule["learning_rate"], schedule["min_lr"])
            source = f"{extension}/{name}"
            rows.append(("gpt2", size, seed, step, segment, "train",
                         f"{train:.6f}", "train_loss", f"{rate:.10g}", source))
            rows.append(("gpt2", size, seed, step, segment, "valid",
                         f"{val:.6f}", "val_loss", f"{rate:.10g}", source))
    if cross_check:
        for name in sorted(copied):
            want = read_csv(os.path.join(before_dir, name))
            got = by_segment.get("original", {})
            for step, pair in want.items():
                if step not in got:
                    problems.append(f"{extension}: the original run's iteration {step} is missing")
                elif got[step] != pair:
                    problems.append(f"{extension}: iteration {step} reads {got[step]}, "
                                    f"the original run wrote {pair}")
    # The adopted checkpoint is the step with the lowest validation loss over the whole
    # lineage, which for an arm the extension never improved on sits in the original run.
    # Name the directory it is in, because those are two different directories.
    measured = [(step, pair[1]) for series in by_segment.values() for step, pair in series.items()]
    best_step = min(measured, key=lambda t: t[1])[0] if measured else None
    reached = max(step for step, _ in measured) if measured else None
    cks = sorted(int(os.path.basename(d).split("-")[1])
                 for d in glob.glob(os.path.join(run, "checkpoint-*")))
    best = (f"checkpoint-{best_step}" if best_step in cks
            else f"{original}/checkpoint-{best_step}")
    return rows, best, reached, None, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--allow-running", action="store_true",
                    help="write the file even where a run has not stopped yet")
    args = ap.parse_args()

    rows, provenance, problems = [], [], []
    for arch, size, original, extension, resume in RUNS:
        root = args.bert_root if arch == "bert" else args.gpt2_root
        reader = bert_rows if arch == "bert" else gpt2_rows
        got, best, reached, ceiling, trouble = reader(root, size, original, extension,
                                                      resume, args.cross_check)
        rows += got
        problems += trouble
        if arch == "bert" and ceiling and reached >= ceiling and not args.allow_running:
            problems.append(f"{extension}: reached {reached} of {ceiling} -- it ran to the "
                            f"ceiling, which the order asks to be told apart from a stop")
        note = f"{arch} {size}: {extension} reached {reached}"
        if ceiling:
            note += f" of {ceiling}"
        provenance.append(f"# checkpoint: {note}, adopted {best}")
    if problems:
        for p in problems:
            print("DISAGREEMENT:", p)
        raise SystemExit(f"{len(problems)} disagreement(s) with what the runs wrote; nothing written")

    rows.sort(key=lambda r: (ARCH_ORDER[r[0]], SIZE_ORDER[r[1]], r[3],
                             0 if r[4] == "original" else 1, SPLIT_ORDER[r[5]]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang, the seven adopted runs continued past their "
                 "schedule (2026-11-17 order, item 2)\n")
        fh.write("# source: each run's own files: an HF run's trainer_state.json at its newest "
                 "checkpoint, a nanoGPT run's logging_*.csv\n")
        fh.write(f"# generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        for line in provenance:
            fh.write(line + "\n")
        fh.write("# note: lr is the rate in effect at that step, not the arm's peak. BERT's is "
                 "read from the log the run wrote; GPT-2's is computed with train.py's get_lr "
                 "from the schedule in its manifest, because the nanoGPT CSV has no rate "
                 "column. rt1-molnl-loss-curves-20261112.tsv used this column for the run's "
                 "peak instead\n")
        fh.write("# note: segment is original for the rows the original run measured and "
                 "extension for the rows the continuation measured. GPT-2's resume step 1500 "
                 "therefore carries one row per run per split: two measurements of one step by "
                 "two runs\n")
        fh.write("# note: the extension holds a constant tenth of the arm's peak; BERT ramps to "
                 "it over 500 steps because the original schedule ends at 0.0, GPT-2 is already "
                 "there because lr_decay_iters stayed at 1500\n")
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} rows")
    for line in provenance:
        print("   ", line[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
