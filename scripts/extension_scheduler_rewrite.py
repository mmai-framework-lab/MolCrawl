#!/usr/bin/env python3
"""Re-point a checkpoint's scheduler.pt at the schedule an extension is about to run.

An extension past a decayed schedule needs a constant rate, and the checkpoint's
scheduler.pt carries the original one: ``LambdaLR.load_state_dict`` does
``self.__dict__.update(state_dict)``, so restoring it brings back ``base_lrs`` (the
original peak) and ``last_epoch`` (the step it stopped at), and the extension's
``learning_rate`` and ``warmup_steps`` are both discarded.

Deleting the file fixes the rate and costs the optimizer: transformers 4.45.1 guards the
optimizer load and the scheduler load with one condition on scheduler.pt's existence
(``Trainer._load_optimizer_and_scheduler``), so without it the Adam moments are not
restored either -- silently. Measured on 2026-11-18 (job 363888) and again on the molnl
BERT extensions, whose checkpoints carry an Adam step counted from the resume rather than
from the run's start.

Re-pointing the file keeps both: ``base_lrs`` becomes the extension's rate and
``last_epoch`` becomes 0, so restoring it applies the extension's own schedule from its
first step -- warmup included -- and optimizer.pt loads because the file is there.

    python scripts/extension_scheduler_rewrite.py <scheduler.pt> <learning rate> [--check]

``--check`` only reports whether the file is already re-pointed, and writes nothing. The
contract is the one workflows/bert-grid.sbatch and workflows/molnl-bert-ladder.sbatch
check for before they accept an extension: ``last_epoch == 0`` and every entry of
``base_lrs`` equal to the config's rate.
"""

from __future__ import annotations

import argparse

import torch


def read(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state, {"base_lrs": [float(x) for x in state.get("base_lrs") or []],
                   "last_epoch": state.get("last_epoch"),
                   "_step_count": state.get("_step_count"),
                   "_last_lr": [float(x) for x in state.get("_last_lr") or []]}


def is_rewritten(summary, lr):
    base = summary["base_lrs"]
    return bool(base) and summary["last_epoch"] == 0 and all(
        abs(b - lr) <= abs(lr) * 1e-9 for b in base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("learning_rate", type=float)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    state, before = read(args.path)
    print(f"{args.path}")
    print(f"    before: base_lrs {before['base_lrs']} last_epoch {before['last_epoch']} "
          f"_step_count {before['_step_count']} _last_lr {before['_last_lr']}")
    if args.check:
        print(f"    re-pointed at {args.learning_rate}: "
              f"{'yes' if is_rewritten(before, args.learning_rate) else 'no'}")
        return 0

    state["base_lrs"] = [args.learning_rate] * len(state["base_lrs"])
    state["last_epoch"] = 0
    # A scheduler that has stepped once, standing at the first step of this schedule:
    # the same shape a fresh LambdaLR has after the Trainer builds it.
    state["_step_count"] = 1
    state["_last_lr"] = [0.0] * len(state["base_lrs"])
    if "_is_initial" in state:
        state["_is_initial"] = True
    torch.save(state, args.path)

    _, after = read(args.path)
    print(f"    after : base_lrs {after['base_lrs']} last_epoch {after['last_epoch']} "
          f"_step_count {after['_step_count']} _last_lr {after['_last_lr']}")
    if not is_rewritten(after, args.learning_rate):
        raise SystemExit("the file was written but does not read back as re-pointed")
    print("    ok: the launcher's gate will accept this checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
