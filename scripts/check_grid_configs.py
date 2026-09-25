#!/usr/bin/env python3
"""Resolve a set of grid configs and check they differ only where they are meant to.

A grid config imports its base rather than restating it, so what a file says is not what
a run gets: grep cannot see the schedule, and a typo in one import is a point that
silently trains under different settings. This resolves each file the way main.py will
and prints what came out, then fails if two points of one grid disagree on anything but
the learning rate.

    python scripts/check_grid_configs.py --expect max_steps=24000 --expect warmup_steps=2400 \
        molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_*_24k_lr*.py
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

# What a point of a grid is allowed to differ in. Everything else the run reads has to
# match across the points, or the grid is not measuring the learning rate.
VARIES = ("learning_rate", "min_lr", "out_dir", "tensorboard_dir")
REPORT = ("max_steps", "warmup_steps", "learning_rate", "seed", "batch_size",
          "gradient_accumulation_steps", "expected_global_batch", "max_length",
          "document_masking", "bf16", "dataloader_num_workers", "dataloader_pin_memory",
          "save_steps", "log_interval", "fixed_eval_mask", "eval_mask_seed",
          "early_stopping", "save_on_improve")


def resolve(path):
    values = runpy.run_path(path)
    return {k: values.get(k) for k in REPORT}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                    help="a value every config must resolve to; repeatable")
    a = ap.parse_args()

    expected = dict(e.split("=", 1) for e in a.expect)
    resolved = {}
    for path in a.configs:
        try:
            resolved[path] = resolve(path)
        except Exception as exc:                      # noqa: BLE001 - report, don't raise
            print(f"FAILED to resolve {path}: {type(exc).__name__}: {exc}")
            return 1

    width = max(len(os.path.basename(p)) for p in resolved)
    print(f"{'config':<{width}}  " + "  ".join(k for k in REPORT))
    for path, values in resolved.items():
        print(f"{os.path.basename(path):<{width}}  "
              + "  ".join(str(values[k]) for k in REPORT))

    problems = []
    for path, values in resolved.items():
        for key, want in expected.items():
            if str(values.get(key)) != want:
                problems.append(f"{os.path.basename(path)}: {key} is {values.get(key)!r}, "
                                f"expected {want}")
    for key in REPORT:
        if key in VARIES:
            continue
        seen = {str(v[key]) for v in resolved.values()}
        if len(seen) > 1:
            problems.append(f"{key} differs across the grid: {sorted(seen)}")

    rates = [v["learning_rate"] for v in resolved.values()]
    if len(set(map(str, rates))) != len(rates):
        problems.append(f"two configs share a learning rate: {sorted(map(str, rates))}")

    print()
    for problem in problems:
        print(f"PROBLEM  {problem}")
    print(f"{len(resolved)} configs, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
