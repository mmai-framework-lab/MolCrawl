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

# What a point of a grid is allowed to differ in. Everything else the run reads has to
# match across the points, or the grid is not measuring the learning rate.
VARIES = ("learning_rate", "min_lr", "out_dir", "tensorboard_dir")
# Both frameworks, because both have grids: HF names a schedule in steps, nanoGPT in
# iterations, and a key one of them does not have resolves to None for every point --
# which is a value they agree on, so it does not trip the comparison below.
REPORT = ("max_steps", "max_iters", "lr_decay_iters", "warmup_steps", "warmup_iters",
          "learning_rate", "min_lr", "seed", "batch_size", "gradient_accumulation_steps",
          "expected_global_batch", "max_length", "block_size", "document_masking", "bf16",
          "dtype", "dataloader_num_workers", "dataloader_pin_memory", "save_steps",
          "log_interval", "eval_interval", "eval_sequences", "fixed_eval_mask",
          "eval_mask_seed", "early_stopping", "save_on_improve")


def resolve(path, extra=()):
    values = runpy.run_path(path)
    return {k: values.get(k) for k in tuple(REPORT) + tuple(extra)}


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
            resolved[path] = resolve(path, expected)
        except Exception as exc:                      # noqa: BLE001 - report, don't raise
            print(f"FAILED to resolve {path}: {type(exc).__name__}: {exc}")
            return 1

    # A key every point resolves to None is one this framework does not have; printing a
    # column of None for it buries the ones that matter.
    columns = [k for k in resolved[next(iter(resolved))]
               if any(v[k] is not None for v in resolved.values())]
    width = max(len(os.path.basename(p)) for p in resolved)
    print(f"{'config':<{width}}  " + "  ".join(columns))
    for path, values in resolved.items():
        print(f"{os.path.basename(path):<{width}}  "
              + "  ".join(str(values[k]) for k in columns))

    problems = []
    for path, values in resolved.items():
        for key, want in expected.items():
            if str(values.get(key)) != want:
                problems.append(f"{os.path.basename(path)}: {key} is {values.get(key)!r}, "
                                f"expected {want}")
    for key in resolved[next(iter(resolved))]:
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
