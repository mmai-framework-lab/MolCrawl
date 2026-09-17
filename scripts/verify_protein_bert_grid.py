"""Resolve the protein BERT base and grid configs and check what they will run.

Each grid config imports its base and overrides a handful of names, so reading the
file shows only the overrides. This executes all nine and prints what they resolve
to, and fails if two arms would write to the same directory -- the one mistake that
would silently merge two runs.

    MODEL_OUTPUT_ROOT=<dir> LEARNING_SOURCE_DIR=<protein root> \
        python scripts/verify_protein_bert_grid.py
"""

from __future__ import annotations

import glob
import os
import runpy
import sys


def main() -> int:
    if not os.environ.get("MODEL_OUTPUT_ROOT"):
        print("set MODEL_OUTPUT_ROOT: unset, model_path resolves under the input tree")
        return 2
    paths = sorted(
        p
        for p in glob.glob("molcrawl/tasks/pretrain/configs/protein_sequence/bert_*.py")
        if "esmopt" not in p and ("_lr" in p or p.endswith(("bert_small.py", "bert_medium.py", "bert_large.py")))
    )
    seen: dict[str, list[str]] = {}
    print(f"{'config':24s}{'max_steps':>10}{'warmup':>8}{'lr':>8}{'mb':>5}{'acc':>5}{'eb':>6}{'beta2':>7}  model_path")
    for p in paths:
        g = runpy.run_path(p, run_name="__main__")
        seen.setdefault(g["model_path"], []).append(os.path.basename(p))
        print(
            f"{os.path.basename(p):24s}{g['max_steps']:>10}{g['warmup_steps']:>8}"
            f"{g['learning_rate']:>8.0e}{g['batch_size']:>5}{g['gradient_accumulation_steps']:>5}"
            f"{g['batch_size'] * g['gradient_accumulation_steps'] * 4:>6}{g.get('adam_beta2')!s:>7}"
            f"  {g['model_path']}"
        )
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"\n{len(paths)} configs, {len(seen)} distinct model_path")
    for k, v in dup.items():
        print(f"  SHARED {k}: {', '.join(v)}")
    return 1 if dup or len(paths) != 12 else 0


if __name__ == "__main__":
    sys.exit(main())
