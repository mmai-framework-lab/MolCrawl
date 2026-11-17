#!/usr/bin/env python3
"""What a nanoGPT checkpoint says about the run that wrote it, without loading weights.

The counters that decide where a resumed run stops live in the checkpoint, not in the
log: ``iter_num``, ``best_val_loss``, ``best_val_step`` and ``early_stopping_counter``
(models/gpt2/train.py). A run resumed from one of these starts with that counter, so
"patience evaluations from the resume point" and "patience evaluations of this run's
lineage" are different stopping steps, and only the file says which one applies.

``mmap=True`` keeps the tensors on disk: the xl checkpoints are 19 GB and none of that
is needed to read six scalars.

    python scripts/molnl_gpt2_checkpoint_state.py <checkpoint dir or training_state.bin>...
"""

from __future__ import annotations

import os
import sys

import torch

KEYS = ("iter_num", "best_val_loss", "best_val_step", "early_stopping_counter")


def path_of(arg):
    if os.path.isdir(arg):
        for name in ("training_state.bin", "ckpt.pt"):
            candidate = os.path.join(arg, name)
            if os.path.exists(candidate):
                return candidate
        raise SystemExit(f"{arg}: no training_state.bin or ckpt.pt inside")
    return arg


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for arg in argv:
        path = path_of(arg)
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        values = {k: state.get(k) for k in KEYS}
        config = state.get("config") or {}
        for k in ("max_iters", "lr_decay_iters", "eval_interval", "early_stopping",
                  "early_stopping_patience", "learning_rate", "min_lr"):
            if k in config:
                values[k] = config[k]
        print(path)
        for k, v in values.items():
            print(f"    {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
