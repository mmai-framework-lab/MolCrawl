#!/usr/bin/env python3
"""Did a resumed HF run restore the optimizer's moments, or start them from zero?

``Trainer._load_optimizer_and_scheduler`` loads optimizer.pt only when scheduler.pt is
present in the same checkpoint::

    if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):

An extension that removes scheduler.pt -- which is how a constant rate is forced past a
decayed schedule -- therefore also loses the Adam moments, silently: nothing is logged.
Adam's own per-parameter step counter says which happened. A checkpoint written shortly
after a resume carries a step count near the steps since the resume if the moments were
reset, and near the global step if they were restored.

    python scripts/verify_resume_optimizer_state.py <checkpoint dir>...

Reads optimizer.pt with mmap, so the tensors stay on disk.
"""

from __future__ import annotations

import os
import sys

import torch


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for arg in argv:
        path = arg if arg.endswith(".pt") else os.path.join(arg, "optimizer.pt")
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        state = blob.get("state", {})
        steps = []
        zeros = 0
        for i, (_key, entry) in enumerate(sorted(state.items())[:5]):
            step = entry.get("step")
            steps.append(float(step) if step is not None else None)
            exp_avg = entry.get("exp_avg")
            if exp_avg is not None and i == 0:
                zeros = int((exp_avg == 0).sum()) if exp_avg.numel() < 10**8 else -1
        print(path)
        print(f"    parameters with state : {len(state)}")
        print(f"    adam step (first five): {steps}")
        if zeros >= 0:
            print(f"    zero entries in the first exp_avg: {zeros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
