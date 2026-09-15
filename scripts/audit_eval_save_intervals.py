#!/usr/bin/env python3
"""Report the eval/save interval each pretrain config actually uses.

Produces the table §4 of the 2026-09-15 interval directive asks for: per
modality and per architecture, the eval interval, the save interval, and how
many eval points the schedule yields.

Values are read as source text, not by importing the configs: importing them
builds tokenizers and opens datasets, which needs the training environment and
the data trees. Assignments inside ``if`` blocks are therefore reported with
every value they can take, in file order, joined by ``|``.
"""

import argparse
import os
import re
import sys

KEYS = (
    "log_interval",
    "eval_interval",
    "save_steps",
    "save_checkpoint_steps",
    "always_save_checkpoint",
    "keep_legacy_ckpt",
    "max_checkpoints",
    "save_total_limit",
    "checkpoint_keep_best",
    "checkpoint_keep_latest",
    "max_steps",
    "max_iters",
    "batch_size",
    "per_device_eval_batch_size",
)

# The eval interval is spelled differently by the two trainers: HF Trainer takes
# eval_steps=log_interval (models/bert/main.py), nanoGPT takes eval_interval.
EVAL_KEY = {"bert": "log_interval", "roberta": "log_interval",
            "gpt2": "eval_interval", "llama": "eval_interval"}
SAVE_KEY = {"bert": "save_steps", "roberta": "save_steps",
            "gpt2": "save_checkpoint_steps", "llama": "save_checkpoint_steps"}
STEP_KEY = {"bert": "max_steps", "roberta": "max_steps",
            "gpt2": "max_iters", "llama": "max_iters"}


def read_values(path):
    """Every top-level-name assignment in the file, keyed by name."""
    text = open(path, encoding="utf-8").read()
    found = {}
    for key in KEYS:
        hits = re.findall(rf"^[ \t]*{key}\s*(?::[^=]+)?=\s*([^\n#]+)", text, re.M)
        if hits:
            found[key] = "|".join(h.strip() for h in hits)
    return found


def arch_of(filename):
    return filename.split("_", 1)[0]


def eval_points(steps, interval):
    """How many eval points the schedule yields, or None if not derivable."""
    try:
        return int(steps) // int(interval)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="molcrawl/tasks/pretrain/configs",
                    help="config root, relative to the repo or absolute")
    ap.add_argument("--arch", action="append",
                    help="limit to these architectures (repeatable)")
    args = ap.parse_args(argv)

    root = args.configs
    if not os.path.isdir(root):
        print(f"no such config root: {root}", file=sys.stderr)
        return 2

    header = f"{'config':52s} {'eval':>10s} {'save':>10s} {'steps':>10s} {'evals':>7s}"
    for modality in sorted(os.listdir(root)):
        mdir = os.path.join(root, modality)
        if not os.path.isdir(mdir):
            continue
        print(f"\n=== {modality}")
        print(header)
        for name in sorted(os.listdir(mdir)):
            if not name.endswith(".py") or name == "__init__.py":
                continue
            arch = arch_of(name)
            if arch not in EVAL_KEY:
                continue
            if args.arch and arch not in args.arch:
                continue
            vals = read_values(os.path.join(mdir, name))
            ev = vals.get(EVAL_KEY[arch], "-")
            sv = vals.get(SAVE_KEY[arch], "-")
            st = vals.get(STEP_KEY[arch], "-")
            n = eval_points(st, ev)
            extra = []
            for k in ("always_save_checkpoint", "keep_legacy_ckpt", "max_checkpoints"):
                if k in vals:
                    extra.append(f"{k}={vals[k]}")
            # Only the HF trainers take a separate eval micro-batch; nanoGPT
            # evaluates at batch_size. A mismatch is what makes an eval point
            # cost far more than the training steps around it.
            if (arch in ("bert", "roberta")
                    and vals.get("batch_size") != vals.get("per_device_eval_batch_size")):
                extra.append(f"train_mb={vals.get('batch_size')} "
                             f"eval_mb={vals.get('per_device_eval_batch_size')}")
            print(f"{name:52s} {ev:>10s} {sv:>10s} {st:>10s} "
                  f"{(str(n) if n is not None else '-'):>7s}")
            if extra:
                print(f"{'':52s}   {'  '.join(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
