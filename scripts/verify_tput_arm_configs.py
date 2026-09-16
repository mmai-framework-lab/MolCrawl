"""Check that the throughput arms differ only in the four execution flags.

An arm is only a measurement of one thing if everything else resolved the same.
The grid configs were checked this way before the nine runs went out; the same
check applies here, and it has to execute the configs rather than grep them,
because each arm imports its base and a grep finds no max_steps in the file.

    python scripts/verify_tput_arm_configs.py base.py arm1.py arm2.py ...
"""

from __future__ import annotations

import runpy
import sys

# Everything that must match the base. The four execution flags are excluded on
# purpose -- those are what the arms vary.
SAME = (
    "max_steps", "warmup_steps", "log_interval", "save_steps", "batch_size",
    "gradient_accumulation_steps", "per_device_eval_batch_size", "max_length",
    "seed", "document_masking", "boundary_token_id", "dataset_dir",
    "expected_global_batch", "model_size", "learning_rate",
)
FLAGS = ("bf16", "tf32", "dataloader_num_workers", "dataloader_pin_memory")


def load(path: str) -> dict:
    return runpy.run_path(path, run_name="__main__")


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 2:
        print(__doc__)
        return 2
    base_path, arms = paths[0], paths[1:]
    base = load(base_path)

    print(f"base: {base_path}")
    for k in SAME:
        print(f"  {k:32s} {base.get(k, '(absent)')}")
    print(f"  {'effective global batch':32s} "
          f"{base.get('batch_size', 0) * base.get('gradient_accumulation_steps', 0) * 4}")
    print()

    bad = 0
    for p in arms:
        g = load(p)
        diffs = [k for k in SAME if g.get(k, "(absent)") != base.get(k, "(absent)")]
        flags = {k: g.get(k, "(absent)") for k in FLAGS}
        eb = g.get("batch_size", 0) * g.get("gradient_accumulation_steps", 0) * 4
        status = "OK" if not diffs and eb == 2560 else "DIFFERS"
        print(f"{status:8s} {p.split('/')[-1]:34s} "
              + "  ".join(f"{k.split('_')[-1] if k.startswith('dataloader') else k}={v}"
                          for k, v in flags.items())
              + f"  eb={eb}")
        for k in diffs:
            print(f"         {k}: base {base.get(k, '(absent)')} -> arm {g.get(k, '(absent)')}")
            bad += 1
        if eb != 2560:
            print(f"         effective global batch is {eb}, not 2,560")
            bad += 1
    print(f"\n{'all arms differ only in the four flags' if not bad else f'{bad} unintended difference(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
