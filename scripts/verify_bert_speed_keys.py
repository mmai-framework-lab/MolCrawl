"""Print the execution settings each BERT config actually resolves to.

main.py reads bf16, tf32, dataloader_num_workers and dataloader_pin_memory through
``globals().get(..., False)``, so a config that does not mention them runs with all
four off and the manifest has nothing to record. This executes each config the way
the trainer does and prints what it resolves to, alongside the batch numbers, so a
change to the input pipeline can be seen not to have moved anything else.

    python scripts/verify_bert_speed_keys.py <config.py> [<config.py> ...]
"""

from __future__ import annotations

import runpy
import sys

KEYS = ("bf16", "tf32", "dataloader_num_workers", "dataloader_pin_memory")


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    print(
        f"{'config':44s}{'bf16':>6}{'tf32':>6}{'work':>6}{'pin':>6}"
        f"{'mb':>6}{'accum':>7}{'eb(4gpu)':>10}{'declared':>10}{'seed':>7}"
    )
    bad = 0
    for p in paths:
        try:
            g = runpy.run_path(p, run_name="__main__")
        except Exception as e:  # noqa: BLE001
            print(f"{p.split('configs/')[-1]:44s}  FAILED TO EXECUTE: {type(e).__name__}: {e}")
            bad += 1
            continue
        mb = int(g.get("batch_size", 0) or 0)
        acc = int(g.get("gradient_accumulation_steps", 0) or 0)
        dec = g.get("expected_global_batch")
        eb = mb * acc * 4
        flag = "" if dec is None or int(dec) == eb else "  <- declared differs"
        if flag:
            bad += 1
        print(
            f"{p.split('configs/')[-1]:44s}"
            f"{str(g.get('bf16', False)):>6}{str(g.get('tf32', False)):>6}"
            f"{str(g.get('dataloader_num_workers', 0)):>6}"
            f"{str(g.get('dataloader_pin_memory', False)):>6}"
            f"{mb:>6}{acc:>7}{eb:>10}"
            f"{('-' if dec is None else int(dec)):>10}{str(g.get('seed', '-')):>7}{flag}"
        )
    print("\ndeclared '-' means the config states no expected_global_batch, so main.py's guard does not run for it.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
