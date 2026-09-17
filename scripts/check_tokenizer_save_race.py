"""Reproduce, or rule out, the custom-tokenizer save/load race between runs.

rna and genome BERT configs save their tokenizer to a fixed directory and read it
straight back. Runs started together therefore write into the same files while
others read them, and a reader that opens a file mid-write fails. Two of three
arms launched together died exactly there (117320, 117321) -- with the directory
already present from earlier runs, so writing it once in advance does not help.

This executes a config many times from several processes at once and counts how
many executions fail to load the tokenizer. Run it against the code before and
after a change to the save, on a private copy of the tree, not the shared one.

    python scripts/check_tokenizer_save_race.py --config <cfg> [--procs 8] [--iters 6]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import runpy
import traceback


def _worker(config: str, iters: int, q) -> None:
    ok = bad = 0
    first = None
    for _ in range(iters):
        try:
            runpy.run_path(config, run_name="__main__")
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad += 1
            if first is None:
                first = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    q.put((ok, bad, first))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    q: mp.Queue = mp.Queue()
    ps = [mp.Process(target=_worker, args=(args.config, args.iters, q)) for _ in range(args.procs)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()

    ok = sum(r[0] for r in res)
    bad = sum(r[1] for r in res)
    errors = sorted({r[2] for r in res if r[2]})
    print(f"config {args.config}")
    print(f"{args.procs} processes x {args.iters} executions = {ok + bad}")
    print(f"loaded {ok}, failed {bad}")
    for e in errors:
        print(f"  e.g. {e}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
