"""Time the BERT data read alone: Arrow against memmap, no GPU.

The 18.8 s/step figure is a whole step -- read, forward, backward. The question
underneath it is whether the read is what costs, and the read can be measured
without a GPU at all, which is what this does.

Two things the earlier five-arm harness got wrong are fixed here by construction:

  disjoint rows   Every arm shares seed 106 and global batch 2,560, so all five
                  arms drew the same first 76,800 rows -- 600 MiB, which sits in
                  page cache, so arm 1 paid the cold read and the rest did not.
                  Here each arm reads its OWN window, far apart from the others,
                  so no arm is warmed by the arm before it.
  four readers    One rank reading alone is not the production case. RNA GPT-2's
                  Arrow cost was 11 ms for an 8 KB row with four ranks reading
                  concurrently and much less alone, so the arms fork four
                  processes onto disjoint windows the way four ranks contend.
  random order    The Trainer shuffles, so a step's 640 rows per rank are 640
                  scattered 8 KB reads across 262.6 GiB, not one run of pages.
                  Reading the window in order instead measures a case the run
                  never has, and flatters Arrow by whatever readahead supplies.
                  --access random is the production case; sequential is kept
                  alongside it to show how much of the cost is the ordering.

Masking is deliberately excluded. DataCollatorForLanguageModeling is CPU compute
on rows already in memory; mixing it in would blur the one quantity being
separated here.

    python scripts/probe_bert_read_path.py --arrow-dir <dir> --bin-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np


def _indices(start: int, count: int, span: int, access: str, seed: int) -> list:
    if access == "sequential":
        return list(range(start, start + count))
    # Scattered over a window far larger than the rows drawn, which is what a
    # shuffled sampler does to a 34 M-row split.
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(span, size=count, replace=False).astype(np.int64) + start)


def _arrow_window(arrow_dir: str, start: int, count: int, batch: int, q,
                  span: int, access: str, seed: int) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(arrow_dir)["train"]
    idx = _indices(start, count, span, access, seed)
    t0 = time.perf_counter()
    rows = 0
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        # __getitems__ is what the torch DataLoader calls on a HF dataset; going
        # through it keeps this on the same code path the Trainer uses.
        out = ds[chunk]["input_ids"]
        rows += len(out)
    q.put((time.perf_counter() - t0, rows))


def _memmap_window(bin_dir: str, start: int, count: int, batch: int, q,
                   span: int, access: str, seed: int) -> None:
    meta = json.loads((Path(bin_dir) / "train.json").read_text())
    arr = np.memmap(Path(bin_dir) / "train.bin", dtype=np.uint16, mode="r",
                    shape=(int(meta["rows"]), int(meta["block"])))
    idx = _indices(start, count, span, access, seed)
    t0 = time.perf_counter()
    rows = 0
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        block = np.asarray(arr[chunk], dtype=np.int64)
        rows += block.shape[0]
    q.put((time.perf_counter() - t0, rows))


def run_arm(name: str, target, arg0: str, base: int, procs: int, rows_each: int,
            batch: int, bytes_per_row: int, span: int, access: str) -> dict:
    q: mp.Queue = mp.Queue()
    workers = []
    for r in range(procs):
        # Each reader gets its own stretch, and the stretches of one arm sit far
        # from every other arm's, so nothing here is served by a warm cache.
        start = base + r * rows_each
        p = mp.Process(target=target,
                       args=(arg0, start, rows_each, batch, q, span, access, 1000 + r))
        p.start()
        workers.append(p)
    res = [q.get() for _ in workers]
    for p in workers:
        p.join()

    wall = max(t for t, _ in res)
    rows = sum(n for _, n in res)
    per_batch_ms = (wall / (rows_each / batch)) * 1000
    mib = rows * bytes_per_row / 1024 / 1024
    return {
        "arm": name,
        "wall_s": wall,
        "rows": rows,
        "per_batch_ms": per_batch_ms,
        "mib_per_s": mib / wall,
        "slowest_reader_s": wall,
        "spread": max(t for t, _ in res) / min(t for t, _ in res),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrow-dir", required=True)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--procs", type=int, default=4, help="readers, i.e. ranks")
    ap.add_argument("--rows-each", type=int, default=25600,
                    help="rows per reader; 640 is one rank's rows for one step")
    ap.add_argument("--batch", type=int, default=8, help="micro-batch")
    ap.add_argument("--micro-per-step", type=int, default=80,
                    help="gradient_accumulation_steps: micro-batches per step")
    ap.add_argument("--access", choices=("random", "sequential", "both"),
                    default="both", help="random is what the shuffled sampler does")
    ap.add_argument("--total-rows", type=int, default=34_407_040,
                    help="rows in the train split; only spaces the windows apart")
    args = ap.parse_args()

    # Windows are spaced widely so no two arms share pages.
    stride = args.total_rows // 8
    plan = [
        ("arrow", _arrow_window, args.arrow_dir, 8_196),
        ("memmap", _memmap_window, args.bin_dir, 2_048),
    ]

    span = stride  # each reader scatters across a window this wide
    print(f"readers={args.procs}  rows/reader={args.rows_each:,}  "
          f"micro-batch={args.batch}  micro/step={args.micro_per_step}")
    print(f"each arm+access reads its own window; windows are {stride:,} rows apart")
    print(f"random access scatters over {span:,} rows per reader\n")

    modes = ("random", "sequential") if args.access == "both" else (args.access,)
    slot = 0
    for access in modes:
        print(f"--- {access} access ---")
        got = []
        for name, fn, arg0, bpr in plan:
            base = stride * (slot + 1)
            slot += 1
            r = run_arm(name, fn, arg0, base, args.procs, args.rows_each,
                        args.batch, bpr, span // args.procs, access)
            r["step_read_s"] = r["per_batch_ms"] / 1000 * args.micro_per_step
            r["bytes_per_row"] = bpr
            got.append(r)
            print(f"{name:8s} rows={r['rows']:>9,}  {r['per_batch_ms']:7.2f} ms/micro-batch  "
                  f"{r['mib_per_s']:8.1f} MiB/s  reader spread {r['spread']:.2f}x")
            print(f"         -> data read alone is {r['step_read_s']:6.2f} s of one "
                  f"{args.batch}x{args.micro_per_step} step")
        a, m = got
        print(f"  arrow / memmap : {a['per_batch_ms'] / m['per_batch_ms']:.2f}x per micro-batch")
        print(f"  step read time : {a['step_read_s']:.2f} s -> {m['step_read_s']:.2f} s "
              f"(saves {a['step_read_s'] - m['step_read_s']:.2f} s)\n")
    print(f"\nhost={os.uname().nodename}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
