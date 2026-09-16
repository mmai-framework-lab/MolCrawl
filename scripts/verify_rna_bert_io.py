"""Measure what the RNA BERT run actually reads, and whether the GPT-2 memmap fits it.

Three numbers decide how a throughput measurement should be read, and all three
have been taken from the GPT-2 side of this workstream rather than from the BERT
dataset itself:

  rows      how many rows the BERT train split holds. The config derives max_steps
            from 34,407,040, but that figure came from the packed GPT-2 blocks;
            dataset_info.json disagrees and reports the same number for train and
            valid, which is the "one total written into every split" defect seen in
            genome. So the count is taken from the dataset the trainer opens.
  epochs    how many passes max_steps amounts to. A short measurement window is
            only re-reading rows if an epoch is long compared with the window.
  window    how many rows one arm of the throughput harness touches, and how much
            of an epoch that is.

Also reports whether the flat uint16 memmap built for GPT-2 carries what the BERT
collator needs, so the answer is a file listing rather than a recollection.

    python scripts/verify_rna_bert_io.py --arrow-dir <dir> [--bin-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PiB"


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrow-dir", required=True)
    ap.add_argument("--bin-dir", default=None)
    ap.add_argument("--global-batch", type=int, default=2560)
    ap.add_argument("--max-steps", type=int, default=40320)
    ap.add_argument("--arm-steps", type=int, default=30)
    ap.add_argument("--config-rows", type=int, default=34407040,
                    help="the row count the config derived max_steps from")
    args = ap.parse_args()

    from datasets import load_from_disk

    arrow = Path(args.arrow_dir)
    print(f"=== Arrow dataset: {arrow}")
    ds = load_from_disk(str(arrow))
    splits = {k: v for k, v in ds.items()} if hasattr(ds, "items") else {"train": ds}

    rows = {}
    for name, d in splits.items():
        p = arrow / name
        nbytes = dir_bytes(p) if p.is_dir() else 0
        shards = len(list(p.glob("*.arrow"))) if p.is_dir() else 0
        rows[name] = d.num_rows
        print(f"  {name:6s} rows={d.num_rows:>12,}  on-disk={human(nbytes):>12}  "
              f"shards={shards:>4}  columns={d.column_names}")
        if d.num_rows:
            print(f"         bytes/row={nbytes / d.num_rows:,.0f}")

    train = rows.get("train", 0)
    print(f"\n=== epochs at global batch {args.global_batch:,}")
    print(f"  config assumed rows : {args.config_rows:,}")
    print(f"  measured train rows : {train:,}")
    if train:
        delta = train - args.config_rows
        print(f"  difference          : {delta:+,}  ({delta / args.config_rows * 100:+.1f} %)")
        per_epoch = train / args.global_batch
        print(f"  steps per epoch     : {per_epoch:,.1f}")
        print(f"  {args.max_steps:,} steps       : {args.max_steps / per_epoch:.2f} epochs")
        print(f"  config claimed      : {args.max_steps * args.global_batch / args.config_rows:.2f} epochs")

        win = args.arm_steps * args.global_batch
        print(f"\n=== one throughput arm ({args.arm_steps} steps)")
        print(f"  rows touched        : {win:,}")
        print(f"  share of one epoch  : {win / train * 100:.3f} %")
        tb = dir_bytes(arrow / "train") if (arrow / "train").is_dir() else 0
        if tb:
            print(f"  bytes touched       : {human(tb * win / train)}  (of {human(tb)})")
        print("  arms share a seed and a global batch, so every arm consumes the same")
        print("  first rows of the same permutation -- arm 2 onward re-read arm 1's rows.")

    if args.bin_dir:
        b = Path(args.bin_dir)
        print(f"\n=== GPT-2 memmap: {b}")
        if not b.is_dir():
            print("  (absent)")
            return 0
        for f in sorted(b.iterdir()):
            if f.is_file():
                print(f"  {f.name:28s} {human(f.stat().st_size):>12}")
        for meta in sorted(b.glob("*.json")):
            try:
                print(f"  --- {meta.name}: {json.dumps(json.load(meta.open()))[:300]}")
            except Exception as e:  # noqa: BLE001
                print(f"  --- {meta.name}: unreadable ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
