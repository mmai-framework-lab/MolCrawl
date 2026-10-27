#!/usr/bin/env python3
"""Count the items in a corpus, as against the windows the model trains on.

Directive `all-corpus-size-order-2026-10-27.md` §1 asks for `n_items` -- sequences,
molecules, records, cells -- beside `n_tokens`. The window counts were already
measured (`scripts/audit_data_parameters.py`); the item counts were not, and two of
the figures in use are wrong by more than a factor of two because of it.

Reads metadata, not content. A parquet file's row count is in its footer and an
arrow file's is in its schema block, so counting rows over a 76 GiB tree touches a
few KB per file. Nothing here loads a token column.

Usage:
    python scripts/audit_corpus_item_counts.py \
        --parquet-dir rna=<dir with per-chunk .parquet> \
        --arrow train=<file.arrow> --arrow valid=<file.arrow> \
        --report <out.json>

Both options repeat. `--parquet-dir` sums every `*.parquet` under the directory;
`--arrow` counts one file and names it by the label given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def parquet_rows(directory):
    """Row count per file, and the group count, from the footers alone."""
    import pyarrow.parquet as pq

    files = sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".parquet")
    )
    total = 0
    per_file = {}
    unreadable = {}
    for path in files:
        try:
            total_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            unreadable[os.path.basename(path)] = f"{type(exc).__name__}: {exc}"
            continue
        per_file[os.path.basename(path)] = total_rows
        total += total_rows
    return {"files": len(files), "rows": total, "unreadable": unreadable}, per_file


def group_by_prefix(per_file):
    """Sum per leading `<name>.` component, which is the tissue for the rna chunks."""
    groups = defaultdict(int)
    for name, rows in per_file.items():
        m = re.match(r"^(.*)\.\d+-\d+\.parquet$", name)
        groups[m.group(1) if m else name] += rows
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


def arrow_rows(path):
    """Row count of one arrow file, or of a save_to_disk directory of them.

    ``datasets.save_to_disk`` writes a *directory* named ``<split>.arrow`` holding
    ``data-NNNNN-of-NNNNN.arrow`` shards. The first submission of this audit passed
    that directory where a file was expected and stopped. Both spellings are
    accepted now, because which one a modality has is not predictable from the name.

    ``dataset_info.json`` sits beside the shards and is deliberately not read: it is
    written once per build and copied into every split directory, so for these
    builds all three splits report the train count. Reading it returns one number
    three times. The shards are counted instead.
    """
    import pyarrow as pa

    if os.path.isdir(path):
        shards = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".arrow")
        )
        if not shards:
            raise ValueError(f"no .arrow shards under {path}")
        total = 0
        per_shard = {}
        for shard in shards:
            counted = arrow_rows(shard)
            per_shard[os.path.basename(shard)] = counted["rows"]
            total += counted["rows"]
        return {"rows": total, "format": "save_to_disk-dir",
                "shards": len(shards), "per_shard": per_shard}

    with pa.memory_map(path, "rb") as source:
        try:
            reader = pa.ipc.open_file(source)
            return {"rows": reader.num_record_batches and sum(
                reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
            ), "format": "ipc-file", "batches": reader.num_record_batches}
        except pa.ArrowInvalid:
            pass
    with pa.memory_map(path, "rb") as source:
        reader = pa.ipc.open_stream(source)
        rows = 0
        batches = 0
        for batch in reader:
            rows += batch.num_rows
            batches += 1
        return {"rows": rows, "format": "ipc-stream", "batches": batches}


def _pairs(values, flag):
    out = []
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"{flag} wants LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        out.append((label, path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", action="append", metavar="LABEL=DIR")
    ap.add_argument("--arrow", action="append", metavar="LABEL=FILE")
    ap.add_argument("--report", required=True)
    ap.add_argument("--per-file", action="store_true",
                    help="also record every file's own count, not just the totals")
    args = ap.parse_args(argv)

    if not args.parquet_dir and not args.arrow:
        raise SystemExit("nothing to count: pass --parquet-dir and/or --arrow")

    result = {"parquet": {}, "arrow": {}}

    for label, directory in _pairs(args.parquet_dir, "--parquet-dir"):
        if not os.path.isdir(directory):
            raise SystemExit(f"{label}: not a directory: {directory}")
        print(f"=== {label}: {directory}", flush=True)
        summary, per_file = parquet_rows(directory)
        groups = group_by_prefix(per_file)
        entry = {"dir": directory, **summary, "groups": groups}
        if args.per_file:
            entry["per_file"] = per_file
        result["parquet"][label] = entry
        print(f"  {summary['files']:,} files, {summary['rows']:,} rows "
              f"({len(groups)} groups)", flush=True)
        if summary["unreadable"]:
            print(f"  unreadable: {len(summary['unreadable'])}", flush=True)
        for name, rows in list(groups.items())[:5]:
            print(f"    {name:36s} {rows:>12,}", flush=True)

    for label, path in _pairs(args.arrow, "--arrow"):
        if not os.path.exists(path):
            raise SystemExit(f"{label}: no such path: {path}")
        print(f"=== {label}: {path}", flush=True)
        try:
            entry = arrow_rows(path)
        except Exception as exc:  # noqa: BLE001
            entry = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  unreadable: {entry['error']}", flush=True)
        else:
            parts = entry.get("batches", entry.get("shards"))
            print(f"  {entry['rows']:,} rows ({entry['format']}, {parts} parts)",
                  flush=True)
        entry["path"] = path
        result["arrow"][label] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
