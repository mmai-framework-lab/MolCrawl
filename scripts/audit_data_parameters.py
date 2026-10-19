"""Measure each modality's splits, row length and token total from the arrow itself.

``dataset_info.json`` is written once per build and copied into every split
directory, so for some builds all three splits report the same number -- reading it
is how a stale figure gets quoted. This loads each split and counts rows, then reads
one row to get the block length, so the numbers come from the data.

Token totals are rows x block length, which is the packed total including any
separator and padding. For a build that pads, that is not the number of real tokens;
the script says which it is by reporting the length it found rather than assuming one.
"""

from __future__ import annotations

import argparse
import json
import os

SPLITS = ("train", "valid", "test")


def measure(path):
    """Rows per split and the row length, read from the arrow."""
    from datasets import load_from_disk

    ds = load_from_disk(path)
    out = {"path": os.path.abspath(path), "splits": {}, "row_length": None}
    if hasattr(ds, "keys"):
        for name in ds:
            out["splits"][name] = int(ds[name].num_rows)
        first = next(iter(ds))
        col = "input_ids" if "input_ids" in ds[first].column_names else ds[first].column_names[0]
        out["column"] = col
        if ds[first].num_rows:
            out["row_length"] = len(ds[first][0][col])
    else:
        out["splits"]["(single)"] = int(ds.num_rows)
        col = "input_ids" if "input_ids" in ds.column_names else ds.column_names[0]
        out["column"] = col
        if ds.num_rows:
            out["row_length"] = len(ds[0][col])
    rows = sum(out["splits"].values())
    out["rows_total"] = rows
    out["tokens_rows_times_length"] = rows * out["row_length"] if out["row_length"] else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    results = {}
    for spec in args.dataset:
        name, path = spec.split("=", 1)
        print(f"=== {name} ===", flush=True)
        if not os.path.isdir(path):
            print("  ディレクトリが無い", flush=True)
            results[name] = {"path": path, "error": "missing"}
            continue
        try:
            r = measure(path)
        except Exception as exc:                    # noqa: BLE001 - recorded, not raised
            print(f"  読めない: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"path": path, "error": repr(exc)}
            continue
        results[name] = r
        for s, n in r["splits"].items():
            print(f"  {s:8s} {n:>12,}", flush=True)
        print(f"  合計 {r['rows_total']:,} 行 × 行長 {r['row_length']} "
              f"= {r['tokens_rows_times_length']:,}  列 {r['column']}", flush=True)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n記録 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
