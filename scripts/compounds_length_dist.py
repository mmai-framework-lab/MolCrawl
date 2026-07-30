"""compounds 実分子長 (pad 除外) 分布集計。

Computes for organix13 + chembl train split:
- statistics: mean, median, p95, p99, max
- histogram bins: 0-32, 33-64, 65-128, 129-256, 257+

pad_id = 0 (BERT convention, verified via input_ids trailing zeros).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
LSD = Path("/lustre/home/matsubara/learning_source_20260520_uncapped")

TARGETS = {
    "organix13": LSD / "compounds" / "organix13" / "compounds" / "training_ready_hf_dataset",
    "chembl":    LSD / "compounds" / "chembl" / "training_ready_hf_dataset",
}
BINS = [(0, 32), (33, 64), (65, 128), (129, 256), (257, 100_000)]
PAD_ID = 0


def real_length_of_row(ids: list[int]) -> int:
    """Count of non-pad tokens in the row (contiguous pad tail assumed)."""
    n = 0
    for x in ids:
        if x != PAD_ID:
            n += 1
    return n


def summarise(label: str, dataset_path: Path) -> dict:
    print(f"\n=== {label}: {dataset_path.relative_to(LSD.parent)} ===", flush=True)
    ds = load_from_disk(str(dataset_path))["train"]
    n_rows = len(ds)
    seq_len = len(ds[0]["input_ids"])
    print(f"  rows={n_rows:,}  fixed seq_len={seq_len}  (pad_id={PAD_ID})", flush=True)

    lengths = np.zeros(n_rows, dtype=np.int32)
    batch = 5000
    idx = 0
    for start in range(0, n_rows, batch):
        end = min(start + batch, n_rows)
        arr = np.asarray(ds[start:end]["input_ids"], dtype=np.int64)
        # count of non-pad per row
        lens = (arr != PAD_ID).sum(axis=1)
        lengths[idx:idx + len(lens)] = lens
        idx += len(lens)
        if start // batch % 50 == 0:
            print(f"    processed {end:,} / {n_rows:,}", flush=True)

    mean = float(lengths.mean())
    median = float(np.median(lengths))
    p95 = float(np.percentile(lengths, 95))
    p99 = float(np.percentile(lengths, 99))
    max_ = int(lengths.max())
    min_ = int(lengths.min())
    print(f"  mean={mean:.2f}  median={median:.1f}  p95={p95:.1f}  p99={p99:.1f}  max={max_}  min={min_}", flush=True)

    histogram = {}
    for lo, hi in BINS:
        label_bin = f"{lo}-{hi if hi != 100_000 else '256+'}"
        count = int(((lengths >= lo) & (lengths <= hi)).sum())
        pct = 100.0 * count / n_rows
        histogram[label_bin] = {"count": count, "pct": pct}
        print(f"    {label_bin:>10}: {count:>10,}  ({pct:5.1f} %)", flush=True)

    return {
        "label": label,
        "path": str(dataset_path),
        "rows": n_rows,
        "seq_len_fixed": seq_len,
        "pad_id": PAD_ID,
        "mean": mean,
        "median": median,
        "p95": p95,
        "p99": p99,
        "max": max_,
        "min": min_,
        "histogram": histogram,
    }


def main() -> int:
    results = []
    for label, path in TARGETS.items():
        results.append(summarise(label, path))

    out_dir = REPO / "tmp" / "compounds_padding_check_20260707"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "length_distribution.json").write_text(json.dumps(results, indent=2))

    with (out_dir / "length_distribution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "bin", "count", "pct", "rows_total", "mean", "median", "p95", "p99", "max"])
        for r in results:
            for bin_label, stat in r["histogram"].items():
                w.writerow([r["label"], bin_label, stat["count"], f"{stat['pct']:.2f}",
                            r["rows"], f"{r['mean']:.2f}", r["median"], r["p95"], r["p99"], r["max"]])
    print(f"\n[done] wrote {out_dir}/length_distribution.{{json,csv}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
