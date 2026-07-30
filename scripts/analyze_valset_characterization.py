"""Val / Train set characterization for the 9 Evo2 subsets (BERT + GPT-2 splits).

Implements `valset-characterization-instructions-v3.md`:
  - 100 bp non-overlapping windows
  - 6 metrics per window: Shannon entropy, N content, GC content,
    homopolymer rate, 4-mer diversity, low-entropy block rate (1.0 / 1.2)
  - Per-window CSV, summary CSV, text report
  - val split: all rows used; train split: 100k rows random sample (seed=42)

Diagnostic purpose:
  1. Identify the cause of the GPT-2 eukaryote val_loss anomaly (~0.25)
  2. Record subset characteristics for downstream interpretation (ClinVar etc.)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_from_disk


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
TOKENIZER_IDS = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}  # special token >= 5
NUCLEOTIDE_IDS = {0, 1, 2, 3, 4}  # incl. N
NON_N_IDS = {0, 1, 2, 3}
GC_IDS = {2, 3}
N_ID = 4

WINDOW_SIZE = 100
HOMOPOLYMER_MIN_RUN = 5
KMER_K = 4
KMER_THEO_MAX = 4 ** KMER_K  # 256
ENTROPY_THRESHOLDS = (1.0, 1.2)

TRAIN_SAMPLE_SIZE = 100_000
TRAIN_SAMPLE_SEED = 42

SUBSETS = [
    "mammal_centered",
    "eukaryote_matched_random_seed1",
    "eukaryote_matched_random_seed2",
    "eukaryote_matched_random_seed3",
    "eukaryote_matched_random_seed4",
    "eukaryote_matched_random_seed5",
    "global_random_seed1",
    "global_random_seed2",
    "global_random_seed3",
]
MODEL_TYPES = ["bert", "gpt2"]
SPLITS = ["valid", "train"]   # actual key is "valid"; report uses "val" for spec parity


# ──────────────────────────────────────────────────────────────────────────
# Per-window metrics
# ──────────────────────────────────────────────────────────────────────────
def window_metrics(window: np.ndarray) -> dict:
    """Compute the 6 metrics for one 100-token window.

    Returns dict with keys: entropy, n_fraction, gc_content, homopolymer_rate,
    kmer4_diversity. Any metric that cannot be computed (all-N window for
    entropy; <10 valid 4-mers for kmer4_diversity) is set to np.nan.
    """
    n_total = window.size  # always WINDOW_SIZE
    # N content
    n_count = int(np.sum(window == N_ID))
    n_fraction = n_count / n_total

    # Mask out non-nucleotide ids defensively (window should only contain 0..4)
    non_n_mask = window != N_ID
    non_n = window[non_n_mask]
    n_non_n = non_n.size

    # GC content (on non-N positions)
    if n_non_n == 0:
        gc_content = np.nan
        entropy = np.nan
    else:
        gc_count = int(np.sum(np.isin(non_n, list(GC_IDS))))
        gc_content = gc_count / n_non_n

        # Shannon entropy on A/T/G/C only
        counts = np.bincount(non_n, minlength=4)[:4]  # length 4 for ids 0..3
        p = counts / n_non_n
        nz = p > 0
        entropy = float(-np.sum(p[nz] * np.log2(p[nz])))

    # Homopolymer rate: bases inside runs of length >= HOMOPOLYMER_MIN_RUN
    # Operates on all ids (0..4); a long stretch of N also counts as low-complexity.
    homopolymer_rate = _homopolymer_rate(window)

    # 4-mer diversity: unique 4-mers (N-excluded) / 256
    kmer4_diversity = _kmer4_diversity(window)

    return {
        "entropy": entropy,
        "n_fraction": n_fraction,
        "gc_content": gc_content,
        "homopolymer_rate": homopolymer_rate,
        "kmer4_diversity": kmer4_diversity,
    }


def _homopolymer_rate(window: np.ndarray) -> float:
    """Fraction of bases that lie inside a run of length >= HOMOPOLYMER_MIN_RUN.

    Operates on the raw window (ids 0..4) so a run of Ns also counts.
    """
    if window.size == 0:
        return np.nan
    # Run lengths via diff
    change = np.concatenate(([1], np.diff(window) != 0, [1])).astype(np.int8)
    boundaries = np.flatnonzero(change)
    run_lens = np.diff(boundaries)
    long_runs = run_lens[run_lens >= HOMOPOLYMER_MIN_RUN]
    return float(np.sum(long_runs) / window.size)


def _kmer4_diversity(window: np.ndarray) -> float:
    """Unique 4-mers (N-excluded) / 256. Returns nan if < 10 valid 4-mers."""
    valid_mask = np.ones(window.size, dtype=bool)
    valid_mask &= window != N_ID
    # Slide window of length 4: a 4-mer is valid iff all 4 positions are non-N
    if window.size < KMER_K:
        return np.nan
    # vectorized: position i .. i+3 must all be non-N
    in_valid = valid_mask
    # build a boolean mask of valid 4-mer start positions
    kmer_valid = np.ones(window.size - KMER_K + 1, dtype=bool)
    for offset in range(KMER_K):
        kmer_valid &= in_valid[offset : window.size - KMER_K + 1 + offset]
    n_valid = int(np.sum(kmer_valid))
    if n_valid < 10:
        return np.nan
    # encode 4-mers as integers in base 4 (only ids 0..3, since N excluded)
    base = window.astype(np.int64)
    kmer_codes = (
        base[0 : window.size - 3] * 64
        + base[1 : window.size - 2] * 16
        + base[2 : window.size - 1] * 4
        + base[3 : window.size]
    )
    unique = np.unique(kmer_codes[kmer_valid])
    return float(unique.size / KMER_THEO_MAX)


# ──────────────────────────────────────────────────────────────────────────
# Row → windows
# ──────────────────────────────────────────────────────────────────────────
def row_to_windows(ids: np.ndarray) -> np.ndarray:
    """Strip special tokens (id >= 5) then split into 100-bp non-overlapping windows.

    Returns a (n_windows, WINDOW_SIZE) int8 array. Trailing < WINDOW_SIZE tokens are dropped.
    """
    # Filter to nucleotide ids only (0..4). Special tokens (CLS/SEP/PAD/UNK/MASK) are removed.
    ids = ids[ids < 5]
    n_windows = ids.size // WINDOW_SIZE
    if n_windows == 0:
        return np.empty((0, WINDOW_SIZE), dtype=np.int8)
    truncated = ids[: n_windows * WINDOW_SIZE]
    return truncated.reshape(n_windows, WINDOW_SIZE).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────
# Dataset iteration
# ──────────────────────────────────────────────────────────────────────────
def iter_rows(dataset, indices: Iterable[int]):
    """Yield input_ids np arrays for the given indices."""
    for i in indices:
        row = dataset[int(i)]
        yield np.asarray(row["input_ids"], dtype=np.int32)


def _select_indices(n_total: int, split: str, rng: np.random.Generator) -> list[int]:
    if split == "valid":
        return list(range(n_total))
    elif split == "train":
        if n_total <= TRAIN_SAMPLE_SIZE:
            return list(range(n_total))
        sampled = rng.choice(n_total, size=TRAIN_SAMPLE_SIZE, replace=False)
        return sorted(sampled.tolist())
    else:
        raise ValueError(split)


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────
class DatasetAccumulator:
    """Streaming aggregator for one (subset, model_type, split) combination."""

    def __init__(self) -> None:
        self.n_rows = 0
        # Window-level metrics arrays for percentile / mean (kept in memory; 1 dataset)
        self.entropies: list[float] = []
        self.n_fractions: list[float] = []
        self.gc_contents: list[float] = []
        self.homopolymer_rates: list[float] = []
        self.kmer4_diversities: list[float] = []
        self.n_windows = 0
        self.n_nan_entropy_windows = 0  # all-N windows
        self.n_low_entropy_windows = {th: 0 for th in ENTROPY_THRESHOLDS}

    def add_window(self, m: dict) -> None:
        self.n_windows += 1
        e = m["entropy"]
        if np.isnan(e):
            self.n_nan_entropy_windows += 1
        else:
            self.entropies.append(e)
            for th in ENTROPY_THRESHOLDS:
                if e < th:
                    self.n_low_entropy_windows[th] += 1
        if not np.isnan(m["n_fraction"]):
            self.n_fractions.append(m["n_fraction"])
        if not np.isnan(m["gc_content"]):
            self.gc_contents.append(m["gc_content"])
        if not np.isnan(m["homopolymer_rate"]):
            self.homopolymer_rates.append(m["homopolymer_rate"])
        if not np.isnan(m["kmer4_diversity"]):
            self.kmer4_diversities.append(m["kmer4_diversity"])

    def summary_row(self, subset: str, model_type: str, split: str) -> dict:
        entropies = np.asarray(self.entropies, dtype=np.float32)
        n_eff = entropies.size  # non-nan entropy window count
        rec = {
            "subset": subset,
            "model_type": model_type,
            "split": split,
            "n_rows": self.n_rows,
            "n_windows": self.n_windows,
            "n_nan_windows": self.n_nan_entropy_windows,
            "entropy_mean": float(np.mean(entropies)) if n_eff else np.nan,
            "entropy_median": float(np.median(entropies)) if n_eff else np.nan,
            "entropy_std": float(np.std(entropies)) if n_eff else np.nan,
            "entropy_p10": float(np.percentile(entropies, 10)) if n_eff else np.nan,
            "entropy_p90": float(np.percentile(entropies, 90)) if n_eff else np.nan,
        }
        for th in ENTROPY_THRESHOLDS:
            key = f"low_entropy_rate_{th}"
            rec[key] = self.n_low_entropy_windows[th] / n_eff if n_eff else np.nan
        rec["n_fraction_mean"] = (
            float(np.mean(self.n_fractions)) if self.n_fractions else np.nan
        )
        rec["gc_mean"] = float(np.mean(self.gc_contents)) if self.gc_contents else np.nan
        rec["homopolymer_rate_mean"] = (
            float(np.mean(self.homopolymer_rates)) if self.homopolymer_rates else np.nan
        )
        rec["kmer4_diversity_mean"] = (
            float(np.mean(self.kmer4_diversities)) if self.kmer4_diversities else np.nan
        )
        return rec


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def process_dataset(
    subset: str,
    model_type: str,
    split: str,
    base_dir: Path,
    per_window_writer: csv.DictWriter | None,
) -> dict:
    """Returns the summary row for one (subset, model_type, split) combination."""
    ds_path = base_dir / subset / f"training_ready_hf_dataset_{model_type}"
    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise KeyError(f"split {split!r} not found in {ds_path}; have {list(ds.keys())}")
    split_ds = ds[split]
    rng = np.random.default_rng(TRAIN_SAMPLE_SEED)
    indices = _select_indices(len(split_ds), split, rng)

    acc = DatasetAccumulator()
    acc.n_rows = len(indices)

    log_split = "val" if split == "valid" else split   # spec parity
    t0 = time.time()

    for row_idx, ids in zip(indices, iter_rows(split_ds, indices)):
        windows = row_to_windows(ids)
        for w_idx, win in enumerate(windows):
            m = window_metrics(win)
            acc.add_window(m)
            if per_window_writer is not None:
                per_window_writer.writerow(
                    {
                        "subset": subset,
                        "model_type": model_type,
                        "split": log_split,
                        "row_idx": row_idx,
                        "window_idx": w_idx,
                        "entropy": _fmt_float(m["entropy"]),
                        "n_fraction": _fmt_float(m["n_fraction"]),
                        "gc_content": _fmt_float(m["gc_content"]),
                        "homopolymer_rate": _fmt_float(m["homopolymer_rate"]),
                        "kmer4_diversity": _fmt_float(m["kmer4_diversity"]),
                    }
                )

    elapsed = time.time() - t0
    print(
        f"  [{subset}/{model_type}/{log_split}] rows={acc.n_rows:,} "
        f"windows={acc.n_windows:,} ({elapsed:.1f} s)",
        flush=True,
    )
    summary = acc.summary_row(subset, model_type, log_split)
    return summary


def _fmt_float(x) -> str:
    """Render NaN as empty, otherwise 6 decimal places (compact CSV)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.6f}"


# ──────────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────────
def render_report(summary_rows: list[dict]) -> str:
    """Build the text report with the 3 required comparison tables + 考察."""
    rows = summary_rows
    out: list[str] = []
    out.append("Val / Train set characterization report")
    out.append("=" * 78)
    out.append("")
    out.append(f"Settings: window={WINDOW_SIZE} bp non-overlapping,")
    out.append(f"          homopolymer_min_run={HOMOPOLYMER_MIN_RUN},")
    out.append(f"          kmer_k={KMER_K} (theoretical max {KMER_THEO_MAX}),")
    out.append(f"          low_entropy thresholds={list(ENTROPY_THRESHOLDS)},")
    out.append(f"          val split: all rows; train split: {TRAIN_SAMPLE_SIZE:,} rows random "
                f"(seed={TRAIN_SAMPLE_SEED}).")
    out.append("")

    # ── Table 1: entropy_mean (val) for BERT vs GPT-2 ──────────────────
    out.append("Table 1: entropy_mean on val split (BERT vs GPT-2)")
    out.append("-" * 78)
    out.append(f"{'subset':40s} {'BERT-val':>10s} {'GPT2-val':>10s}")
    for subset in SUBSETS:
        bert = _find(rows, subset, "bert", "val", "entropy_mean")
        gpt2 = _find(rows, subset, "gpt2", "val", "entropy_mean")
        out.append(f"{subset:40s} {_fmt_cell(bert):>10s} {_fmt_cell(gpt2):>10s}")
    out.append("")

    # ── Table 2: entropy_mean val vs train per (subset, model) ─────────
    out.append("Table 2: entropy_mean val vs train (split bias check)")
    out.append("-" * 78)
    out.append(
        f"{'subset':40s} {'BERT-val':>10s} {'BERT-tr':>10s} {'GPT2-val':>10s} {'GPT2-tr':>10s}"
    )
    for subset in SUBSETS:
        bv = _find(rows, subset, "bert", "val", "entropy_mean")
        bt = _find(rows, subset, "bert", "train", "entropy_mean")
        gv = _find(rows, subset, "gpt2", "val", "entropy_mean")
        gt = _find(rows, subset, "gpt2", "train", "entropy_mean")
        out.append(
            f"{subset:40s} {_fmt_cell(bv):>10s} {_fmt_cell(bt):>10s} "
            f"{_fmt_cell(gv):>10s} {_fmt_cell(gt):>10s}"
        )
    out.append("")

    # ── Table 3: low_entropy_rate_1.0 (val) ────────────────────────────
    out.append("Table 3: low_entropy_rate_1.0 on val split (BERT vs GPT-2)")
    out.append("-" * 78)
    out.append(f"{'subset':40s} {'BERT-val':>10s} {'GPT2-val':>10s}")
    for subset in SUBSETS:
        bert = _find(rows, subset, "bert", "val", "low_entropy_rate_1.0")
        gpt2 = _find(rows, subset, "gpt2", "val", "low_entropy_rate_1.0")
        out.append(f"{subset:40s} {_fmt_cell(bert):>10s} {_fmt_cell(gpt2):>10s}")
    out.append("")

    # ── 考察 ────────────────────────────────────────────────────────────
    out.append("考察(数値ベース)")
    out.append("-" * 78)
    bullets = _build_findings(rows)
    out.extend(bullets)
    out.append("")

    out.append("Per-window CSV:   valset_stats_per_window.csv")
    out.append("Summary CSV:      valset_stats_summary.csv")
    out.append("")
    return "\n".join(out)


def _find(rows: list[dict], subset: str, model: str, split: str, field: str):
    for r in rows:
        if r["subset"] == subset and r["model_type"] == model and r["split"] == split:
            return r.get(field)
    return None


def _fmt_cell(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _build_findings(rows: list[dict]) -> list[str]:
    """Plain-text data-driven observations (no value judgements)."""
    out = []

    # eukaryote vs others, entropy_mean (val) — GPT-2 side
    def avg(group: list[str], model: str, split: str, field: str):
        vals = [
            _find(rows, s, model, split, field)
            for s in group
        ]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    euk = [s for s in SUBSETS if s.startswith("eukaryote_matched_random_seed")]
    glb = [s for s in SUBSETS if s.startswith("global_random_seed")]
    mam = ["mammal_centered"]

    out.append(f"  - GPT-2 val の entropy_mean 平均:")
    out.append(f"      mammal:    {avg(mam, 'gpt2', 'val', 'entropy_mean'):.4f}")
    out.append(f"      eukaryote: {avg(euk, 'gpt2', 'val', 'entropy_mean'):.4f}")
    out.append(f"      global:    {avg(glb, 'gpt2', 'val', 'entropy_mean'):.4f}")
    out.append(f"  - GPT-2 val の low_entropy_rate_1.0 平均:")
    out.append(f"      mammal:    {avg(mam, 'gpt2', 'val', 'low_entropy_rate_1.0'):.4f}")
    out.append(f"      eukaryote: {avg(euk, 'gpt2', 'val', 'low_entropy_rate_1.0'):.4f}")
    out.append(f"      global:    {avg(glb, 'gpt2', 'val', 'low_entropy_rate_1.0'):.4f}")
    out.append("")

    out.append("  - val と train の entropy_mean 差(同じ subset 内、subset 偏り vs split 偏りの切り分け):")
    for s in SUBSETS:
        for model in MODEL_TYPES:
            v = _find(rows, s, model, "val", "entropy_mean")
            t = _find(rows, s, model, "train", "entropy_mean")
            if v is None or t is None or np.isnan(v) or np.isnan(t):
                continue
            diff = v - t
            tag = "↓" if diff < -0.05 else ("↑" if diff > 0.05 else "≈")
            out.append(f"      {s:40s} [{model:4s}] val={v:.4f} train={t:.4f} diff={diff:+.4f} {tag}")
    out.append("")

    out.append(
        "  解釈の指針: val ≈ train なら subset 全体の生物学的特性が原因、"
        "val < train なら split 時の偏りが原因の可能性が高い。"
    )
    return out


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence",
        help="LEARNING_SOURCE_DIR/genome_sequence",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: <base-dir>/analysis/valset_characterization/",
    )
    ap.add_argument(
        "--subsets",
        default=",".join(SUBSETS),
        help="comma-separated subset names (default: all 9)",
    )
    ap.add_argument(
        "--models",
        default=",".join(MODEL_TYPES),
        help="comma-separated model types (default: bert,gpt2)",
    )
    ap.add_argument(
        "--splits",
        default=",".join(SPLITS),
        help="comma-separated splits (default: valid,train)",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "analysis" / "valset_characterization"
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s for s in args.subsets.split(",") if s]
    models = [m for m in args.models.split(",") if m]
    splits = [s for s in args.splits.split(",") if s]

    per_window_path = out_dir / "valset_stats_per_window.csv"
    summary_path = out_dir / "valset_stats_summary.csv"
    report_path = out_dir / "valset_stats_report.txt"

    per_window_fields = [
        "subset", "model_type", "split", "row_idx", "window_idx",
        "entropy", "n_fraction", "gc_content", "homopolymer_rate", "kmer4_diversity",
    ]
    summary_fields = [
        "subset", "model_type", "split",
        "n_rows", "n_windows", "n_nan_windows",
        "entropy_mean", "entropy_median", "entropy_std",
        "entropy_p10", "entropy_p90",
        "low_entropy_rate_1.0", "low_entropy_rate_1.2",
        "n_fraction_mean", "gc_mean", "homopolymer_rate_mean", "kmer4_diversity_mean",
    ]

    summary_rows: list[dict] = []

    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  base_dir={base_dir}")
    print(f"  out_dir ={out_dir}")
    print(f"  {len(subsets)} subsets × {len(models)} models × {len(splits)} splits = "
          f"{len(subsets)*len(models)*len(splits)} datasets")
    print()

    with per_window_path.open("w", newline="") as pw_f:
        per_window_writer = csv.DictWriter(pw_f, fieldnames=per_window_fields)
        per_window_writer.writeheader()

        for subset in subsets:
            for model_type in models:
                for split in splits:
                    summary = process_dataset(
                        subset=subset,
                        model_type=model_type,
                        split=split,
                        base_dir=base_dir,
                        per_window_writer=per_window_writer,
                    )
                    summary_rows.append(summary)

    # Write summary CSV
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            row = {k: _fmt_float(v) if isinstance(v, float) else v for k, v in r.items()}
            w.writerow(row)

    # Write text report
    report_text = render_report(summary_rows)
    report_path.write_text(report_text)

    print()
    print(f"[done] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  per-window CSV: {per_window_path} ({per_window_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"  summary CSV:    {summary_path}")
    print(f"  report:         {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
