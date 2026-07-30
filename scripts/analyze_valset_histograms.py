"""Generate the optional outputs from `valset-characterization-instructions-v3.md`:

  1. entropy_hist_val.png   — 9-subset overlapped entropy histogram on val split
  2. entropy_hist_train.png — same on train split
  3. appendix to report.txt — "集団特性まとめ" section (per-subset profile)

Reads the existing per_window CSV (the heavy 1.8 GiB file) and summary CSV
produced by ``analyze_valset_characterization.py`` and emits the matplotlib
figures + an appended block in ``valset_stats_report.txt``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

SUBSET_COLORS = {
    "mammal_centered": "#1f77b4",                      # blue
    "eukaryote_matched_random_seed1": "#2ca02c",       # green family
    "eukaryote_matched_random_seed2": "#5cba50",
    "eukaryote_matched_random_seed3": "#7fc97f",
    "eukaryote_matched_random_seed4": "#a8d8a8",
    "eukaryote_matched_random_seed5": "#d8edcc",
    "global_random_seed1": "#d62728",                  # red family
    "global_random_seed2": "#e85a5a",
    "global_random_seed3": "#f08c8c",
}

# Short labels for legend readability
SUBSET_LABELS = {
    "mammal_centered": "mammal",
    "eukaryote_matched_random_seed1": "eukaryote_s1",
    "eukaryote_matched_random_seed2": "eukaryote_s2",
    "eukaryote_matched_random_seed3": "eukaryote_s3",
    "eukaryote_matched_random_seed4": "eukaryote_s4",
    "eukaryote_matched_random_seed5": "eukaryote_s5",
    "global_random_seed1": "global_s1",
    "global_random_seed2": "global_s2",
    "global_random_seed3": "global_s3",
}


# ──────────────────────────────────────────────────────────────────────────
# Streaming load: entropies per (subset, model_type, split)
# ──────────────────────────────────────────────────────────────────────────
def load_entropies(
    per_window_csv: Path,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Stream the per_window CSV and bucket entropy values by (subset, model, split).

    GPT-2 has more samples per row but the histograms are comparable on the
    same x-axis (entropy bits), so we keep both BERT and GPT-2 buckets.
    """
    print(f"[load] reading {per_window_csv} (this may take ~30 s for 1.8 GiB)...", flush=True)
    t0 = time.time()
    buckets: dict[tuple[str, str, str], list[float]] = {}
    n_rows = 0
    with per_window_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy_s = row["entropy"]
            if not entropy_s:
                continue
            key = (row["subset"], row["model_type"], row["split"])
            buckets.setdefault(key, []).append(float(entropy_s))
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  ...{n_rows:>10,} rows ({elapsed:.0f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"[load] done: {n_rows:,} rows in {elapsed:.1f}s, {len(buckets)} buckets", flush=True)
    return {k: np.asarray(v, dtype=np.float32) for k, v in buckets.items()}


# ──────────────────────────────────────────────────────────────────────────
# Histogram generation
# ──────────────────────────────────────────────────────────────────────────
def plot_entropy_histogram(
    buckets: dict[tuple[str, str, str], np.ndarray],
    split: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    bins = np.linspace(0.0, 2.0, 41)  # 0.05 bit per bin

    for ax, model_type in zip(axes, ["bert", "gpt2"]):
        for subset in SUBSETS:
            key = (subset, model_type, split)
            data = buckets.get(key)
            if data is None or data.size == 0:
                continue
            color = SUBSET_COLORS[subset]
            label = SUBSET_LABELS[subset]
            ax.hist(
                data,
                bins=bins,
                histtype="step",
                density=True,
                alpha=0.7,
                linewidth=1.5,
                color=color,
                label=label,
            )
        ax.set_xlabel("Shannon entropy (bits, per 100 bp window)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_type.upper()} — {split} split")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 2.0)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    fig.suptitle(
        f"Per-window Shannon entropy distribution ({split} split, 9 subsets overlaid)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[hist] wrote {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Appendix: subset profile summary
# ──────────────────────────────────────────────────────────────────────────
def load_summary(summary_csv: Path) -> list[dict]:
    with summary_csv.open() as f:
        return list(csv.DictReader(f))


def render_profile_appendix(summary_rows: list[dict]) -> str:
    """Build the per-subset profile section to append to the report."""
    def find(subset: str, model: str, split: str, field: str):
        for r in summary_rows:
            if r["subset"] == subset and r["model_type"] == model and r["split"] == split:
                v = r.get(field, "")
                return float(v) if v else None
        return None

    out: list[str] = []
    out.append("")
    out.append("=" * 78)
    out.append("Appendix: subset profile summary (ClinVar 結果解釈の参考用)")
    out.append("=" * 78)
    out.append("")
    out.append(
        "Per-subset profile on val split (averaged across BERT/GPT-2 since they "
        "draw from the same nucleotide stream)."
    )
    out.append("")
    out.append(
        f"{'subset':40s} {'entropy':>8s} {'GC':>6s} {'N%':>6s} {'homo%':>7s} {'4mer-d':>7s}"
    )
    out.append("-" * 78)

    for subset in SUBSETS:
        # average BERT and GPT-2 val side metrics
        vals = {}
        for field in (
            "entropy_mean",
            "gc_mean",
            "n_fraction_mean",
            "homopolymer_rate_mean",
            "kmer4_diversity_mean",
        ):
            collected = []
            for model in ("bert", "gpt2"):
                v = find(subset, model, "val", field)
                if v is not None:
                    collected.append(v)
            vals[field] = float(np.mean(collected)) if collected else float("nan")
        out.append(
            f"{subset:40s} "
            f"{vals['entropy_mean']:>8.4f} "
            f"{vals['gc_mean']*100:>5.2f}% "
            f"{vals['n_fraction_mean']*100:>5.2f}% "
            f"{vals['homopolymer_rate_mean']*100:>6.2f}% "
            f"{vals['kmer4_diversity_mean']:>7.4f}"
        )

    out.append("")
    out.append("読み方 (要点):")
    out.append("  - entropy が他より低い subset があれば、その subset は配列の複雑度")
    out.append("    自体が低い(= モデルにとって予測しやすい)")
    out.append("  - GC% は哺乳類 ~40%、細菌 / 古細菌は多様(20〜75%)。global 系で広い分布")
    out.append("    があれば mixed kingdom データの証拠")
    out.append("  - homo% が高ければ単純反復配列(satellite DNA 等)が多い")
    out.append("  - 4mer-d は配列の多様性(N 除く 4-mer 種類数 / 256)")
    out.append("")
    out.append("ClinVar 結果との照合時の指針:")
    out.append("  - ClinVar AUROC が高い subset で entropy / 4mer-d が極端に低くないか確認")
    out.append("    → 高 AUROC が「単純な配列だから当てやすい」のせいでないかを切り分け")
    out.append("  - subset 間で GC% / N% が極端に異なる場合、それが下流タスク性能差の")
    out.append("    一因になり得ると注記しておく")
    out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis-dir",
        default="/lustre/home/matsubara/learning_source_20260529_evo2species/"
        "genome_sequence/analysis/valset_characterization",
    )
    args = ap.parse_args()
    analysis_dir = Path(args.analysis_dir)

    per_window_csv = analysis_dir / "valset_stats_per_window.csv"
    summary_csv = analysis_dir / "valset_stats_summary.csv"
    report_txt = analysis_dir / "valset_stats_report.txt"

    for p in (per_window_csv, summary_csv, report_txt):
        if not p.exists():
            print(f"ERROR: missing {p}", flush=True)
            return 1

    # ── 1) load entropies ─────────────────────────────────────────────
    buckets = load_entropies(per_window_csv)

    # ── 2) histograms ─────────────────────────────────────────────────
    for split in ("val", "train"):
        out_path = analysis_dir / f"entropy_hist_{split}.png"
        plot_entropy_histogram(buckets, split, out_path)

    # ── 3) appendix to report.txt ─────────────────────────────────────
    summary_rows = load_summary(summary_csv)
    appendix = render_profile_appendix(summary_rows)

    # Append only if not already appended (idempotent)
    current = report_txt.read_text()
    marker = "Appendix: subset profile summary"
    if marker not in current:
        report_txt.write_text(current + appendix)
        print(f"[report] appended subset profile section to {report_txt}", flush=True)
    else:
        # Replace appendix portion: keep everything before the appendix marker,
        # then append the freshly-rendered appendix.
        idx = current.find(marker)
        # Walk back to the "===" header line above the marker
        header_idx = current.rfind("\n", 0, idx)
        # Walk one line further back for the "=" delim
        delim_idx = current.rfind("\n", 0, header_idx)
        body = current[:delim_idx].rstrip() + "\n"
        report_txt.write_text(body + appendix)
        print(f"[report] replaced appendix section in {report_txt}", flush=True)

    print(f"[done] outputs in {analysis_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
