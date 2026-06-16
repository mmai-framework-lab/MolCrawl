"""Compile per-subset species composition for the 21 Evo2 subsets.

Produces both a per-subset detail block and a cross-subset summary table,
saving a markdown report. Highlights:

  - Which species ends up at the alphabetically-last parquet position
    (= dominates the Phase-4 valid/test tail under the old split logic).
  - Major-group breakdown (vertebrate / invertebrate / plant / fungi /
    protist / bacteria / archaea).
  - GC content and N% distribution.
  - GCF vs GCA prefix counts.
  - Subsets sharing the same tail parquet (= identical val/test before fix).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSETS_DIR = REPO_ROOT / "assets" / "genome_species_list" / "subsets"

ALL_SUBSETS = [
    "mammal_centered",
    *(f"eukaryote_matched_random_seed{i}" for i in range(1, 11)),
    *(f"global_random_seed{i}" for i in range(1, 11)),
]


def load_subset(name: str) -> pd.DataFrame:
    return pd.read_csv(SUBSETS_DIR / f"{name}.csv")


def tail_accession(df: pd.DataFrame) -> str:
    """Return the alphabetically-last assembly_accession (= tail parquet)."""
    return sorted(df["assembly_accession"].astype(str).tolist())[-1]


def summarize_subset(df: pd.DataFrame) -> dict:
    n = len(df)
    total_bp = int(df["genome_size_bp"].sum())
    mean_gc = float(df["gc_content"].mean())
    median_gc = float(df["gc_content"].median())
    mean_n = float(df["n_fraction"].mean())
    n_gcf = df["assembly_accession"].str.startswith("GCF_").sum()
    n_gca = df["assembly_accession"].str.startswith("GCA_").sum()
    tail = tail_accession(df)
    tail_row = df[df["assembly_accession"] == tail].iloc[0]
    tail_species = tail_row["species_name"]
    tail_bp = int(tail_row["genome_size_bp"])
    tail_gc = float(tail_row["gc_content"])
    tail_share = tail_bp / max(total_bp, 1)

    group_counts = df["major_group"].fillna("(unknown)").value_counts().to_dict()

    superkingdom_counts = (
        df["superkingdom"].fillna("(unknown)").value_counts().to_dict()
    )

    return {
        "n_assemblies": n,
        "total_bp": total_bp,
        "mean_gc": mean_gc,
        "median_gc": median_gc,
        "mean_n_fraction": mean_n,
        "n_gcf": int(n_gcf),
        "n_gca": int(n_gca),
        "tail_accession": tail,
        "tail_species": tail_species,
        "tail_bp": tail_bp,
        "tail_gc": tail_gc,
        "tail_share": tail_share,
        "major_groups": group_counts,
        "superkingdoms": superkingdom_counts,
    }


def format_int_bp(bp: int) -> str:
    if bp >= 1e9:
        return f"{bp/1e9:.2f} Gbp"
    if bp >= 1e6:
        return f"{bp/1e6:.1f} Mbp"
    return f"{bp/1e3:.0f} kbp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "tmp" / "docs_tmp_local"
                             / "20260616-subset-composition-report.md"),
        help="Output markdown report path",
    )
    args = ap.parse_args()

    summaries = {s: summarize_subset(load_subset(s)) for s in ALL_SUBSETS}

    # Group subsets by their tail accession to find shared-tail clusters.
    by_tail: dict[str, list[str]] = defaultdict(list)
    for s, info in summaries.items():
        by_tail[info["tail_accession"]].append(s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Evo2 subset composition — 21 subset overview")
    lines.append("")
    lines.append(
        "21 subset の構成種・GC分布・末尾 parquet を集計したものです。"
        "Phase 4 (parquet → Arrow split) の末尾切り出しで valid/test が"
        "どの種で代表されるか、subset 間で末尾が共有されているか、"
        "「ホエール大学」型の事故が他にもあるかを一覧化しています。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: cross-subset overview table ────────────────────────
    lines.append("## 1. 全 21 subset の俯瞰")
    lines.append("")
    lines.append("| subset | 種数 | 総 bp | 平均 GC | 末尾 parquet (= valid/test 主要素) |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for s in ALL_SUBSETS:
        info = summaries[s]
        tail_label = (
            f"{info['tail_accession']}  ({info['tail_species']}, "
            f"{format_int_bp(info['tail_bp'])}, "
            f"share={info['tail_share']*100:.1f}%)"
        )
        lines.append(
            f"| {s} | {info['n_assemblies']} | "
            f"{format_int_bp(info['total_bp'])} | "
            f"{info['mean_gc']*100:.1f}% | "
            f"{tail_label} |"
        )
    lines.append("")
    lines.append(
        "**share** = 当該末尾 parquet が subset 内に占めるバイト割合。"
        "share が高いほど valid/test が末尾種 1 つで占有される度合いが強い。"
    )
    lines.append("")

    # ── Section 2: shared-tail clusters ───────────────────────────────
    lines.append("## 2. 末尾 parquet が共有されている subset 群")
    lines.append("")
    lines.append(
        "Phase 4 の split は parquet 末尾を valid/test にする実装でした。"
        "同じ末尾 parquet を共有する subset 同士は、修正前は valid/test が"
        "完全に一致します(=「ホエール大学」型問題)。"
    )
    lines.append("")
    lines.append("| 共有末尾 accession | 種 | サブセット |")
    lines.append("| --- | --- | --- |")
    for tail, subs in sorted(by_tail.items(), key=lambda kv: -len(kv[1])):
        if len(subs) < 2:
            continue
        df0 = load_subset(subs[0])
        sp = df0[df0["assembly_accession"] == tail]["species_name"].iloc[0]
        lines.append(f"| `{tail}` | {sp} | {', '.join(subs)} |")
    lines.append("")
    lines.append("(共有 2 件未満は省略)")
    lines.append("")

    # ── Section 3: per-subset detail ──────────────────────────────────
    lines.append("## 3. subset 別の詳細プロファイル")
    lines.append("")
    for s in ALL_SUBSETS:
        info = summaries[s]
        df = load_subset(s)
        lines.append(f"### `{s}`")
        lines.append("")
        lines.append(f"- 種数: **{info['n_assemblies']}**")
        lines.append(f"- 総ゲノム: **{format_int_bp(info['total_bp'])}**")
        lines.append(
            f"- 平均 GC: {info['mean_gc']*100:.2f}% "
            f"(中央値 {info['median_gc']*100:.2f}%)"
        )
        lines.append(f"- 平均 N 含量: {info['mean_n_fraction']*100:.2f}%")
        lines.append(f"- GCF / GCA: {info['n_gcf']} / {info['n_gca']}")
        lines.append("")
        # Major group breakdown
        lines.append("**major_group 内訳**:")
        groups = sorted(info["major_groups"].items(), key=lambda kv: -kv[1])
        for g, c in groups:
            lines.append(f"  - {g}: {c}")
        lines.append("")
        # Top 5 species by genome size
        lines.append("**ゲノム上位 5 種(末尾 parquet 候補)**:")
        top5 = df.nlargest(5, "genome_size_bp")[
            ["assembly_accession", "species_name", "genome_size_bp", "gc_content"]
        ]
        for _, row in top5.iterrows():
            lines.append(
                f"  - `{row['assembly_accession']}` {row['species_name']} "
                f"({format_int_bp(int(row['genome_size_bp']))}, "
                f"GC={row['gc_content']*100:.1f}%)"
            )
        lines.append("")
        # Tail parquet detail
        lines.append(
            f"**末尾 parquet (alphabetical order last)**: "
            f"`{info['tail_accession']}` ({info['tail_species']}, "
            f"{format_int_bp(info['tail_bp'])}, "
            f"share={info['tail_share']*100:.1f}%)"
        )
        if info["tail_share"] > 0.05:
            lines.append("  ⚠️  末尾 share が 5% 超 — Phase 4 修正前は valid/test がこの種で大きく占められる")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
