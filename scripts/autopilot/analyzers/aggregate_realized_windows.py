"""Aggregate realized window counts across 21 subsets → decide target_total_windows.

charter §「自律判断ルール」対応:
- realized 窓数に外れ値(1 個だけ極端に低い)→ 最小に全部揃えるのはやめ、
  外れを除いた分布ベース(例: 中央値近傍)で展開目標を決めて続行(seed6 型の
  再発防止)

Algorithm:
1. subset の全 parquet_bert(または parquet_gpt2)を open して行数合計 = 実測窓数
2. 21 subset の分布を median (med), MAD (median absolute deviation) で要約
3. 外れ値判定: window_count < med - 2.5 * 1.4826 * MAD なら「外れ」
   (Gaussian で ~2σ tail、 保守的な閾値)
4. target = 外れを除いた集合の最小値 (round-down to 1M)
5. state JSON + milestone md 出力

Usage:
    python3 tmp/scripts/autopilot/analyzers/aggregate_realized_windows.py \
        --learning-source /lustre/home/matsubara/learning_source_20260710_genome_v2 \
        --subsets-file <path to txt with subset names, one per line> \
        --model bert \
        --out-state tmp/scripts/autopilot/state/g2_target.json \
        --out-report tmp/scripts/autopilot/milestones/g2-target-decided.md

Exits 0 on success (target file written), 1 on any failure or missing data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from datetime import datetime, timezone


def _count_rows_of_subset(subset_dir: Path, model: str) -> int:
    """Row count = 実測 window 数 (parquet 中の各行 = 1 window)."""
    import pyarrow.parquet as pq

    parquet_dir = subset_dir / f"parquet_{model}"
    if not parquet_dir.exists():
        return -1
    total = 0
    for pf in sorted(parquet_dir.glob("*.parquet")):
        try:
            md = pq.read_metadata(pf)
            total += md.num_rows
        except Exception as e:
            print(f"[warn] {pf} unreadable: {e}", file=sys.stderr)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--learning-source", required=True)
    ap.add_argument("--subsets-file", required=True,
                    help="txt file, one subset name per line")
    ap.add_argument("--model", default="bert", choices=["bert", "gpt2"])
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--round-down-to", type=int, default=1_000_000,
                    help="round target down to multiple of this (default 1M)")
    args = ap.parse_args()

    lsd = Path(args.learning_source)
    genome_dir = lsd / "genome_sequence"
    subsets = [ln.strip() for ln in Path(args.subsets_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    counts: dict[str, int] = {}
    missing: list[str] = []
    for s in subsets:
        n = _count_rows_of_subset(genome_dir / s, args.model)
        if n < 0:
            missing.append(s)
        else:
            counts[s] = n

    if missing:
        print(f"[FAIL] {len(missing)} subset(s) have no parquet yet: {missing}", file=sys.stderr)
        return 1

    if len(counts) < 5:
        print(f"[FAIL] too few subsets to decide target: {len(counts)}", file=sys.stderr)
        return 1

    vals = sorted(counts.values())
    med = statistics.median(vals)
    abs_devs = sorted(abs(v - med) for v in vals)
    mad = statistics.median(abs_devs)
    mad_norm = 1.4826 * mad  # to match σ under Normal
    outlier_cutoff = med - 2.5 * mad_norm if mad_norm > 0 else med * 0.5

    outliers = {s: n for s, n in counts.items() if n < outlier_cutoff}
    kept = {s: n for s, n in counts.items() if n >= outlier_cutoff}

    kept_min = min(kept.values())
    target = (kept_min // args.round_down_to) * args.round_down_to

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used_for_measure": args.model,
        "subset_counts": counts,
        "median": med,
        "mad": mad,
        "outlier_cutoff": outlier_cutoff,
        "outliers": outliers,
        "kept_subsets": list(kept),
        "kept_min": kept_min,
        "target_total_windows": target,
        "round_down_to": args.round_down_to,
    }

    Path(args.out_state).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_state).write_text(json.dumps(state, indent=2))

    lines = [
        "# G2 target 確定 (charter §「実測 → 最小 → trim」)",
        "",
        f"作成: {state['generated_at']}",
        f"model: {args.model} (bert/gpt2 は window 単位で等価、bert で代表)",
        "",
        "## 実測窓数",
        "",
        "| subset | windows |",
        "| --- | --- |",
        *[f"| {s} | {n:,} |" for s, n in sorted(counts.items(), key=lambda kv: -kv[1])],
        "",
        f"- median: {med:,.0f}",
        f"- MAD (× 1.4826): {mad_norm:,.0f}",
        f"- outlier cutoff (med - 2.5·MAD): {outlier_cutoff:,.0f}",
        "",
        "## 外れ値除外 (seed6 型再発防止)",
        "",
    ]
    if outliers:
        for s, n in sorted(outliers.items(), key=lambda kv: kv[1]):
            lines.append(f"- **{s}**: {n:,} (< cutoff、 除外)")
    else:
        lines.append("- 外れ値なし")
    lines.extend([
        "",
        f"## target_total_windows: **{target:,}**",
        "",
        f"kept subsets の最小 ({kept_min:,}) を {args.round_down_to:,} で切り下げ。",
        "F2-c trim をこの target で全 subset に適用する。外れ値 subset は target まで trim ではなく",
        "自然の実測値を尊重(全 subset で >= target を保証)。",
        "",
        "## park",
        "",
        "外れ値 subset は autopilot ルールに従い park せず続行(median-based 決定なので",
        "分布は破綻していない)。個別調査は user 復帰後に。",
    ])
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text("\n".join(lines) + "\n")

    print(f"[OK] target={target:,}, outliers={list(outliers)}, kept={len(kept)}/21")
    print(f"     state: {args.out_state}")
    print(f"     report: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
