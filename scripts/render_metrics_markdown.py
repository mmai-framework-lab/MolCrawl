"""Render the boss-requested 表 A / 表 B Markdown tables from
metrics_summary.csv (output of collect_metrics_discovery.py).

Reads ``tmp/metrics_summary.csv`` and writes
``tmp/metrics_summary.md`` containing:

  - 表 A: GPT-2 (autoregressive)
  - 表 B: MLM-family (BERT / DNABERT2 / ESM-2 / ChemBERTa2 / RoBERTa /
                     RNAformer / LLaMA causal)
  - 表 C: genome_sequence subset 21 (Evo2 species campaign — separate,
                                    not part of the canonical foundation
                                    model survey)
  - 註: 数値上の異常 + 評価方式の注意

canonical source 優先(20260520_uncapped → 20260316 → 20260407 → 20260302)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import OrderedDict

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
CSV = REPO / "tmp" / "metrics_summary.csv"
OUT = REPO / "tmp" / "metrics_summary.md"

PRIORITY = [
    "learning_source_20260520_uncapped",
    "learning_source_20260316",
    "learning_source_20260407_compounds",
    "learning_source_20260302",
]
SUBSET_SRC = "learning_source_20260529_evo2species"

SIZE_ORDER = OrderedDict([
    ("small", 0), ("medium", 1), ("large", 2), ("ex-large", 3), ("xl", 4),
])
MODALITY_ORDER = OrderedDict([
    ("genome_sequence", 0), ("protein_sequence", 1), ("rna", 2),
    ("compounds", 3), ("molecule_nat_lang", 4),
])
ARCH_ORDER = OrderedDict([
    ("BERT", 0), ("RoBERTa", 1), ("DNABERT2", 2), ("ESM-2", 3),
    ("ChemBERTa2", 4), ("RNAformer", 5), ("LLaMA", 6),
])


def size_key(s: str) -> int:
    return SIZE_ORDER.get(s, 99)


def main() -> int:
    rows = list(csv.DictReader(CSV.open()))

    # canonical: per (modality, arch, size, variant), pick highest-priority source.
    canonical: dict[tuple, dict] = {}
    subset_rows: list[dict] = []
    for r in rows:
        if r["source_dir"] == SUBSET_SRC:
            subset_rows.append(r)
            continue
        if r["source_dir"] not in PRIORITY:
            continue
        key = (r["modality"], r["arch"], r["size"], r["variant"])
        if key not in canonical or (
            PRIORITY.index(r["source_dir"]) <
            PRIORITY.index(canonical[key]["source_dir"])
        ):
            canonical[key] = r

    gpt2 = sorted(
        [r for r in canonical.values() if r["arch"] == "GPT-2"],
        key=lambda r: (
            MODALITY_ORDER.get(r["modality"], 99), r["variant"],
            size_key(r["size"]),
        ),
    )
    mlm = sorted(
        [r for r in canonical.values() if r["arch"] != "GPT-2"],
        key=lambda r: (
            MODALITY_ORDER.get(r["modality"], 99),
            ARCH_ORDER.get(r["arch"], 99),
            r["variant"], size_key(r["size"]),
        ),
    )
    subset = sorted(
        subset_rows,
        key=lambda r: (
            ARCH_ORDER.get(r["arch"], 99) if r["arch"] != "GPT-2" else 99,
            r["variant"],
        ),
    )

    def src_short(s: str) -> str:
        return s.replace("learning_source_", "ls_")

    def cell(v: str) -> str:
        return v if v else "-"

    out_lines = []
    out_lines.append("# Foundation model 学習メトリクス一覧")
    out_lines.append("")
    out_lines.append(f"自動生成: `tmp/scripts/render_metrics_markdown.py`  /  入力: `tmp/metrics_summary.csv` ({len(rows)} rows)")
    out_lines.append("")
    out_lines.append("- canonical source: `ls_20260520_uncapped` 優先 → なければ `ls_20260316` → `ls_20260407_compounds` → `ls_20260302`")
    out_lines.append("- `_backup` / `_pre_retokenize` / `_overfit_backup` / `_patience5_backup` / `broken_vocab` / `yigarashi` で始まる variant は除外")
    out_lines.append(f"- genome_sequence subset campaign (`{SUBSET_SRC}`、 21 subset Evo2 work) は別表 C に分離")
    out_lines.append("")
    out_lines.append("---")
    out_lines.append("")

    # 表 A
    out_lines.append("## 表 A — GPT-2 (CLM)")
    out_lines.append("")
    out_lines.append("| modality | size | variant | val_loss | perplexity | BPC | steps | tokens (M) | src |")
    out_lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in gpt2:
        out_lines.append(
            f"| {r['modality']} | {r['size']} | {r['variant']} | "
            f"{cell(r['val_loss'])} | {cell(r['perplexity'])} | {cell(r['bpc'])} | "
            f"{cell(r['steps'])} | {cell(r['tokens_seen_M'])} | {src_short(r['source_dir'])} |"
        )
    out_lines.append("")

    # 表 B
    out_lines.append("## 表 B — MLM 系(BERT / DNABERT2 / ESM-2 / ChemBERTa2 / RoBERTa / RNAformer) + LLaMA causal")
    out_lines.append("")
    out_lines.append("BERT 系は MLM CE loss、 LLaMA は CLM CE loss。 BPC は `loss / ln(2)`、 pseudo-perp は `exp(loss)`(MLM の場合 GPT-2 perplexity と直接比較不可)。")
    out_lines.append("")
    out_lines.append("| modality | arch | size | variant | eval_loss | pseudo_ppl | BPC | mlm_acc | steps | src |")
    out_lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in mlm:
        out_lines.append(
            f"| {r['modality']} | {r['arch']} | {r['size']} | {r['variant']} | "
            f"{cell(r['val_loss'])} | {cell(r['pseudo_perp'])} | {cell(r['bpc'])} | "
            f"{cell(r['mlm_accuracy'])} | {cell(r['steps'])} | {src_short(r['source_dir'])} |"
        )
    out_lines.append("")

    # 表 C
    out_lines.append("## 表 C — genome_sequence subset campaign (Evo2、21 subset × BERT/GPT-2)")
    out_lines.append("")
    out_lines.append(f"source: `{src_short(SUBSET_SRC)}`。 LR sweep 用の `-lr*` 行を含む。")
    out_lines.append("")
    out_lines.append("| arch | subset / variant | val_loss | perp | BPC | steps | tokens (M) | src_file |")
    out_lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in subset:
        perp = r["perplexity"] or r["pseudo_perp"]
        out_lines.append(
            f"| {r['arch']} | {r['variant']} | {cell(r['val_loss'])} | "
            f"{cell(perp)} | {cell(r['bpc'])} | {cell(r['steps'])} | "
            f"{cell(r['tokens_seen_M'])} | {r['source'][:40]} |"
        )
    out_lines.append("")

    out_lines.append("---")
    out_lines.append("")

    # 註 — 異常点
    out_lines.append("## 註 — 数値上の異常点(誠実報告)")
    out_lines.append("")
    out_lines.append("### degenerate / chance-level に張り付き")
    out_lines.append("")
    flagged = []
    for r in mlm:
        try:
            vl = float(r["val_loss"]) if r["val_loss"] else None
        except ValueError:
            vl = None
        if vl is not None and vl > 5.5:
            flagged.append(
                f"- **{r['modality']}** / {r['arch']} / {r['size']} / {r['variant']}: "
                f"eval_loss = {vl:.4f} (pseudo_ppl = {r['pseudo_perp']})"
            )
    if not flagged:
        out_lines.append("- (該当なし)")
    else:
        out_lines.extend(flagged)
    out_lines.append("")
    out_lines.append("→ 主に `genome_sequence` の MLM 系(BERT / RoBERTa / DNABERT2 small-medium) と `rna` BERT large。 small モデル + 単塩基トークナイザー(vocab=10)+ MLM ヘッド の組み合わせでは chance level を抜けない可能性が高い。同 modality の LLaMA / GPT-2 / DNABERT2-large は機能している。")
    out_lines.append("")
    out_lines.append("### MLM accuracy が全行 `-`")
    out_lines.append("")
    out_lines.append("HF Trainer デフォルトでは `eval_accuracy` を計算しない。各 modality の学習スクリプトに `compute_metrics` 実装が無い。後追いで MLM accuracy を出すには、各 ckpt × valid set で masked prediction を別途算定する必要あり(数時間)。")
    out_lines.append("")

    # 註 — 構造的乖離
    out_lines.append("## 註 — 指示書スクリプトとの構造的乖離")
    out_lines.append("")
    out_lines.append("| 想定 | 実体 |")
    out_lines.append("|---|---|")
    out_lines.append("| 単一 LEARNING_SOURCE_DIR | **4 source dir** に分散(20260316 / 20260407_compounds / 20260520_uncapped / 20260529_evo2species)|")
    out_lines.append("| `mol_nat_lang/` | **`molecule_nat_lang/`** |")
    out_lines.append("| `genome_sequence/...-clinvar` | **`genome_sequence_clinvar/`**(別親 dir) |")
    out_lines.append("| `protein_sequence/...-proteingym` | **`protein_sequence_proteingym/`**(同上) |")
    out_lines.append("| `{model_dir}/trainer_state.json` | **`{model_dir}/checkpoint-NNNNN/trainer_state.json`** |")
    out_lines.append("| BERT/GPT-2/DNABERT2/ESM-2/ChemBERTa2 のみ | **LLaMA / RoBERTa / RNAformer** も存在(指示書記載なし) |")
    out_lines.append("")

    OUT.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {OUT}  ({len(out_lines)} lines)")
    print(f"  表 A: {len(gpt2)} rows")
    print(f"  表 B: {len(mlm)} rows")
    print(f"  表 C: {len(subset)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
