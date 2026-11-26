"""What each quoted molnl floor is, and which of them answers which metric.

The 2026-11-26 data order's item 5 asks, for four numbers, what quantity was measured
on what set and how, and then which is correct. The four are all in this project's own
records; this reads the re-measurement beside them and writes the table.

The point the table has to carry is that they are not one quantity four times:

* a masked-language floor is read against ``eval_loss_mask`` and scores only the
  positions the collator selects;
* an all-token floor is read against an autoregressive ``val_loss`` and scores every
  position;
* and a floor measured on the train split is not the floor on the valid split.

So "which is correct" has no single answer: it is correct next to the metric it was
measured for, and the two pairs that were put against each other are a train floor and
a valid all-token floor.

    molnl_floor_provenance.py --results <dir of the job's JSONs> --out <table.md>

Read-only: it reads the JSONs the measurement wrote and writes one markdown file.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os

#: What the reports and the order quote, and where each one is written down.
QUOTED = [
    {
        "value": 3.8562,
        "quantity": "unigram at [MASK] positions",
        "split": "train",
        "rows": "a fixed random 10,000-block sample of 318,118",
        "record": "report/mlm-baselines-20261106-molnl-train/molnl-train.json",
        "quoted_by": "2026-11-10 以降の報告書（差し戻し guard の系統）",
        "expect": "molnl-train-10000",
    },
    {
        "value": 3.8638,
        "quantity": "unigram at [MASK] positions",
        "split": "（ログに書かれていない）",
        "rows": "8,000 blocks",
        "record": "workflows/slurm-logs/mc-molnl-probeprep-22503.out（2026-08-12）",
        "quoted_by": "2026-11-26 データの提出 項目 5／review-11・13・14／"
                     "rt1-bert-grid-summary-1028.tsv",
        "expect": "molnl-train-8000",
    },
    {
        "value": 4.6449,
        "quantity": "unigram at [MASK] positions",
        "split": "valid",
        "rows": "全 1,631 blocks",
        "record": "report/mlm-baselines-20261102/full/molnl.json",
        "quoted_by": "2026-11-10 以降の報告書",
        "expect": "molnl-valid-whole",
    },
    {
        "value": 4.6514,
        "quantity": "unigram over every token（自己回帰の床）",
        "split": "valid",
        "rows": "全 1,631 blocks・1,668,513 の次トークン位置",
        "record": "report/molnl-gpt2-unigram-floor.json",
        "quoted_by": "2026-11-26 データの提出 項目 5／molnl-gpt2-grid-val-loss.tsv",
        "expect": "molnl-valid-all-tokens",
    },
]

#: The two metrics the grid judges on, and the floor each one is read against.
ANSWERS = [
    ("BERT の eval_loss_mask（valid）", "valid の [MASK] 位置のユニグラム", 4.6449),
    ("GPT-2 の val_loss（valid）", "valid の全トークンのユニグラム", 4.6514),
    ("訓練バッチを読む判定（差し戻し guard）", "train の [MASK] 位置のユニグラム",
     None),
]

TOLERANCE = 0.0005


def read_results(directory: str):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as handle:
            try:
                out[label] = json.load(handle)
            except json.JSONDecodeError:
                continue
    return out


def unigram_of(payload):
    """The unigram floor out of either script's output."""
    for key in ("unigram", "unigram_floor_train_frequencies", "unigram_floor"):
        if key in payload:
            return float(payload[key])
    return None


def render(results):
    lines = ["# molnl の基準線 4 値 出どころと再測定", ""]
    lines.append("2026-11-26 データの提出 項目 5 への回答の素材。"
                 "4 値はいずれもこのプロジェクト自身の記録にあり、"
                 "**同じ量を 4 回測ったものではない**。")
    lines.append("")

    lines.append("## 1. 引かれている 4 値が何なのか")
    lines.append("")
    lines.append("| 値 | 量 | 分割 | 採点した行 | 記録 |")
    lines.append("|---:|---|---|---|---|")
    for item in QUOTED:
        lines.append(f"| {item['value']} | {item['quantity']} | {item['split']} | "
                     f"{item['rows']} | `{item['record']}` |")
    lines.append("")

    lines.append("## 2. 一つのパイプラインで測り直した結果")
    lines.append("")
    lines.append("| 引かれている値 | 期待する測定 | 測り直し | 差 | 再現 |")
    lines.append("|---:|---|---:|---:|:--:|")
    reproduced = {}
    for item in QUOTED:
        payload = results.get(item["expect"])
        if payload is None:
            lines.append(f"| {item['value']} | `{item['expect']}` | 未測定 | | — |")
            reproduced[item["value"]] = None
            continue
        got = unigram_of(payload)
        if got is None:
            lines.append(f"| {item['value']} | `{item['expect']}` | 値が読めない | | — |")
            reproduced[item["value"]] = None
            continue
        delta = got - item["value"]
        ok = abs(delta) <= TOLERANCE
        reproduced[item["value"]] = ok
        lines.append(f"| {item['value']} | `{item['expect']}` | {got:.4f} | "
                     f"{delta:+.4f} | {'はい' if ok else '**いいえ**'} |")
    lines.append("")

    extra = [label for label in sorted(results)
             if label not in {i["expect"] for i in QUOTED}]
    if extra:
        lines.append("追加で測ったもの（引かれている値と対になっていない測定）。")
        lines.append("")
        lines.append("| 測定 | 分割 | 採点した行 | 分割の全行 | ユニグラム |")
        lines.append("|---|---|---:|---:|---:|")
        for label in extra:
            payload = results[label]
            got = unigram_of(payload)
            lines.append(
                f"| `{label}` | {payload.get('split', '')} | "
                f"{payload.get('eval_rows', '')} | {payload.get('split_rows', '')} | "
                f"{got:.4f} |" if got is not None else
                f"| `{label}` | | | | 値が読めない |")
        lines.append("")

    lines.append("## 3. どちらが正しいか")
    lines.append("")
    lines.append("**metric ごとに別の値が正しい。**床は metric と同じ量・同じ分割で"
                 "測ったものでなければ、下回ったかどうかの判定が成り立たない。")
    lines.append("")
    lines.append("| 読み合わせる metric | 正しい床 | 値 |")
    lines.append("|---|---|---:|")
    for metric, floor, value in ANSWERS:
        shown = f"{value}" if value is not None else "§2 の train 行（下記）"
        lines.append(f"| {metric} | {floor} | {shown} |")
    lines.append("")
    lines.append("- **4.6449 と 4.6514 はどちらも正しい。**別の量であり、"
                 "読み合わせる metric が違う。同じ分割の 2 値として並べると誤る。")
    lines.append("- **3.8562 と 3.8638 は同じ量の 2 つの見積り**であり、"
                 "違いは採点した行数（10,000 と 8,000）である。"
                 "分割の全 318,118 行に対する推定値なので、"
                 "§2 の大きい標本の行が最も確からしい。")
    lines.append("- train の床を valid の metric の隣に置くことはしない"
                 "（2026-11-10 に全資料で直した件）。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="the directory the measurement wrote its JSONs into")
    ap.add_argument("--out", required=True, help="the markdown table to write")
    args = ap.parse_args(argv)

    results = read_results(args.results)
    if not results:
        raise SystemExit(f"no JSON results under {args.results}")
    text = render(results)
    # BOM: these go to the report directory and are attached as they are.
    with io.open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
        handle.write(text)

    print(f"=== {len(results)} measurement(s) read from {args.results} ===")
    for label in sorted(results):
        got = unigram_of(results[label])
        payload = results[label]
        print(f"  {label:28s} split {str(payload.get('split')):6s} "
              f"rows {str(payload.get('eval_rows')):>7s}  "
              f"unigram {got:.4f}" if got is not None else f"  {label}: no value")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
