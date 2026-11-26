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

#: The two metrics the grid judges on, and which measurement answers each. The value
#: is read out of the re-measurement rather than written here: a generated table that
#: carries a conclusion of its own can disagree with the numbers above it, which is
#: what the first pass of this file did.
ANSWERS = [
    ("BERT の eval_loss_mask（valid）", "valid の [MASK] 位置のユニグラム",
     "molnl-valid-whole"),
    ("GPT-2 の val_loss（valid）", "valid の全トークンのユニグラム",
     "molnl-valid-all-tokens"),
    ("訓練バッチを読む判定（差し戻し guard）", "train の [MASK] 位置のユニグラム",
     "TRAIN_LARGEST"),
]

#: What the 2026-08-12 probe excluded from masking and the current pipeline does not.
#: Named as a suspect, not as a finding: it has not been measured either way.
PROBE_EXCLUSION = "[50256, 50258]"

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
    """The unigram floor out of either script's output.

    The two scripts name it differently and nest it differently: the MLM baseline
    writes ``unigram`` at the top level, and the all-token baseline writes
    ``baselines["unigram (degenerate solution)"]``. Reading only the first is why the
    all-token rows came back empty on the first pass.
    """
    for key in ("unigram", "unigram_floor_train_frequencies", "unigram_floor"):
        if key in payload:
            return float(payload[key])
    nested = payload.get("baselines")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key.startswith("unigram"):
                return float(value)
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
                f"| `{label}` | {payload.get('split') or 'valid（全トークン）'} | "
                f"{payload.get('eval_rows', '')} | {payload.get('split_rows', '')} | "
                f"{got:.4f} |" if got is not None else
                f"| `{label}` | | | | 値が読めない |")
        lines.append("")

    # The train samples, smallest first: the row count is what the two quoted train
    # numbers were said to differ by, so the trend across them is what settles it.
    train = []
    for label, payload in results.items():
        if payload.get("split") != "train":
            continue
        got = unigram_of(payload)
        if got is None:
            continue
        train.append((int(payload.get("eval_rows") or 0), label, got))
    train.sort()

    lines.append("## 3. どちらが正しいか")
    lines.append("")
    lines.append("**metric ごとに別の値が正しい。**床は metric と同じ量・同じ分割で"
                 "測ったものでなければ、下回ったかどうかの判定が成り立たない。")
    lines.append("")
    lines.append("| 読み合わせる metric | 正しい床 | 値 | 出どころ |")
    lines.append("|---|---|---:|---|")
    for metric, floor, source in ANSWERS:
        if source == "TRAIN_LARGEST":
            if train:
                rows, label, got = train[-1]
                lines.append(f"| {metric} | {floor} | {got:.4f} | "
                             f"`{label}`（{rows:,} 行） |")
            else:
                lines.append(f"| {metric} | {floor} | 未測定 | |")
            continue
        payload = results.get(source)
        got = unigram_of(payload) if payload else None
        shown = f"{got:.4f}" if got is not None else "未測定"
        lines.append(f"| {metric} | {floor} | {shown} | `{source}` |")
    lines.append("")

    valid_mlm = unigram_of(results.get("molnl-valid-whole") or {})
    valid_all = unigram_of(results.get("molnl-valid-all-tokens") or {})
    if valid_mlm is not None and valid_all is not None:
        lines.append(f"- **{valid_mlm:.4f} と {valid_all:.4f} はどちらも正しい。**"
                     "別の量であり、読み合わせる metric が違う。"
                     "同じ分割の 2 値として並べると誤る。")

    # Whether the two train numbers differ by the sample is a question the samples
    # answer. It is asserted either way only from them.
    quoted_8000 = next((i for i in QUOTED if i["expect"] == "molnl-train-8000"), None)
    got_8000 = unigram_of(results.get("molnl-train-8000") or {})
    if quoted_8000 and got_8000 is not None and train:
        spread = ", ".join(f"{rows:,} 行 {got:.4f}" for rows, _, got in train)
        if abs(got_8000 - quoted_8000["value"]) <= TOLERANCE:
            lines.append(f"- **{quoted_8000['value']} と "
                         f"{[i['value'] for i in QUOTED if i['expect'] == 'molnl-train-10000'][0]}"
                         f" は同じ量の 2 つの見積り**であり、違いは採点した行数である"
                         f"（{spread}）。分割の全行に対する推定値なので、"
                         f"最も大きい標本の値を採る。")
        else:
            largest_rows, _, largest = train[-1]
            lines.append(f"- **{quoted_8000['value']} は再現しない。**"
                         f"同じ 8,000 行で測ると {got_8000:.4f} である。"
                         f"行数を変えても {spread} で、"
                         f"{quoted_8000['value']} は最も大きい標本"
                         f"（{largest_rows:,} 行 {largest:.4f}）より "
                         f"{quoted_8000['value'] - largest:+.4f} 高く、"
                         f"この動きの外にある。**標本の大きさでは説明できず、"
                         f"測った量が違う。**")
            lines.append(f"- {quoted_8000['value']} のログは"
                         f"「マスク対象から除外する special ids = {PROBE_EXCLUSION}」"
                         f"と印字しており、現行のパイプラインはこの除外を行わない。"
                         f"除外すれば床は上がる向きなので符号は合うが、"
                         f"**これは容疑であり、まだ測っていない。**")
            lines.append(f"- したがって train の床として引くのは "
                         f"{largest:.4f}（{largest_rows:,} 行）とし、"
                         f"{quoted_8000['value']} は退役させるのが妥当である。"
                         f"ただしこの値で床線を描いた図表があるため、"
                         f"差し替えるかは判断を要する。")
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
