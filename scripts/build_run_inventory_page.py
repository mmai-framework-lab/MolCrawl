"""Build the run-inventory page from the previous one and a fresh scan.

The page is a stocktake, and a stocktake is only trustworthy if the rows it
already published do not move. So the previously published rows are carried over
unchanged -- their learning rate, seed and batch were recovered from binaries in
2026-08-31 and cannot be recovered again -- and the scan only adds runs that were
not in it, plus the dataset each run read where the manifest now records it.

    python scripts/build_run_inventory_page.py --old-rows <json> --new-tsv <tsv> \
        --template <html> --out <html> --asof 2026-10-28 --previous 2026-09-01

Paths live in the inputs, not here: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

PREFIX = "learning_source"
# The published row: tree, run, host, origin, start, end, hours, steps, arch,
# batch, lr, seed, max_iters, params_rec, curve_rec -- then the two this build adds.
DATASET, COHORT = 15, 16


def suffix(tree: str) -> str:
    """The tree column as the page prints it: '' for the base tree, '_x' for the rest."""
    return tree[len(PREFIX):] if tree.startswith(PREFIX) else tree


def load_new(path: Path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {}
    for r in rows:
        out[(suffix(r["tree"]), r["run"])] = r
    return out


def to_row(r, cohort):
    return [suffix(r["tree"]), r["run"], r["host"] or "—", r["origin"],
            r["start"], r["end"], r["hours"], str(r["steps"] or ""), r["arch"],
            str(r["batch"] or ""), str(r["lr"] or ""), str(r["seed"] or ""),
            str(r["max_iters"] or ""), r["params_rec"], r["curve_rec"],
            r["dataset"], cohort]


def stat(label, value, unit="", warn=False):
    u = f'<small>{unit}</small>' if unit else ""
    cls = ' class="stat w"' if warn else ' class="stat"'
    return f'<div{cls}><span class="k">{label}</span><span class="v">{value}{u}</span></div>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-rows", required=True)
    ap.add_argument("--new-tsv", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--previous", required=True)
    args = ap.parse_args()

    old = json.loads(Path(args.old_rows).read_text(encoding="utf-8"))
    new = load_new(Path(args.new_tsv))
    old_cohort, new_cohort = f"〜{args.previous}", f"{args.previous} 以降"

    rows, seen = [], set()
    missing = []
    for r in old:
        key = (r[0], r[1])
        seen.add(key)
        hit = new.get(key)
        if hit is None:
            missing.append(f"{r[0] or '(本体)'}/{r[1]}")
        rows.append(list(r) + [hit["dataset"] if hit else "", old_cohort])
    added = [to_row(r, new_cohort) for k, r in new.items() if k not in seen]
    rows.extend(added)

    live = sum(1 for r in rows if r[3] == "実走行")
    copied = sum(1 for r in rows if r[3] == "コピー")
    hours = sum(float(r[6]) for r in rows if r[6] not in ("", "不明"))
    manifest = sum(1 for r in rows if "run_manifest" in r[13])
    norec = sum(1 for r in rows if r[13] == "なし")
    withds = sum(1 for r in rows if r[15])

    stats = ('<div class="stats">'
             + stat("全 run", f"{len(rows):,}")
             + stat("今期の追加", f"{len(added):,}")
             + stat("実走行", f"{live:,}")
             + stat("持ち込み", f"{copied:,}")
             + stat("実時間", f"{hours:,.0f}", "h")
             + stat("manifest あり", f"{manifest:,}")
             + stat("記録なし", f"{norec:,}", warn=True)
             + "</div>")

    # What the period added, counted from the rows rather than recalled.
    per_tree: dict[tuple[str, str], int] = {}
    for r in added:
        per_tree[(r[0], r[8])] = per_tree.get((r[0], r[8]), 0) + 1
    body = "".join(
        f"<tr><td class='m'>{html.escape(t or '(本体)')}</td><td class='m'>{a}</td>"
        f"<td class='n'>{n}</td></tr>"
        for (t, a), n in sorted(per_tree.items(), key=lambda kv: -kv[1]))
    added_hours = sum(float(r[6]) for r in added if r[6] not in ("", "不明"))
    added_ds = sum(1 for r in added if r[15])

    dek = ('<p class="dek"><code>learning_source*</code> 全 18 ツリーを走査した '
           f'<strong>{len(rows):,} run</strong>。'
           f'{args.previous} の棚卸しから <strong>{len(added):,} run</strong> 増えました。'
           '開始時刻は <span class="m">events.out.tfevents.&lt;unixtime&gt;</span> と '
           '<span class="m">logging_YYYYMMDD_HHMMSS.csv</span> のファイル名から、終了は '
           'checkpoint / ログの mtime から。<strong>列見出しで並べ替え、上の欄で絞り込みができます。</strong></p>')

    notes = f'''<div class="note add">
  <span class="lab">{args.asof} 追記 — 今期に増えた run と、塞がった穴</span>
  <p><strong>{args.previous} から {len(added):,} run 増えました（実時間 {added_hours:,.0f} h）。</strong>
  内訳は下表のとおりです。数はこの表の行から数えたもので、報告書から引き写したものではありません。</p>
  <div class="tw" style="margin:10px 0"><table style="min-width:0">
  <thead><tr><th>ツリー</th><th>実装</th><th style="text-align:right">run</th></tr></thead>
  <tbody>{body}</tbody></table></div>
  <p><strong>読んだデータセットが記録されるようになりました。</strong>
  {args.previous} の版は「どの run にも記録がない」と書きましたが、<span class="m">run_manifest.json</span> を持つ run は
  <span class="m">data.dataset_dir</span> を残します。<strong>いま {withds:,} run</strong>（今期の追加分では {added_ds:,} run）で
  読んだデータセットが分かり、<strong>データセット列</strong>に出しています。manifest は学習率・seed・実効バッチ・
  走らせた commit も平文で持つので、<strong>この表の値はバイナリからの復元ではなく、run 自身の申告です。</strong></p>
  <p><strong>{args.previous} までの {len(old):,} run は値を動かしていません。</strong>
  それらの学習率・seed・バッチは 2026-08-31 に <span class="m">ckpt.pt</span> と
  <span class="m">training_args.bin</span> から回収したもので、<strong>今回の走査では復元できません</strong>
  （manifest が無いため）。棚卸し列でどちらの版の行かを絞り込めます。
  {"<strong>前回の表にあって、今回ディスク上に見つからなかった run が " + str(len(missing)) + " 本あります</strong>（" + "、".join("<span class='m'>" + html.escape(m) + "</span>" for m in missing) + "）。行は残してあります。" if missing else "前回の表の run は全て今回も見つかりました。"}</p>
</div>'''

    footer = (f'集計元 <span class="m">report/run-inventory-{args.asof}.tsv</span>'
              f'（走査スクリプト <span class="m">scripts/run_inventory_scan.py</span>）'
              f'<br>{args.previous} 時点の集計元 <span class="m">tmp/run-inventory-2026-08-31.tsv</span> ／ '
              f'保全台帳 <span class="m">tmp/learning-source-md5-2026-08-31.jsonl</span>')

    cohorts = "".join(f'<option>{c}</option>' for c in (old_cohort, new_cohort))
    page = Path(args.template).read_text(encoding="utf-8")
    for key, value in (("{{ASOF}}", args.asof), ("{{DEK}}", dek), ("{{STATS}}", stats),
                       ("{{COHORTS}}", cohorts), ("{{NOTES}}", notes), ("{{FOOTER}}", footer),
                       ("{{ROWS}}", json.dumps(rows, ensure_ascii=False, separators=(",", ":")))):
        assert key in page, key
        page = page.replace(key, value)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out}  {len(rows):,} run（追加 {len(added):,}／欠落 {len(missing)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
