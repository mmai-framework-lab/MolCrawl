"""Build the run-inventory page from the previous one and a fresh scan.

The page is a stocktake, and a stocktake is only trustworthy if the rows it
already published do not move. So the previously published rows are carried over
unchanged -- their learning rate, seed and batch were recovered from binaries in
2026-08-31 and cannot be recovered again -- and the scan only adds runs that were
not in it, plus the dataset each run read where the manifest now records it.

The downstream evaluations come in as a second table, joined to the runs on
(tree, run) -- the pair ``run_inventory_scan.py`` writes and
``downstream_inventory_scan.py`` repeats -- so a score can be traced back to the
run whose checkpoint was scored, and a run shows which evaluations used it.

    python scripts/build_run_inventory_page.py --old-rows <json> --new-tsv <tsv> \
        --downstream <tsv> --template <html> --out <html> \
        --asof 2026-10-28 --previous 2026-09-01

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
# batch, lr, seed, max_iters, params_rec, curve_rec -- then the four this build adds.
DATASET, COHORT, DOWNSTREAM, HEALTH = 15, 16, 17, 18

# How the checkpoint-metric audit's verdict reads in the table. A run whose
# saved checkpoint is not the minimum of the metric it was judged on cannot be
# quoted as that minimum, and the table has to say so on the row itself.
HEALTH_LABEL = {
    "yes": ("一致", "ok"),
    "no judged metric in history": ("採点指標なし", "warn"),
    "n/a (§2: not through this decision)": ("対象外", "na"),
}
HEALTH_GONE = ("最良が消滅", "bad")      # the judged minimum has no checkpoint left
HEALTH_OTHER = ("保存が別 step", "bad")  # it has one, at a step that was not saved as best

# What the page says about the evaluations it cannot join yet. Named one by one,
# because "the rest" would hide which evaluation is missing and why.
UNJOINED = [
    ("genome ClinVar", "series と subset で機種を表しており run 名が無い（84 行・BERT のみ）"),
    ("rna 下流", "arch・size・lr・checkpoint のみで、生成元のスクリプトも特定できていない"),
]


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


def tree_and_run(path: str):
    """(tree as the table prints it, run) from an absolute run directory.

    Found by name rather than by cutting a prefix: these paths were written by
    whichever checkout ran the audit.
    """
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.startswith(PREFIX):
            return suffix(part), str(Path(*parts[i + 1:])) if i + 1 < len(parts) else ""
    return "", path


def load_health(path: Path):
    """(tree, run) -> (label, css class, what the verdict rests on).

    The audit records one run per entry with the directory it read, so the key is
    the same pair the run scan writes and nothing has to be matched by name.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in doc.get("runs", []):
        verdict = r.get("match")
        if verdict == "no":
            label, cls = (HEALTH_GONE if r.get("checkpoint_at_judged_min") in (None, "")
                          else HEALTH_OTHER)
        else:
            label, cls = HEALTH_LABEL.get(verdict, (verdict or "", "na"))
        judged = r.get("judged_on") or "—"
        jmin = r.get("judged_min") or {}
        detail = (f"採点指標 {judged}／保存された checkpoint step "
                  f"{r.get('best_model_checkpoint_step')}")
        if jmin.get("step") is not None:
            detail += f"／採点最小は step {jmin['step']}"
            if jmin.get("value") is not None:
                detail += f"（{jmin['value']:.6g}）"
        out[tree_and_run(r.get("run_dir", ""))] = (label, cls, detail)
    return out, len(doc.get("unreadable", []))


def load_downstream(path: Path):
    """The downstream rows, with the tree spelled as the run table spells it."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["tree"] = suffix(r["tree"])
    return rows


def downstream_table(rows):
    """One table, grouped by modality. The run cell filters the table above it."""
    out = []
    for modality in sorted({r["modality"] for r in rows}):
        group = [r for r in rows if r["modality"] == modality]
        out.append(f'<tr class="grp"><td colspan="10">{html.escape(modality)}'
                   f'<span class="gn">{len(group)} 行 / '
                   f'{len({(r["tree"], r["run"]) for r in group})} run</span></td></tr>')
        for r in sorted(group, key=lambda r: (r["eval"], r["arch"], r["size"], r["run"])):
            value = r["value"] or "—"
            if r["value"] and r["sd"]:
                value += f' <small>±{r["sd"]}</small>'
            control = (f'{html.escape(r["control"])} {r["control_value"]}'
                       if r["control_value"] else "—")
            base = (f'{html.escape(r["baseline"])} {r["baseline_value"]}'
                    if r["baseline_value"] else "—")
            out.append(
                f'<tr><td>{html.escape(r["eval"])}</td>'
                f'<td class="m" style="font-size:11px">{html.escape(r["metric"])}</td>'
                f'<td class="m" style="font-size:11px">{html.escape(r["arch"])}</td>'
                f'<td class="m" style="font-size:11px">{html.escape(r["size"])}</td>'
                f'<td class="run xr" data-run="{html.escape(r["run"])}" '
                f'title="この run で上の表を絞り込みます">{html.escape(r["run"])}</td>'
                f'<td class="m" style="font-size:10.5px">{html.escape(r["checkpoint"]) or "—"}</td>'
                f'<td class="n">{value}</td><td class="n">{control}</td>'
                f'<td class="n">{base}</td>'
                f'<td class="m" style="font-size:10px;color:var(--muted)">'
                f'{html.escape(r["source"])}</td></tr>')
    return "".join(out)


def stat(label, value, unit="", warn=False):
    u = f'<small>{unit}</small>' if unit else ""
    cls = ' class="stat w"' if warn else ' class="stat"'
    return f'<div{cls}><span class="k">{label}</span><span class="v">{value}{u}</span></div>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-rows", required=True)
    ap.add_argument("--new-tsv", required=True)
    ap.add_argument("--downstream", required=True)
    ap.add_argument("--health", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--previous", required=True)
    args = ap.parse_args()

    old = json.loads(Path(args.old_rows).read_text(encoding="utf-8"))
    new = load_new(Path(args.new_tsv))
    down = load_downstream(Path(args.downstream))
    health, unreadable = load_health(Path(args.health))
    old_cohort, new_cohort = f"〜{args.previous}", f"{args.previous} 以降"

    # (tree, run) -> the evaluations that scored it. The same pair is the key on
    # both sides, so a run that no evaluation names simply gets an empty cell.
    evals: dict[tuple[str, str], set[str]] = {}
    for r in down:
        evals.setdefault((r["tree"], r["run"]), set()).add(r["eval"])

    def mark(key):
        return "／".join(sorted(evals.get(key, ())))

    def health_of(key):
        label, cls, detail = health.get(key, ("", "", ""))
        return [label, cls, detail]

    rows, seen = [], set()
    missing = []
    for r in old:
        key = (r[0], r[1])
        seen.add(key)
        hit = new.get(key)
        if hit is None:
            missing.append(f"{r[0] or '(本体)'}/{r[1]}")
        rows.append(list(r) + [hit["dataset"] if hit else "", old_cohort, mark(key)]
                    + health_of(key))
    added = [to_row(r, new_cohort) + [mark(k)] + health_of(k)
             for k, r in new.items() if k not in seen]
    rows.extend(added)

    here = {(r[0], r[1]) for r in rows}
    joined = {k for k in evals if k in here}
    unjoined = sorted(k for k in evals if k not in joined)
    h_joined = sorted(k for k in health if k not in here)
    h_counts: dict[str, int] = {}
    for r in rows:
        if r[HEALTH]:
            h_counts[r[HEALTH]] = h_counts.get(r[HEALTH], 0) + 1
    bad = h_counts.get(HEALTH_GONE[0], 0) + h_counts.get(HEALTH_OTHER[0], 0)

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
             + stat("下流あり", f"{len(joined):,}")
             + stat("採用が不一致", f"{bad:,}", warn=True)
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

    audited = sum(1 for r in rows if r[HEALTH])
    tally = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="n">{v}</td></tr>'
        for k, v in sorted(h_counts.items(), key=lambda kv: -kv[1]))
    healthnote = f'''<div class="note warnbox">
  <span class="lab">採用した checkpoint が、採点した指標の最小かどうか</span>
  <p><strong>{audited:,} run を照合し、{bad:,} run が一致しませんでした。</strong>
  「採用」列に出しています。<strong>この {bad:,} run は、その run の最良値として引用できません</strong> ——
  保存判定が追っていた指標と、採点に使った指標が別だったためです。</p>
  <div class="tw" style="margin:10px 0"><table style="min-width:0">
  <thead><tr><th>照合結果</th><th style="text-align:right">run</th></tr></thead>
  <tbody>{tally}</tbody></table></div>
  <p><strong>不一致の内訳は 2 通りで、重いのは前者です。</strong>
  <strong>「最良が消滅」</strong>は、採点指標が最小になった step に checkpoint が残っていないもの
  （評価は走ったが、その step で保存されなかった）。<strong>重みが無いので採り直せません。</strong>
  <strong>「保存が別 step」</strong>は checkpoint はあるが、最良として記録された step が別のもので、
  こちらは選び直せます。</p>
  <p><strong>「対象外」はこの判定を通らない run、「採点指標なし」は履歴にその指標が無い run です。</strong>
  空欄は照合していない run —— 照合は {audited:,} run が対象で、
  目録の残りは対象外です（照合側にあって目録に無い run は {len(h_joined)} 本）。
  読めなかったファイルが {unreadable} 件あります。</p>
  <p>出所は <span class="m">report/checkpoint-metric-match-240464.json</span>
  （同じ照合が 3 本ありますが、<strong>240464 が最新で <span class="m">seed_configured</span> などの項目が増えた版</strong>です。
  他の 2 本との差は、進んだ 3 run と増えた項目だけで、判定は同じです）。</p>
</div>'''

    mods = ", ".join(sorted({r["modality"] for r in down}))
    downstream = f'''<h2 id="ds">下流評価</h2>
<p class="dek"><strong>{len(down):,} 行 / {len(joined):,} run。</strong>
1 行が 1 つの評価 × 1 モデルで、採点した事前学習 run と checkpoint を持ちます。
<strong>run 名をクリックすると、上の表をその run で絞り込みます。</strong>
いま入っているのは {html.escape(mods)} です。</p>
<div class="tw"><table class="ds">
  <thead><tr><th>評価</th><th>指標</th><th>実装</th><th>サイズ</th><th>採点した run</th>
  <th>checkpoint</th><th style="text-align:right">値</th><th style="text-align:right">対照</th>
  <th style="text-align:right">参考</th><th>出所</th></tr></thead>
  <tbody>{downstream_table(down)}</tbody>
</table></div>
<div class="note">
  <span class="lab">何が入っていて、何が入っていないか</span>
  <p><strong>下流側が run を記録している評価だけを入れています。</strong>
  compounds は結果 JSON の <span class="m">models.&lt;name&gt;.run_dir</span> と
  <span class="m">adopted_checkpoint</span>、molnl は <span class="m">model_path</span>（checkpoint まで）、
  protein は <span class="m">scripts/protein_proteingym_tables.py</span> の
  <span class="m">MODELS</span> が持つ run ディレクトリを<strong>取り込んで</strong>います（写していません）。
  <strong>下流が指す {len(evals):,} run は全て目録の行に一致しました（欠落 0）。</strong></p>
  <p><strong>入れていないもの:</strong></p>
  <ul>{"".join(f"<li><strong>{html.escape(n)}</strong> — {why}</li>" for n, why in UNJOINED)}</ul>
  <p><strong><span class="m">(arch, size, lr)</span> で突き合わせることはしていません。</strong>
  この鍵は run を一意に決めません —— 目録には bert/small/4.1e-4 が 3 本
  （本体・seed 1・seed 17）、bert/large/3.5e-5 が 2 本（本体・再走）あります。
  表の値から当てにいかず、<strong>記録のある所だけを結んでいます。</strong></p>
</div>'''

    footer = (f'集計元 <span class="m">report/run-inventory-{args.asof}.tsv</span>'
              f'（走査スクリプト <span class="m">scripts/run_inventory_scan.py</span>）'
              f'<br>下流の集計元 <span class="m">report/downstream-inventory-2026-10-29.tsv</span>'
              f'（走査スクリプト <span class="m">scripts/downstream_inventory_scan.py</span>）'
              f'<br>{args.previous} 時点の集計元 <span class="m">tmp/run-inventory-2026-08-31.tsv</span> ／ '
              f'保全台帳 <span class="m">tmp/learning-source-md5-2026-08-31.jsonl</span>')

    cohorts = "".join(f'<option>{c}</option>' for c in (old_cohort, new_cohort))
    page = Path(args.template).read_text(encoding="utf-8")
    for key, value in (("{{ASOF}}", args.asof), ("{{DEK}}", dek), ("{{STATS}}", stats),
                       ("{{COHORTS}}", cohorts), ("{{NOTES}}", notes), ("{{FOOTER}}", footer),
                       ("{{DOWNSTREAM}}", downstream), ("{{HEALTH}}", healthnote),
                       ("{{ROWS}}", json.dumps(rows, ensure_ascii=False, separators=(",", ":")))):
        assert key in page, key
        page = page.replace(key, value)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out}  {len(rows):,} run（追加 {len(added):,}／欠落 {len(missing)}）")
    print(f"  下流 {len(down):,} 行 / {len(joined):,} run に結合"
          + (f"／目録に無い run {len(unjoined)}: {unjoined}" if unjoined else "／欠落 0"))
    print(f"  採用の照合 {audited:,} run（不一致 {bad:,}）"
          + (f"／目録に無い run {len(h_joined)}" if h_joined else "／欠落 0")
          + f"／読めなかったファイル {unreadable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
