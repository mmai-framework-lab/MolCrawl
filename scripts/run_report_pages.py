"""One page per run: when it ran, on what code, over what data.

The inventory is a table, and a table has room for a run's identity but not for
its conditions. A run's ``run_manifest.json`` holds those -- the commit it ran
on, the dataset it opened, how the global batch was composed, what the objective
masked, which metric decided saving, and for every setting whether the value was
the default or an override. This renders one page per run from it.

Only runs that have a manifest get a page: for the rest the conditions were
never written down and a page would be a page of blanks. The pages are ordered
by start time and ``--limit`` takes the most recent, because the set grows and
the recent runs are the ones being argued about.

    python scripts/run_report_pages.py --roots <config.json> --runs-tsv <tsv> \
        --health <json> --downstream <tsv> --out-dir <dir> --limit 30

Paths live in the inputs, not here: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

PREFIX = "learning_source"
STRIPPABLE = re.compile(r"^/tmp/[^/]+/")   # a dataset staged onto node-local NVMe


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def slug(tree: str, run: str) -> str:
    """A file name that still reads as the run it belongs to."""
    s = f"{tree[len(PREFIX):] or 'base'}__{run}".strip("_")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower() + ".html"


def shorten(path: str) -> str:
    """Drop the part of a path that says only where on disk this ran.

    Kept from the run tree where there is one, otherwise from the checkout --
    which worktree ran a job is worth reading, the directory above it is not,
    and these pages are published.
    """
    if not path:
        return ""
    p = STRIPPABLE.sub("", path)
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.startswith(PREFIX) or part.startswith("MolCrawl"):
            return str(Path(*parts[i:]))
    return Path(p).name


def dig(obj, *path, default=None):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


CSS = """
:root{--ground:#f3f4f6;--surface:#fff;--surface-2:#e8eaee;--surface-3:#f7f8fa;
--ink:#161a1f;--ink-2:#3b424b;--muted:#69707a;--line:#d9dde3;--line-2:#c0c6ce;
--accent:#26506e;--accent-w:#e5edf4;--warn:#9c3f31;--warn-w:#f7e6e3;
--live:#2b6a45;--live-w:#e3efe8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#0f1215;--surface:#181c21;--surface-2:#212730;--surface-3:#1c2128;
--ink:#e6eaef;--ink-2:#c3cad3;--muted:#8d95a0;--line:#293038;--line-2:#3b434e;
--accent:#79aed4;--accent-w:#152734;--warn:#e28a7c;--warn-w:#31201e;
--live:#75bd97;--live-w:#152a1f}}
:root[data-theme="dark"]{--ground:#0f1215;--surface:#181c21;--surface-2:#212730;
--surface-3:#1c2128;--ink:#e6eaef;--ink-2:#c3cad3;--muted:#8d95a0;--line:#293038;
--line-2:#3b434e;--accent:#79aed4;--accent-w:#152734;--warn:#e28a7c;--warn-w:#31201e;
--live:#75bd97;--live-w:#152a1f}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-size:14px;line-height:1.7;
font-family:"Hiragino Kaku Gothic ProN","Hiragino Sans","Yu Gothic","Noto Sans JP",system-ui,sans-serif;
font-feature-settings:"palt" 1}
.m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:60rem;margin:0 auto;padding:0 16px 70px}
.mast{padding:30px 0 16px;border-bottom:2px solid var(--ink);margin-bottom:18px}
.eyebrow{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.16em;
text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 10px}
h1{margin:0 0 8px;font-size:clamp(19px,3.4vw,25px);line-height:1.3;font-weight:600;
font-family:ui-monospace,monospace;word-break:break-all}
.dek{margin:0;color:var(--ink-2);font-size:13.5px}
.back{display:inline-block;margin:0 0 14px;font-size:12.5px;color:var(--accent)}
h2{margin:26px 0 8px;font-size:15px;font-weight:600;padding-bottom:5px;
border-bottom:1px solid var(--line-2)}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--surface);
border:1px solid var(--line);border-radius:3px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
th{width:15rem;color:var(--muted);font-weight:600;font-size:11.5px;background:var(--surface-3)}
td{word-break:break-all}
thead th{width:auto;font-size:10px;letter-spacing:.06em;text-transform:uppercase;
background:var(--surface-2);border-bottom:1px solid var(--line-2)}
.b{display:inline-block;font-size:10px;font-weight:700;padding:1px 7px;border-radius:2px;
white-space:nowrap;letter-spacing:.03em}
.b.ok{background:var(--live-w);color:var(--live)}
.b.bad{background:var(--warn-w);color:var(--warn)}
.b.na{background:var(--surface-2);color:var(--muted)}
.b.ov{background:var(--accent-w);color:var(--accent)}
.note{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--surface);
border-radius:0 3px 3px 0;padding:11px 15px;margin:14px 0 0;font-size:12.5px;line-height:1.7}
.note.w{border-left-color:var(--warn)}
.note p{margin:0 0 7px;max-width:64ch}.note p:last-child{margin:0}
footer{border-top:1px solid var(--line);padding-top:14px;margin-top:26px;color:var(--muted);
font-size:11px;line-height:1.7}
"""


def kv(rows) -> str:
    """A definition table, skipping anything the manifest did not record."""
    body = "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>"
                   for k, v in rows if v not in (None, "", "None"))
    return f"<table><tbody>{body}</tbody></table>" if body else \
        '<p style="color:var(--muted);font-size:12.5px">記録がありません。</p>'


def overrides_table(sources: dict) -> str:
    """Only the settings that were not left at their default.

    What distinguishes one run from another is the overrides; printing all of
    them buries that under two dozen rows that are the same everywhere.
    """
    rows = [(k, v) for k, v in sorted(sources.items())
            if isinstance(v, dict) and v.get("from") == "overridden"]
    if not rows:
        return ('<p style="color:var(--muted);font-size:12.5px">'
                'すべて既定値です（上書きなし）。</p>')
    body = "".join(
        f'<tr><td class="m">{esc(k)}</td><td class="m">{esc(v.get("value"))}</td>'
        f'<td class="m" style="color:var(--muted)">{esc(v.get("default"))}</td></tr>'
        for k, v in rows)
    return ("<table><thead><tr><th>設定</th><th>この run の値</th><th>既定値</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def page(row, man, health, down, asof) -> str:
    fw = man.get("framework") or ("nanoGPT" if row["arch"] == "nanoGPT" else "HF Trainer")
    git = dig(man, "run", "git", default={}) or {}
    dirty = git.get("dirty")
    commit = (f'<span class="m">{esc(git.get("commit"))}</span>'
              + (f' <span class="m" style="color:var(--muted)">{esc(git.get("branch"))}</span>'
                 if git.get("branch") else "")
              + (' <span class="b bad">作業ツリーに未コミットあり</span>' if dirty else
                 ' <span class="b ok">clean</span>' if dirty is False else ""))

    batch = man.get("batch") or {}
    per_dev = batch.get("per_device_train_batch_size", batch.get("batch_size"))
    accum = batch.get("gradient_accumulation_steps",
                      batch.get("gradient_accumulation_steps_configured"))
    comp = (f'{esc(per_dev)} × {esc(accum)} × {esc(batch.get("world_size"))} = '
            f'<strong>{esc(batch.get("effective_global_batch"))}</strong> 系列'
            if per_dev is not None else esc(batch.get("effective_global_batch")))
    if batch.get("matches_expected_global_batch") is False:
        comp += (' <span class="b bad">宣言値 '
                 f'{esc(batch.get("expected_global_batch"))} と不一致</span>')

    sched = man.get("schedule") or {}
    steps = sched.get("max_steps", sched.get("max_iters"))
    obj = man.get("objective") or {}
    ev = man.get("eval") or {}
    sel = man.get("selection") or {}
    data = man.get("data") or {}
    place = man.get("placement") or {}
    seed = man.get("seed") or {}

    staged = data.get("staged_to_local_nvme")
    dataset = shorten(data.get("dataset_dir") or "")
    if staged or (data.get("dataset_dir") or "").startswith("/tmp/"):
        dataset += ' <span class="b ov">ノードローカル NVMe へ退避して読み出し</span>'

    label, cls, detail = health or ("", "", "")
    health_html = (f'<span class="b {cls or "na"}">{esc(label)}</span>'
                   f'<br><span style="color:var(--muted);font-size:11.5px">{esc(detail)}</span>'
                   if label else '<span class="b na">未照合</span>')

    ds_rows = "".join(
        f'<tr><td>{esc(d["eval"])}</td><td class="m">{esc(d["metric"])}</td>'
        f'<td class="m">{esc(d["value"] or "—")}</td>'
        f'<td class="m" style="color:var(--muted)">'
        f'{esc((d["control"] + " " + d["control_value"]).strip() or "—")}</td>'
        f'<td class="m" style="font-size:11px">{esc(d["checkpoint"] or "—")}</td></tr>'
        for d in down)
    ds_html = (
        "<table><thead><tr><th>評価</th><th>指標</th><th>値</th><th>対照</th>"
        f"<th>checkpoint</th></tr></thead><tbody>{ds_rows}</tbody></table>"
        if ds_rows else
        '<p style="color:var(--muted);font-size:12.5px">この run を採点した下流評価は'
        'ありません（記録があるものに限る）。</p>')

    resumed = dig(man, "run", "resumed_from_step") or dig(man, "run", "resumed_from_iter")

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(row["run"])}</title>
<style>{CSS}</style></head><body>
<div class="wrap">
<a class="back" href="../index.html">← 学習 run 一覧へ戻る</a>
<header class="mast">
  <p class="eyebrow">MolCrawl · run の条件 · {esc(asof)} 時点</p>
  <h1>{esc(row["run"])}</h1>
  <p class="dek"><span class="m">{esc(row["tree"])}</span> ／ {esc(fw)} ／
  走行 {esc(row["start"])} 〜 {esc(row["end"])}（{esc(row["hours"])} h）</p>
</header>

<h2>いつ、どこで走ったか</h2>
{kv([
    ("manifest を書いた時刻", f'<span class="m">{esc(man.get("written"))}</span>'),
    ("開始", f'<span class="m">{esc(row["start"])}</span>'),
    ("終了", f'<span class="m">{esc(row["end"])}</span>'),
    ("実時間（壁時計）", f'{esc(row["hours"])} h <span style="color:var(--muted)">'
                   f'（GPU-h ではありません）</span>'),
    ("ジョブ", f'<span class="m">{esc(dig(man, "run", "job_id"))}</span>'),
    ("ノード", f'<span class="m">{esc(dig(man, "run", "node"))}</span>'
               f' ／ {esc(place.get("gpus_allocated"))} GPU'
               f' ／ world_size {esc(place.get("world_size"))}'),
    ("途中から再開", f'step {esc(resumed)} から' if resumed else None),
])}

<h2>どの実装で走ったか</h2>
{kv([
    ("フレームワーク", esc(fw)),
    ("commit", commit),
    ("出力先", f'<span class="m">{esc(shorten(dig(man, "run", "output_dir") or dig(man, "run", "out_dir") or ""))}</span>'),
    ("configurator", f'<span class="m">{esc(shorten(dig(man, "introduced", "configurator_path") or ""))}</span>'),
    ("モデルの大きさ", esc(dig(man, "introduced", "model_size"))),
])}
<h2>既定から変えた設定</h2>
{overrides_table(man.get("sources") or {})}

<h2>何を読んだか</h2>
{kv([
    ("データセット", f'<span class="m">{dataset}</span>'),
    ("学習データの行数", f'{int(dig(man, "data", "rows", "train")):,} 行'
        if isinstance(dig(man, "data", "rows", "train"), int) else None),
    ("検証データの行数", f'{int(dig(man, "data", "rows", "eval")):,} 行'
        if isinstance(dig(man, "data", "rows", "eval"), int) else None),
    ("行数の出どころ", esc(data.get("rows_from"))),
    ("learning_source", f'<span class="m">{esc(shorten(data.get("learning_source_dir") or ""))}</span>'),
])}

<h2>どう回したか</h2>
{kv([
    ("グローバルバッチの構成", comp),
    ("系列長", f'{esc(batch.get("seq_len"))} トークン'),
    ("1 step のトークン数", f'{int(batch["tokens_per_step"]):,} トークン'
        if isinstance(batch.get("tokens_per_step"), int) else None),
    ("step 数", f'{int(steps):,}' if isinstance(steps, int) else esc(steps)),
    ("学習率", f'<span class="m">{esc(sched.get("learning_rate"))}</span>'),
    ("warmup", esc(sched.get("warmup_steps", sched.get("warmup_iters")))),
    ("減衰", esc(sched.get("lr_scheduler_type"))
        or (f'lr_decay_iters {esc(sched.get("lr_decay_iters"))} ／ '
            f'min_lr {esc(sched.get("min_lr"))}' if sched.get("lr_decay_iters") else None)),
    ("seed", f'<span class="m">{esc(seed.get("seed"))}</span>'
             + (f' ／ data_seed <span class="m">{esc(seed.get("data_seed"))}</span>'
                if seed.get("data_seed") is not None else "")
             + (f' ／ ランク別 {esc(seed.get("per_rank_offset"))}'
                if seed.get("per_rank_offset") else "")),
])}

<h2>何を学ばせたか</h2>
{kv([
    ("目的関数", esc(obj.get("task"))),
    ("マスク率", esc(obj.get("mlm_probability"))),
    ("collator", esc(obj.get("collator"))),
    ("文書境界トークン", f'id {esc(dig(obj, "document_boundary", "id"))} ／ '
        f'1 ブロックあたり {esc(dig(obj, "document_boundary", "per_block"))} 個'
        if dig(obj, "document_boundary", "id") is not None else None),
])}

<h2>どう採点し、どれを残したか</h2>
{kv([
    ("検証に使った行数", f'{int(ev["rows"]):,} 行' if isinstance(ev.get("rows"), int) else None),
    ("検証行の取り方", "無作為抽出（seed " + esc(ev.get("subset_seed")) + "）"
        if ev.get("subset_random") else "先頭から"),
    ("採点指標", f'<span class="m">{esc(ev.get("judge_on") or sel.get("judge_on"))}</span>'),
    ("保存判定の指標", f'<span class="m">{esc(sel.get("metric_for_best_model"))}</span>'),
    ("評価の間隔", esc(sel.get("eval_steps"))),
    ("保存の間隔", esc(sel.get("save_steps"))),
    ("保持方針", f'{esc(dig(sel, "retention", "mode"))}'
        f'（最良 {esc(dig(sel, "retention", "keep_best"))} + 最新 '
        f'{esc(dig(sel, "retention", "keep_latest"))}）'
        if dig(sel, "retention", "mode") else None),
    ("採用 checkpoint の照合", health_html),
])}

<h2>下流評価</h2>
{ds_html}

<footer>この run の <span class="m">run_manifest.json</span> から生成。
照合は <span class="m">report/checkpoint-metric-match-240464.json</span>、
下流は <span class="m">report/downstream-inventory-2026-10-29.tsv</span>。<br>
値は run 自身の申告で、config ファイルの値ではありません。</footer>
</div></body></html>
"""


def load_health(path: Path):
    out = {}
    for r in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
        parts = Path(r.get("run_dir", "")).parts
        for i, part in enumerate(parts):
            if part.startswith(PREFIX):
                key = (part, str(Path(*parts[i + 1:])) if i + 1 < len(parts) else "")
                verdict = r.get("match")
                if verdict == "yes":
                    label, cls = "一致", "ok"
                elif verdict == "no":
                    label, cls = (("最良が消滅", "bad")
                                  if r.get("checkpoint_at_judged_min") in (None, "")
                                  else ("保存が別 step", "bad"))
                elif verdict == "no judged metric in history":
                    label, cls = "採点指標なし", "na"
                else:
                    label, cls = "対象外", "na"
                jmin = r.get("judged_min") or {}
                detail = f'採点指標 {r.get("judged_on") or "—"}／保存 step {r.get("best_model_checkpoint_step")}'
                if jmin.get("step") is not None:
                    detail += f'／採点最小 step {jmin["step"]}'
                out[key] = (label, cls, detail)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", required=True, help="the run-inventory config naming the trees")
    ap.add_argument("--runs-tsv", required=True)
    ap.add_argument("--health", required=True)
    ap.add_argument("--downstream", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", required=True, help="JSON mapping (tree, run) -> page path")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    roots = {r["tree"]: Path(r["path"])
             for r in json.loads(Path(args.roots).read_text(encoding="utf-8"))["roots"]}
    rows = list(csv.DictReader(open(args.runs_tsv, encoding="utf-8"), delimiter="\t"))
    health = load_health(Path(args.health))
    down: dict[tuple[str, str], list] = {}
    for d in csv.DictReader(open(args.downstream, encoding="utf-8"), delimiter="\t"):
        down.setdefault((d["tree"], d["run"]), []).append(d)

    # Most recent first, and only runs whose conditions were written down.
    cand = [r for r in rows if "run_manifest" in r["params_rec"]]
    cand.sort(key=lambda r: r["start"], reverse=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index, written, skipped = {}, 0, []
    for r in cand[:args.limit]:
        root = roots.get(r["tree"])
        mf = (root / r["run"] / "run_manifest.json") if root else None
        if mf is None or not mf.exists():
            skipped.append(f'{r["tree"]}/{r["run"]}')
            continue
        try:
            man = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f'{r["tree"]}/{r["run"]} ({exc.__class__.__name__})')
            continue
        name = slug(r["tree"], r["run"])
        key = (r["tree"], r["run"])
        (out_dir / name).write_text(
            page(r, man, health.get(key), down.get(key, []), args.asof), encoding="utf-8")
        index[f'{r["tree"]}\t{r["run"]}'] = name
        written += 1

    Path(args.index).write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print(f"wrote {written} 本（manifest を持つ run {len(cand)} 本のうち直近 {args.limit}）")
    print(f"  期間 {cand[min(args.limit, len(cand)) - 1]['start']} 〜 {cand[0]['start']}")
    if skipped:
        print(f"  manifest を読めず飛ばした run {len(skipped)}: {skipped[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
