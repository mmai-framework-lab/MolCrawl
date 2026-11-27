"""How the molnl batch grid is read in prose: which arms diverged, and which batch won.

Two readings of the 2026-11-20 grid are fixed by the 2026-11-27 directive, and both
are computed here so that the numbers in a report come from a run of this file rather
than from a hand count.

**Divergence.** Prose uses a different rule from the per-arm summary's ``collapsed``
column, and the directive says to keep both and to say which is which:

* this file's rule -- after the minimum, at least one evaluation point is at or above
  the model-free floor;
* the summary column's rule -- after the minimum, a point returns to at least the arm's
  own first point, or the arm ends far above its best.

The second is the stricter one in practice, because an arm that collapses from a bad
start never climbs back to where it began. On the 68 arms of this grid the first rule
names four and the second names one, which is why a report that quotes both has to
label them.

**Which batch won.** Per architecture and size: the batch with the lowest judged value,
that value, and the gap to the second best. The gap is what says whether the winner is
separated from the field, and the directive asks for it beside every winner.

The floor is read against the metric, not against the modality: BERT is judged on
``eval_loss_mask`` and its floor is the unigram at the masked positions of the valid
split, while GPT-2 is judged on ``val_loss`` and its floor is the unigram over every
token of the same split. They are different quantities and the numbers are close only
by coincidence.

    python scripts/molnl_grid_reading.py --curves <rt1-...-curves.tsv> [--out <md>]

Read-only: it reads one TSV and writes one markdown file or stdout.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io

#: The floor each judged metric is read against, measured on the valid split under one
#: pipeline on 2026-11-26 (job 399695). A train-split floor is not interchangeable with
#: either of these; 3.8638, which stood here until 2026-11-27, was one.
FLOOR = {"bert": 4.6449, "gpt2": 4.6516}
#: The metric each architecture is judged on.
JUDGED = {"bert": "eval_loss_mask", "gpt2": "val_loss"}


def read_curves(path):
    """The judged valid-split series of every arm, keyed by arm."""
    with io.open(path, encoding="utf-8-sig") as handle:
        body = [line for line in handle if not line.startswith("#")]
    series = collections.defaultdict(list)
    for row in csv.DictReader(body, delimiter="\t"):
        arch = row["arch"]
        if row["split"] != "valid" or row["metric"] != JUDGED.get(arch):
            continue
        series[(arch, row["size"], int(row["batch"]), row["lr"])].append(
            (int(row["step"]), float(row["value"])))
    return {k: sorted(v) for k, v in series.items()}


def diverged(points, floor):
    """Whether a point at or above ``floor`` follows the minimum.

    The minimum is taken at its first occurrence: a curve that sits at its minimum for
    several evaluations has not turned at the last of them.
    """
    values = [v for _, v in points]
    after = values[values.index(min(values)) + 1:]
    return any(v >= floor for v in after)


def divergence(series):
    out = []
    for key, points in sorted(series.items()):
        arch = key[0]
        if diverged(points, FLOOR[arch]):
            values = [v for _, v in points]
            after = values[values.index(min(values)) + 1:]
            out.append((key, min(values), max(after)))
    return out


def winners(series):
    """Per architecture and size: the best batch, its value, and the gap to the second.

    One cell is one (batch, rate). A batch's value is the lowest of its cells, since
    each cell tried several rates and the grid takes the best; the batch a size "won"
    at is the one holding that size's lowest value. The gap is between the two lowest
    batches, not between two rates of the same batch.
    """
    per_batch = collections.defaultdict(dict)
    for (arch, size, batch, _lr), points in series.items():
        best = min(v for _, v in points)
        have = per_batch[(arch, size)].get(batch)
        if have is None or best < have:
            per_batch[(arch, size)][batch] = best
    rows = []
    for (arch, size), by_batch in sorted(per_batch.items()):
        ranked = sorted(by_batch.items(), key=lambda kv: kv[1])
        best_batch, best_value = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        rows.append({
            "arch": arch, "size": size, "batch": best_batch, "value": best_value,
            "second_batch": second[0] if second else None,
            "second_value": second[1] if second else None,
            "gap": (second[1] - best_value) if second else None,
            "batches": len(ranked),
        })
    return rows


def render(series):
    lines = ["# molnl バッチ格子 本文の読み", ""]
    lines.append(f"arm 数 {len(series)}。判定指標は BERT が `eval_loss_mask`・"
                 f"GPT-2 が `val_loss`、床はそれぞれ {FLOOR['bert']} と {FLOOR['gpt2']}。")
    lines.append("")

    lines.append("## 1. 発散（本文の規則）")
    lines.append("")
    lines.append("規則は「**最小より後に、床以上となる評価点が 1 つ以上ある**」。")
    lines.append("per-arm summary の `collapsed` 列は別の規則の出力であり、本数が違う。")
    lines.append("")
    hits = divergence(series)
    lines.append(f"**{len(hits)} 本**（{len(series)} 本中）。")
    lines.append("")
    lines.append("| arch | size | batch | lr | 最小 | 最小より後の最大 |")
    lines.append("|---|---|---:|---|---:|---:|")
    for (arch, size, batch, lr), lo, hi in hits:
        lines.append(f"| {arch} | {size} | {batch} | {lr} | {lo:.4f} | {hi:.4f} |")
    lines.append("")

    lines.append("## 2. サイズごとに最小だった batch")
    lines.append("")
    lines.append("| arch | size | 最小の batch | 値 | 2 位の batch | 2 位の値 | 差 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in winners(series):
        gap = f"{r['gap']:.6f}" if r["gap"] is not None else ""
        sv = f"{r['second_value']:.6f}" if r["second_value"] is not None else ""
        lines.append(f"| {r['arch']} | {r['size']} | {r['batch']} | {r['value']:.6f} | "
                     f"{r['second_batch'] or ''} | {sv} | {gap} |")
    lines.append("")
    lines.append("1 GPU が一度に流す塊数はサイズごとに一定なので、この表の比較には効かない。"
                 "**ただし xl だけ他の 3 サイズと塊数が違うため、サイズを並べる読みには"
                 "系統差が乗る**（大きさは未測定）。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", required=True, help="the rt1 curves TSV of the grid")
    ap.add_argument("--out", help="write the tables here (BOM, for the report dir)")
    args = ap.parse_args(argv)

    series = read_curves(args.curves)
    if not series:
        raise SystemExit(f"no judged valid-split series in {args.curves}")
    text = render(series)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
            handle.write(text)
        print(f"-> {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
