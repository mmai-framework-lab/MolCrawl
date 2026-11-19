"""Where the rna 2026-11-18 extension stands, in the terms the judgment asks for.

rna-judgment-2026-11-18c wants, for every point, the minimum and how far it sits
below the nine-point moving average around it, on both sides of the resume: the
nine-epoch run and the extension. It also wants the size of the evaluation's own
up-and-down measured per arm rather than borrowed from another one -- xl does not
move the way the smaller sizes do.

So this reports, per run:

  the lowest value in the extension and the step it fell on
  the nine-point moving average around that step, and how many points the window
      actually holds -- a minimum near either end of the series cannot have nine
  the difference between the two
  whether the extension has gone below the nine-epoch minimum, and by how much
  the standard deviation of the residual from that moving average, in this arm
  the median of the last 2,000 steps, which says where it is sitting now

and, for BERT, how many of the extension's running-minimum points still have a
checkpoint. The first evaluation after every resume is saved unconditionally --
SaveOnMetricImprovement starts its own best at None and does not restore it -- so
that one is named rather than counted as an improvement.

Reads finished files only: trainer_state.json for BERT, logging_*.csv for GPT-2.
Nothing here writes to a run directory.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

NINE_EPOCH_STEPS = 120960

# (label, directory under BERT_ROOT/rna/bert-output, the nine-epoch eval_loss_mask minimum)
BERT = [
    ("BERT small 3e-4", "rna-small-lr3e4-ext242k", 2.6928),
    ("BERT medium 3e-4", "rna-medium-lr3e4-ext242k", 2.6368),
    ("BERT large 1.7e-4", "rna-large-lr1p7e4-ext242k", 2.6349),
    ("BERT xl 2.5e-5", "rna-xl-lr2p5e5-ext242k", 3.3168),
]

# (label, directory under GPT2_ROOT, resume iter, the nine-epoch best val loss)
GPT2 = [
    ("GPT-2 small 1.2e-3", "rna-gpt2-small-lr1p2e3-9ep-ext242k", 120100, 3.1397974),
    ("GPT-2 medium 1.2e-3", "rna-gpt2-medium-lr1p2e3-9ep-ext242k", 100100, 3.1184802),
    ("GPT-2 large 6e-4", "rna-gpt2-large-lr6e4-9ep-ext242k", 119900, 3.1106853),
    ("GPT-2 xl 6e-4", "rna-gpt2-xl-lr6e4-9ep-ext242k", 119100, 3.1093848),
]


def newest_checkpoint(run_dir):
    steps = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        try:
            steps.append(int(os.path.basename(p).rsplit("-", 1)[1]))
        except (ValueError, IndexError):
            continue
    return (max(steps), sorted(steps)) if steps else (None, [])


def bert_series(run_dir):
    """(step, eval_loss_mask) from the newest checkpoint's restored history."""
    newest, steps = newest_checkpoint(run_dir)
    if newest is None:
        return [], []
    state = os.path.join(run_dir, f"checkpoint-{newest}", "trainer_state.json")
    if not os.path.exists(state):
        return [], steps
    with open(state, encoding="utf-8") as f:
        hist = json.load(f)["log_history"]
    return [(e["step"], e["eval_loss_mask"]) for e in hist if "eval_loss_mask" in e], steps


def gpt2_series(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "logging_*.csv")), key=os.path.getmtime)
    if not files:
        return []
    out = []
    with open(files[-1], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = (row.get(" val_loss") or row.get("val_loss") or "").strip()
            if value and value != "None":
                out.append((int(row["iter"]), float(value)))
    return out


def window_around(series, index, half=4):
    lo, hi = max(0, index - half), min(len(series), index + half + 1)
    return [v for _, v in series[lo:hi]]


def residual_sd(series, half=4):
    """Spread of each point around the moving average through it."""
    res = []
    for k in range(half, len(series) - half):
        window = [series[j][1] for j in range(k - half, k + half + 1)]
        res.append(series[k][1] - sum(window) / len(window))
    return (statistics.pstdev(res), len(res)) if len(res) > 1 else (None, len(res))


def running_minima(series, start_from):
    best, out = start_from, []
    for step, value in series:
        if value < best:
            best = value
            out.append(step)
    return out


def report(label, series, boundary, prior_min, saved=None):
    ext = [(s, v) for s, v in series if s > boundary]
    print(f"== {label}")
    if not ext:
        print("   延長区間の評価点なし")
        return
    print(f"   延長の評価点 {len(ext)} ({ext[0][0]} → {ext[-1][0]})")
    i = min(range(len(ext)), key=lambda k: ext[k][1])
    step, value = ext[i]
    window = window_around(ext, i)
    moving = sum(window) / len(window)
    print(f"   最小 {value:.4f} @ {step}  移動平均 {moving:.4f} ({len(window)} 点)  "
          f"差 {value - moving:+.4f}")
    gap = value - prior_min
    print(f"   9 エポックの最小 {prior_min:.4f} との差 {gap:+.4f}  "
          f"→ {'下回った' if gap < 0 else '下回っていない'}")
    sd, n = residual_sd(ext)
    print(f"   評価ごとの上下(残差の標準偏差) {'%.4f' % sd if sd else 'n/a'} ({n} 点)")
    tail = [v for s, v in ext if s > ext[-1][0] - 2000]
    print(f"   直近 {ext[-1][1]:.4f}  終盤 2,000 step の中央値 {statistics.median(tail):.4f} "
          f"({len(tail)} 点)")
    if saved is not None:
        # Two baselines, because they answer different questions. Against the
        # nine-epoch minimum: has the extension produced anything better than what
        # is already reported. Against the extension's own first point: is the save
        # decision tracking eval_loss_mask at all, which is what the judgment asked
        # to be measured and which an arm that has not beaten the old best would
        # otherwise say nothing about.
        beat = running_minima(ext, prior_min)
        own = running_minima(ext, float("inf"))
        print(f"   9 エポックの最小を下回った走行最小点 {len(beat)}"
              f"  うちディスクに有り {len([s for s in beat if s in saved])}")
        missing = [s for s in own if s not in saved]
        values = dict(ext)
        if missing:
            ranked = sorted(own, key=lambda s: values[s], reverse=True)
            where = [ranked.index(s) + 1 for s in missing]
            print(f"   延長区間の走行最小点 {len(own)}"
                  f"  うちディスクに有り {len(own) - len(missing)}")
            print(f"     無いもの {missing} は値の高い順で {where} 番目"
                  f" -- 保持方式(上位 10 件＋最新)が刈る側にあたる")
        else:
            print(f"   延長区間の走行最小点 {len(own)}  すべてディスクに有り")
        off_grid = sorted(s for s in saved if s > boundary and s % 1000 != 0)
        print(f"   周期外の保存 {off_grid}")
        on_min = [s for s in off_grid if s in own]
        print(f"     うち走行最小点 {on_min}  "
              f"({len(on_min)}/{len(off_grid)}; 再開直後の 1 件は無条件保存)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-root", default=os.environ.get("BERT_ROOT", ""))
    ap.add_argument("--gpt2-root", default=os.environ.get("GPT2_ROOT", ""))
    a = ap.parse_args()
    if not a.bert_root or not a.gpt2_root:
        raise SystemExit("set BERT_ROOT and GPT2_ROOT (or pass --bert-root/--gpt2-root)")

    for label, name, prior in BERT:
        run = os.path.join(a.bert_root, "rna", "bert-output", name)
        series, saved = bert_series(run)
        if not series:
            print(f"== {label}\n   {run} に読める checkpoint が無い")
            continue
        report(label, series, NINE_EPOCH_STEPS, prior, saved=set(saved))

    for label, name, resume, prior in GPT2:
        run = os.path.join(a.gpt2_root, name)
        report(label, gpt2_series(run), resume, prior)
    return 0


if __name__ == "__main__":
    sys.exit(main())
