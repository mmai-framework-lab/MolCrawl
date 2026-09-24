"""BERT 格子の損失の曲線を、元の数値から描く。

GPT-2 側 (rna_loss_curves.py) は run 直下の logging_*.csv を読むが、BERT は
HF Trainer なので系列は checkpoint の trainer_state.json の log_history にある。
step が付くのはこちらだけで、.out のほうは epoch が小数第 2 位までしか出ない。
図だけ出しても引き直せないので、先に TSV を書き、図はその TSV から描く。

崩れた run と学習できた run は同じ図に重ねない。縦軸の幅が 3 桁違い、重ねると
学習できた run の差が潰れるからである。崩れた run の図には基準線を水平に引く。

区間をまたぐ run では再開が直前の checkpoint から始まるため step が巻き戻り、
同じ step が 2 度現れる。log_history は時系列なので後に現れたものを採る。
区間の切れ目は、直前の区間のログが止まった step を縦線で示す。
"""
import argparse
import csv
import glob
import json
import os
import re

RUN_RE = re.compile(r"-(small|medium|large|xl)-lr(\w+)$")
LR_NUM = {"6e4": 6e-4, "3e4": 3e-4, "1p7e4": 1.7e-4, "1p5e4": 1.5e-4,
          "1e4": 1e-4, "1e3": 1e-3, "7p5e5": 7.5e-5}
ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}


def latest_state(run_dir):
    """その run で最後に保存された trainer_state.json。無ければ None。"""
    best, best_step = None, -1
    for p in glob.glob(f"{run_dir}/checkpoint-*/trainer_state.json"):
        m = re.search(r"checkpoint-(\d+)", p)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


def read_series(state_path, metric):
    """(step -> 値) を、後から現れたものが勝つ形で読む。"""
    with open(state_path) as fh:
        state = json.load(fh)
    by_step = {}
    for rec in state.get("log_history", []):
        if metric in rec and "step" in rec:
            by_step[int(rec["step"])] = float(rec[metric])
    return [(s, by_step[s]) for s in sorted(by_step)]


def segment_ends(log_dir, size, lr_tag):
    """各区間のログが止まった step。次の区間はそこから再開している。"""
    if not log_dir:
        return []
    ends = []
    pat = f"{log_dir}/mc-bert-rna-{size}-lr{lr_tag}-s*.out"
    for path in sorted(glob.glob(pat)):
        last = None
        with open(path, errors="replace") as fh:
            for chunk in fh:
                for m in re.finditer(r"\|\s*(\d+)/(\d+)\s*\[", chunk):
                    last = int(m.group(1))
        if last:
            ends.append(last)
    return [e for e in ends[:-1] if e > 0]  # 最後の区間の終端は切れ目ではない


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, help="BERT の出力の根")
    ap.add_argument("--out-dir", required=True, help="TSV と図の置き場")
    ap.add_argument("--log-dir", default="", help="区間の切れ目を取る slurm ログの置き場")
    ap.add_argument("--metric", default="eval_loss_mask")
    ap.add_argument("--baseline", type=float, default=9.372,
                    help="崩れた run の図に引く水平の基準線")
    ap.add_argument("--collapse-at", type=float, default=None,
                    help="末尾がこの値以上の run を崩れたとみなす。既定は基準線の 95%%")
    a = ap.parse_args(argv)

    collapse_at = a.collapse_at if a.collapse_at is not None else a.baseline * 0.95
    os.makedirs(a.out_dir, exist_ok=True)

    series, skipped = {}, []
    for d in sorted(glob.glob(f"{a.runs_root}/*")):
        m = RUN_RE.search(os.path.basename(d))
        if not m:
            continue
        state = latest_state(d)
        if not state:
            skipped.append((os.path.basename(d), "checkpoint の trainer_state.json が無い"))
            continue
        rows = read_series(state, a.metric)
        if not rows:
            skipped.append((os.path.basename(d), f"log_history に {a.metric} が無い"))
            continue
        series[(m.group(1), m.group(2))] = rows
    for name, why in skipped:
        print(f"  SKIP {name}: {why}")
    if not series:
        raise SystemExit(f"{a.runs_root} の下に読める run が無い")

    # 末尾 10 点の中央値で崩れたかを判ずる。1 点だけの跳ねで振り分けないため。
    verdict, tail_n = {}, 10
    for key, rows in series.items():
        tail = sorted(v for _, v in rows[-tail_n:])
        med = tail[len(tail) // 2]
        verdict[key] = ("collapsed" if med >= collapse_at else "learning", med)

    print(f"  run {len(series)}  崩れた判定の閾値 {collapse_at:.3f}")
    for key in sorted(series, key=lambda k: (ORDER[k[0]], -LR_NUM.get(k[1], 0))):
        rows, (kind, med) = series[key], verdict[key]
        bi = min(range(len(rows)), key=lambda i: rows[i][1])
        print(f"    {key[0]:<7} lr{key[1]:<6} 点 {len(rows):>5}  "
              f"最終 step {rows[-1][0]:>7,}  末尾中央値 {med:7.3f}  "
              f"最小 {rows[bi][1]:7.3f} @ {rows[bi][0]:>7,}  {kind}")

    seg = {k: segment_ends(a.log_dir, k[0], k[1]) for k in series}

    tsv = os.path.join(a.out_dir, "bert-loss-curves.tsv")
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["arch", "size", "learning_rate", "step", a.metric, "verdict"])
        for key in sorted(series, key=lambda k: (ORDER[k[0]], -LR_NUM.get(k[1], 0))):
            for step, val in series[key]:
                w.writerow(["bert", key[0], LR_NUM.get(key[1], key[1]),
                            step, f"{val:.6f}", verdict[key][0]])
    print(f"  WROTE {tsv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(keys, title, fname, baseline=None):
        if not keys:
            print(f"  該当する run が無いので {fname} は描かない")
            return
        sizes = sorted({s for s, _ in keys}, key=lambda s: ORDER[s])
        vals = [v for k in keys for _, v in series[k]]
        lo, hi = min(vals), max(vals)
        if baseline is not None:
            lo, hi = min(lo, baseline), max(hi, baseline)
        pad = (hi - lo) * 0.08 or 0.1
        fig, axes = plt.subplots(len(sizes), 1, figsize=(7.2, 3.0 * len(sizes)),
                                 squeeze=False)
        for ax, size in zip(axes[:, 0], sizes):
            for key in sorted((k for k in keys if k[0] == size),
                              key=lambda k: -LR_NUM.get(k[1], 0)):
                rows = series[key]
                lr = LR_NUM.get(key[1], key[1])
                label = f"lr {lr:.1e}" if isinstance(lr, float) else f"lr {lr}"
                ln, = ax.plot([s for s, _ in rows], [v for _, v in rows],
                              lw=1.2, label=label)
                bi = min(range(len(rows)), key=lambda i: rows[i][1])
                ax.plot(rows[bi][0], rows[bi][1], "o", ms=5, color=ln.get_color())
                ax.annotate(f"{rows[bi][0]:,}", rows[bi], fontsize=7,
                            xytext=(0, -11), textcoords="offset points",
                            ha="center", color=ln.get_color())
                for s in seg.get(key, []):
                    ax.axvline(s, color="0.6", lw=0.8, ls=":")
            if baseline is not None:
                ax.axhline(baseline, color="crimson", lw=1.0, ls="--")
                ax.annotate(f"baseline {baseline:g}", (0.995, baseline), xycoords=
                            ("axes fraction", "data"), fontsize=7, ha="right",
                            va="bottom", color="crimson")
            ax.set_title(size, fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel(a.metric)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, ncol=2)
            ax.set_ylim(lo - pad, hi + pad)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = os.path.join(a.out_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  WROTE {p}")

    learning = [k for k in series if verdict[k][0] == "learning"]
    collapsed = [k for k in series if verdict[k][0] == "collapsed"]
    draw(learning, f"bert: {a.metric}, runs that learned", "bert-learning.png")
    draw(collapsed, f"bert: {a.metric}, runs that collapsed", "bert-collapsed.png",
         baseline=a.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
