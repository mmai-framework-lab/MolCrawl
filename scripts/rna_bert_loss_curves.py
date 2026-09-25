"""BERT 格子の損失の曲線を、元の数値から描く。

GPT-2 側 (rna_loss_curves.py) は run 直下の logging_*.csv を読むが、BERT は
HF Trainer なので系列は checkpoint の trainer_state.json の log_history にある。
slurm ログの eval も読めるが、あちらは epoch が小数第 2 位までで step が付かず、
step 単位の横軸が作れない。区間の切れ目だけは slurm ログから取る。

図だけ出しても引き直せないので、先に TSV を書き、図はその TSV から描く。

崩れた run と学習できた run は同じ図に重ねない。縦軸の幅が 3 桁違い、重ねると
学習できた run の差が潰れるからである。崩れた run の図には基準線を水平に引く。

区間をまたぐ run では再開が直前の checkpoint から始まるため step が巻き戻り、
同じ step が 2 度現れる。log_history は時系列なので後に現れたものを採る。
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


def lr_label(tag):
    v = LR_NUM.get(tag)
    return f"lr {v:.1e}" if v else f"lr {tag}"


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
    for path in sorted(glob.glob(f"{log_dir}/mc-bert-rna-{size}-lr{lr_tag}-s*.out")):
        last = None
        with open(path, errors="replace") as fh:
            for chunk in fh:
                for m in re.finditer(r"\|\s*(\d+)/(\d+)\s*\[", chunk):
                    last = int(m.group(1))
        if last:
            ends.append(last)
    return [e for e in ends[:-1] if e > 0]  # 最後の区間の終端は切れ目ではない


def segment_of(step, ends):
    """その step がどの区間に属するか。切れ目が取れなければ 1。"""
    n = 1
    for e in ends:
        if step > e:
            n += 1
    return n


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
    ap.add_argument("--tail-frac", type=float, default=0.2, help="拡大図が見る末尾の割合")
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

    seg_ends = {k: segment_ends(a.log_dir, k[0], k[1]) for k in series}

    # 末尾 10 点の中央値で崩れたかを判ずる。1 点だけの跳ねで振り分けないため。
    verdict = {}
    for key, rows in series.items():
        tail = sorted(v for _, v in rows[-10:])
        med = tail[len(tail) // 2]
        verdict[key] = ("collapsed" if med >= collapse_at else "learning", med)

    ordered = sorted(series, key=lambda k: (ORDER[k[0]], -LR_NUM.get(k[1], 0)))
    print(f"  run {len(series)}  崩れた判定の閾値 {collapse_at:.3f}")
    for key in ordered:
        rows, (kind, med) = series[key], verdict[key]
        bi = min(range(len(rows)), key=lambda i: rows[i][1])
        print(f"    {key[0]:<7} lr{key[1]:<6} 点 {len(rows):>5}  "
              f"最終 step {rows[-1][0]:>7,}  末尾中央値 {med:7.3f}  "
              f"最小 {rows[bi][1]:7.3f} @ {rows[bi][0]:>7,}  "
              f"区間の切れ目 {seg_ends[key] or 'なし'}  {kind}")

    tsv = os.path.join(a.out_dir, "bert-loss-curves.tsv")
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["arch", "size", "learning_rate", "step", "loss", "segment"])
        for key in ordered:
            for step, val in series[key]:
                w.writerow(["bert", key[0], LR_NUM.get(key[1], key[1]),
                            step, f"{val:.6f}", segment_of(step, seg_ends[key])])
    print(f"  WROTE {tsv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(keys, by, title, fname, baseline=None, tail=False):
        """by='size' ならサイズごとに 1 枚で学習率を重ね、'lr' なら逆にする。"""
        if not keys:
            print(f"  該当する run が無いので {fname} は描かない")
            return
        want_size = by == "size"
        if want_size:
            panels = sorted({k[0] for k in keys}, key=lambda s: ORDER[s])
        else:
            panels = sorted({k[1] for k in keys}, key=lambda t: -LR_NUM.get(t, 0))

        def pick(p):
            """その枠に載る run を、枠の中で並ぶ順に返す。"""
            if want_size:
                return sorted((k for k in keys if k[0] == p),
                              key=lambda k: -LR_NUM.get(k[1], 0))
            return sorted((k for k in keys if k[1] == p), key=lambda k: ORDER[k[0]])

        def name(key):
            """凡例に出す名。重ねている側を書く。"""
            return lr_label(key[1]) if want_size else key[0]

        def head(p):
            """枠の見出し。分けている側を書く。"""
            return p if want_size else lr_label(p)

        fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 3.0 * len(panels)),
                                 squeeze=False)
        for ax, p in zip(axes[:, 0], panels):
            ks = pick(p)
            pmax = max(series[k][-1][0] for k in ks)
            if tail:
                x0 = int(pmax * (1 - a.tail_frac))
            elif baseline is None:
                # 学習できた run の図は縦軸を末尾側に絞る。絞った縦軸から
                # はみ出す線を残さないよう、収まる step まで横軸を切る。
                base = int(pmax * 0.2)
                hi_band = max(v for k in ks for s, v in series[k] if s >= base)
                over = [s for k in ks for s, v in series[k] if v > hi_band]
                x0 = max(over) if over else 0
            else:
                # 崩れた run の図は、崩れるところを外せないので全体を入れる。
                x0 = 0
            shown = [v for k in ks for s, v in series[k] if s >= x0]
            lo, hi = min(shown), max(shown)
            if baseline is not None:
                lo, hi = min(lo, baseline), max(hi, baseline)
            pad = (hi - lo) * 0.08 or 0.05
            for key in ks:
                rows = series[key]
                ln, = ax.plot([s for s, _ in rows], [v for _, v in rows],
                              lw=1.2, label=name(key))
                bi = min(range(len(rows)), key=lambda i: rows[i][1])
                ax.plot(rows[bi][0], rows[bi][1], "o", ms=5, color=ln.get_color())
                ax.annotate(f"{rows[bi][0]:,}", rows[bi], fontsize=7,
                            xytext=(0, -11), textcoords="offset points",
                            ha="center", color=ln.get_color())
                for s in seg_ends.get(key, []):
                    ax.axvline(s, color="0.6", lw=0.8, ls=":")
            if baseline is not None:
                ax.axhline(baseline, color="crimson", lw=1.0, ls="--")
                ax.annotate(f"baseline {baseline:g}", (0.995, baseline),
                            xycoords=("axes fraction", "data"), fontsize=7,
                            ha="right", va="bottom", color="crimson")
            ax.set_title(head(p) + (f"  (last {int(a.tail_frac*100)}%)" if tail else ""),
                         fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel(a.metric)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, ncol=2)
            ax.set_xlim(x0, pmax)
            ax.set_ylim(lo - pad, hi + pad)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = os.path.join(a.out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  WROTE {path}")

    pct = int(a.tail_frac * 100)
    for kind, base in (("learning", None), ("collapsed", a.baseline)):
        keys = [k for k in series if verdict[k][0] == kind]
        word = "learned" if kind == "learning" else "collapsed"
        draw(keys, "size", f"bert: {a.metric} by size, runs that {word}",
             f"bert-{kind}-by-size.png", baseline=base)
        draw(keys, "lr", f"bert: {a.metric} by learning rate, runs that {word}",
             f"bert-{kind}-by-lr.png", baseline=base)
        draw(keys, "size", f"bert: last {pct}% of training, runs that {word}",
             f"bert-{kind}-tail.png", baseline=base, tail=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
