"""格子の損失の曲線を、元の数値から描く。

図だけ出しても引き直せないので、先に TSV を書き、図はその TSV から描く。

区間をまたぐ run では、再開が直前の checkpoint から始まるため step が巻き戻り、
同じ step が 2 度現れる。後の区間の値を採る -- 巻き戻した分は捨てられた計算で、
その後に続くのは後の区間だからである。重複した範囲は図に縦線で示す。
"""
import argparse
import csv
import glob
import os
import re


def read_point(d):
    """1 点ぶんの (step, train, val, 区間番号) を、区間の順に読む。"""
    rows = []
    for seg, f in enumerate(sorted(glob.glob(f"{d}/logging_*.csv")), 1):
        with open(f, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) < 3 or not r[0].strip().isdigit():
                    continue
                rows.append((int(r[0]), float(r[1]), float(r[2]), seg))
    # 同じ step が複数の区間にあれば後の区間を採る
    by_step = {}
    for step, tr, va, seg in rows:
        if step not in by_step or seg >= by_step[step][2]:
            by_step[step] = (tr, va, seg)
    return [(s, *by_step[s]) for s in sorted(by_step)]


def seg_starts(point_rows):
    """区間が切り替わった step。図の縦線に使う。"""
    out, prev = [], None
    for step, _, _, seg in point_rows:
        if prev is not None and seg != prev:
            out.append(step)
        prev = seg
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, help="格子の出力の根")
    ap.add_argument("--out-dir", required=True, help="TSV と図の置き場")
    ap.add_argument("--arch", default="gpt2")
    ap.add_argument("--metric", default="validation loss", help="縦軸の名前")
    ap.add_argument("--tail-frac", type=float, default=0.2, help="拡大図が見る末尾の割合")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    order = {"small": 0, "medium": 1, "large": 2, "xl": 3}
    lr_num = {"6e4": 6e-4, "3e4": 3e-4, "1p5e4": 1.5e-4, "1e4": 1e-4,
              "1p7e4": 1.7e-4, "1e3": 1e-3, "7p5e5": 7.5e-5}

    points = {}
    for d in sorted(glob.glob(f"{a.runs_root}/*")):
        m = re.search(r"-(small|medium|large|xl)-lr(\w+)$", os.path.basename(d))
        if not m or not glob.glob(f"{d}/logging_*.csv"):
            continue
        points[(m.group(1), m.group(2))] = read_point(d)
    if not points:
        raise SystemExit(f"no logging_*.csv under {a.runs_root}")
    print(f"  点 {len(points)}")

    tsv = os.path.join(a.out_dir, f"{a.arch}-loss-curves.tsv")
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["arch", "size", "learning_rate", "step", "loss", "segment"])
        for (size, lr), rows in sorted(points.items(), key=lambda k: (order[k[0][0]], -lr_num.get(k[0][1], 0))):
            for step, _tr, va, seg in rows:
                w.writerow([a.arch, size, lr_num.get(lr, lr), step, f"{va:.6f}", seg])
    print(f"  WROTE {tsv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({s for s, _ in points}, key=lambda s: order[s])
    lrs = sorted({lr for _, lr in points}, key=lambda x: -lr_num.get(x, 0))
    lo = min(min(v for _, _, v, _ in r) for r in points.values())
    hi = max(max(v for _, _, v, _ in r[len(r)//10:]) for r in points.values())
    pad = (hi - lo) * 0.08

    def draw(groups, title, fname, xlim=None, ylim=None, mark=True):
        n = len(groups)
        fig, axes = plt.subplots(n, 1, figsize=(7.2, 3.0 * n), squeeze=False)
        for ax, (label, series) in zip(axes[:, 0], groups):
            for name, rows in series:
                xs = [s for s, _, _, _ in rows]
                ys = [v for _, _, v, _ in rows]
                ln, = ax.plot(xs, ys, lw=1.2, label=name)
                if mark:
                    bi = min(range(len(ys)), key=lambda i: ys[i])
                    ax.plot(xs[bi], ys[bi], "o", ms=5, color=ln.get_color())
                    ax.annotate(f"{xs[bi]:,}", (xs[bi], ys[bi]), fontsize=7,
                                xytext=(0, -11), textcoords="offset points",
                                ha="center", color=ln.get_color())
                for s in seg_starts(rows):
                    ax.axvline(s, color="0.6", lw=0.8, ls=":")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel(a.metric)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, ncol=2)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_ylim(*(ylim or (lo - pad, hi + pad)))
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = os.path.join(a.out_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  WROTE {p}")

    # 1: サイズごとに 1 枚、学習率を重ねる
    draw([(f"{s}", [(f"lr {lr_num.get(lr, lr):.1e}", points[(s, lr)])
                    for lr in lrs if (s, lr) in points]) for s in sizes],
         f"{a.arch}: {a.metric} by size", f"{a.arch}-by-size.png")
    # 2: 学習率ごとに 1 枚、サイズを重ねる
    draw([(f"lr {lr_num.get(lr, lr):.1e}", [(s, points[(s, lr)])
                                            for s in sizes if (s, lr) in points]) for lr in lrs],
         f"{a.arch}: {a.metric} by learning rate", f"{a.arch}-by-lr.png")
    # 3: 末尾の拡大。0.012 の動きが見える縦軸に絞る
    maxstep = max(r[-1][0] for r in points.values())
    x0 = int(maxstep * (1 - a.tail_frac))
    tail = [v for r in points.values() for s, _, v, _ in r if s >= x0]
    t_lo, t_hi = min(tail), max(tail)
    tpad = (t_hi - t_lo) * 0.06
    draw([(f"{s}  (last {int(a.tail_frac*100)}%)",
           [(f"lr {lr_num.get(lr, lr):.1e}", points[(s, lr)]) for lr in lrs if (s, lr) in points])
          for s in sizes],
         f"{a.arch}: last {int(a.tail_frac*100)}% of training", f"{a.arch}-tail.png",
         xlim=(x0, maxstep), ylim=(t_lo - tpad, t_hi + tpad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
