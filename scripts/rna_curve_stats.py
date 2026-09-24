"""曲線の TSV から、最小点と末尾の直線あてはめを出す。

図の読みは主観が混じるので、図を描いたのと同じ TSV から数で出す。

最小点: 系列全体の最小とその step。報告済みの値と照らすためのもの。
末尾: 末尾 n 点に最小二乗で直線をあて、傾きと残差の標準偏差を出す。残差が
run 間の差より大きければ、その差は揺らぎに埋もれていて読み取れない。
"""
import argparse
import csv
from collections import defaultdict


def fit_line(xs, ys):
    """最小二乗の (傾き, 切片, 残差の標準偏差)。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    inter = my - slope * mx
    res = [y - (slope * x + inter) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sd = (sum(r * r for r in res) / dof) ** 0.5
    return slope, inter, sd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, help="曲線を描いたときの TSV")
    ap.add_argument("--tail-points", type=int, default=50, help="直線をあてる末尾の点数")
    ap.add_argument("--gap", type=float, default=0.012,
                    help="この差が揺らぎに埋もれるかを見る。run 間の差")
    ap.add_argument("--only", default="", help="この文字列を含む run だけ見る")
    a = ap.parse_args(argv)

    runs = defaultdict(list)
    with open(a.tsv, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        head = next(rd)
        vcol = 4
        for row in rd:
            if len(row) <= vcol:
                continue
            key = (row[1], row[2])
            runs[key].append((int(row[3]), float(row[vcol])))
    if not runs:
        raise SystemExit(f"{a.tsv} に系列が無い")
    print(f"  列 {head}  値の列 {head[vcol]}  run {len(runs)}")

    print("\n  最小点 (TSV の全点から)")
    print(f"    {'run':<20}{'点':>6}{'最小':>10}{'その step':>12}{'最終':>10}")
    for key in sorted(runs):
        rows = sorted(runs[key])
        if a.only and a.only not in f"{key[0]}-{key[1]}":
            continue
        bi = min(range(len(rows)), key=lambda i: rows[i][1])
        print(f"    {key[0]+' lr'+key[1]:<20}{len(rows):>6}{rows[bi][1]:>10.4f}"
              f"{rows[bi][0]:>12,}{rows[-1][1]:>10.4f}")

    print(f"\n  末尾 {a.tail_points} 点に直線をあてた残差 (差 {a.gap} と比べる)")
    print(f"    {'run':<20}{'step の幅':>18}{'傾き/1k step':>14}"
          f"{'残差 sd':>10}{'差/残差':>9}  判定")
    for key in sorted(runs):
        rows = sorted(runs[key])[-a.tail_points:]
        if a.only and a.only not in f"{key[0]}-{key[1]}":
            continue
        if len(rows) < 3:
            print(f"    {key[0]+' lr'+key[1]:<20}  点が {len(rows)} しかない")
            continue
        xs = [float(s) for s, _ in rows]
        ys = [v for _, v in rows]
        slope, _inter, sd = fit_line(xs, ys)
        ratio = a.gap / sd if sd else float("inf")
        call = "差は残差に埋もれる" if ratio < 2 else "差は残差より大きい"
        print(f"    {key[0]+' lr'+key[1]:<20}{rows[0][0]:>8,}-{rows[-1][0]:<9,}"
              f"{slope*1000:>+14.5f}{sd:>10.5f}{ratio:>9.2f}  {call}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
