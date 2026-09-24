"""曲線の TSV から、最小点と末尾の直線あてはめを出す。

図の読みは主観が混じるので、図を描いたのと同じ TSV から数で出す。

最小点: 系列全体の最小とその step。報告済みの値と照らすためのもの。
末尾: 末尾 n 点に最小二乗で直線をあて、傾きと残差の標準偏差を出す。残差が
run 間の差より大きければ、その差は揺らぎに埋もれていて読み取れない。

末尾だけを見ると、どの run も傾きが 0 と見分けられないことがある。窓が短ければ
傾きの標準誤差が大きくなるからで、平らだという証拠ではない。

曲線が折り返したかを、最小点から末尾への上がり幅で測ってはいけない。最小は
数百点の中から選んだ点で、選んだ時点で下に偏っており、平らな系列でも上がり幅が
出る。偏りの大きさも併せて出す。折り返しの判定は、最小の周りの窓の平均と末尾の
窓の平均を比べる。隣り合う評価点は相関するので、有効な点数を落として t を出す。
"""
import argparse
import csv
from collections import defaultdict


def fit_line(xs, ys):
    """最小二乗の (傾き, 切片, 残差の標準偏差, 傾きの標準誤差)。"""
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
    # 傾きが 0 と見分けられるかは残差の大きさだけでは決まらず、
    # 見ている step の幅にもよる。標準誤差まで出して初めて読める。
    se = sd / sxx ** 0.5 if sxx else float("inf")
    return slope, inter, sd, se


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, help="曲線を描いたときの TSV")
    ap.add_argument("--tail-points", type=int, default=50, help="直線をあてる末尾の点数")
    ap.add_argument("--gap", type=float, default=0.012,
                    help="この差が揺らぎに埋もれるかを見る。run 間の差")
    ap.add_argument("--block", type=int, default=2000,
                    help="平均を取る窓の幅。step で数える")
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

    print(f"\n  末尾 {a.tail_points} 点の直線あてはめ (run 間の差 {a.gap} と比べる)")
    print(f"    {'run':<20}{'step の幅':>18}{'傾き/1k step':>14}"
          f"{'その t':>8}{'残差 sd':>9}{'差/残差':>8}")
    tail_sd = {}
    for key in sorted(runs):
        rows = sorted(runs[key])[-a.tail_points:]
        if a.only and a.only not in f"{key[0]}-{key[1]}":
            continue
        if len(rows) < 3:
            print(f"    {key[0]+' lr'+key[1]:<20}  点が {len(rows)} しかない")
            continue
        xs = [float(s) for s, _ in rows]
        ys = [v for _, v in rows]
        slope, _inter, sd, se = fit_line(xs, ys)
        tail_sd[key] = sd
        t = slope / se if se else 0.0
        print(f"    {key[0]+' lr'+key[1]:<20}{rows[0][0]:>8,}-{rows[-1][0]:<9,}"
              f"{slope*1000:>+14.5f}{t:>8.2f}{sd:>9.5f}{a.gap/sd if sd else 0:>8.2f}")

    print(f"\n  指示 §4.2 の判定 (散らばりを {a.gap} と比べる)")
    print(f"    {'run':<20}{'散らばり':>10}{'/0.012':>9}  判定")
    for key in sorted(runs):
        sd = tail_sd.get(key)
        if not sd or (a.only and a.only not in f"{key[0]}-{key[1]}"):
            continue
        r = sd / a.gap
        call = ("折り返しは本物" if r < 0.5 else
                "折り返しとは言えない。曲線は平ら")
        print(f"    {key[0]+' lr'+key[1]:<20}{sd:>10.5f}{r:>9.2f}  {call}")

    print(f"\n  最小点は、選んだ時点で下に偏る (最小 - その周り {a.block:,} step の平均)")
    print(f"    {'run':<20}{'最小':>9}{'周りの平均':>11}{'偏り':>9}{'/散らばり':>10}")
    blocks = {}
    for key in sorted(runs):
        rows = sorted(runs[key])
        sd = tail_sd.get(key)
        if not sd or (a.only and a.only not in f"{key[0]}-{key[1]}"):
            continue
        bi = min(range(len(rows)), key=lambda i: rows[i][1])
        at = rows[bi][0]
        near = [v for s, v in rows if abs(s - at) <= a.block // 2]
        m_near = sum(near) / len(near)
        blocks[key] = (at, m_near, len(near))
        print(f"    {key[0]+' lr'+key[1]:<20}{rows[bi][1]:>9.4f}{m_near:>11.4f}"
              f"{rows[bi][1]-m_near:>+9.4f}{(rows[bi][1]-m_near)/sd:>10.2f}")

    print(f"\n  折り返したか (最小の周りと末尾、それぞれ {a.block:,} step の平均を比べる)")
    print(f"    {'run':<20}{'最小の周り':>11}{'末尾':>9}{'差':>9}"
          f"{'相関':>7}{'その t':>8}  判定")
    for key in sorted(runs):
        rows = sorted(runs[key])
        sd = tail_sd.get(key)
        if not sd or key not in blocks:
            continue
        at, m_near, n_near = blocks[key]
        end = rows[-1][0]
        late = [v for s, v in rows if s > end - a.block]
        if not late or at > end - a.block:
            print(f"    {key[0]+' lr'+key[1]:<20}  最小が末尾の窓の中にあり、比べられない")
            continue
        m_late = sum(late) / len(late)
        # 隣り合う評価点は相関する。有効な点数を落とさないと t を過大に見る。
        res = [v - m_late for v in late]
        num = sum(x * y for x, y in zip(res, res[1:]))
        den = sum(x * x for x in res)
        r1 = max(min(num / den if den else 0.0, 0.95), 0.0)
        shrink = (1 - r1) / (1 + r1)
        se = sd * (1 / (n_near * shrink) + 1 / (len(late) * shrink)) ** 0.5
        diff = m_late - m_near
        t = diff / se if se else 0.0
        call = "折り返した" if abs(t) >= 2 else "平らと区別できない"
        print(f"    {key[0]+' lr'+key[1]:<20}{m_near:>11.4f}{m_late:>9.4f}"
              f"{diff:>+9.4f}{r1:>7.2f}{t:>8.2f}  {call}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
