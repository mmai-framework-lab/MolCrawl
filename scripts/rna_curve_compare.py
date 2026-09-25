"""曲線の TSV から、run どうしを同じ step の窓で比べる。

最小値どうしを比べてはいけない。最小は数百点から選んだ点で、選んだ時点で下に
偏っているうえ、偏りの大きさが run によって違う。同じ step の窓の平均で比べる。

揺れが run 間で共通なら、差を取ると揺れが消える。対応のある差として扱い、
差の系列そのものの散らばりから標準誤差を出す。run ごとの散らばりから出すと、
消えたはずの揺れを二重に数えて t を小さく見積もる。
"""
import argparse
import csv
from collections import defaultdict


def load(path):
    runs = defaultdict(dict)
    with open(path, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd)
        for r in rd:
            if len(r) > 4:
                runs[(r[1], r[2])][int(r[3])] = float(r[4])
    return runs


def window_mean(series, at, half):
    """step=at の前後 half の平均と、その点数。範囲に点が無ければ None。"""
    vals = [v for s, v in series.items() if abs(s - at) <= half]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def paired(a, b, lo):
    """step lo 以降の対応のある差。(平均, 標準誤差, 点数)。"""
    common = sorted(set(a) & set(b))
    common = [s for s in common if s >= lo]
    if len(common) < 3:
        return None
    d = [a[s] - b[s] for s in common]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    # 隣り合う点は相関する。有効な点数を落とす。
    res = [x - m for x in d]
    num = sum(x * y for x, y in zip(res, res[1:]))
    den = sum(x * x for x in res)
    r1 = max(min(num / den if den else 0.0, 0.95), 0.0)
    n_eff = n * (1 - r1) / (1 + r1)
    return m, sd / n_eff ** 0.5 if n_eff > 0 else float("inf"), n, r1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--at", default="", help="窓の中心にする step。コンマ区切り")
    ap.add_argument("--half", type=int, default=2000, help="窓の半分の幅")
    ap.add_argument("--from-step", type=int, default=0, help="対応のある差を取る下限")
    ap.add_argument("--pairs", default="", help="比べる組。パイプ区切りの 2 つをコンマで並べる")
    a = ap.parse_args(argv)

    runs = load(a.tsv)
    keys = sorted(runs)
    print(f"  run {len(keys)}")

    if a.at:
        ats = [int(x) for x in a.at.split(",")]
        print(f"\n  窓の平均 (前後 {a.half:,} step)")
        print(f"    {'run':<22}" + "".join(f"{s:>12,}" for s in ats))
        for k in keys:
            cells = []
            for s in ats:
                m, n = window_mean(runs[k], s, a.half)
                cells.append(f"{m:>12.4f}" if m is not None else f"{'未到達':>12}")
            print(f"    {k[0]+' lr'+k[1]:<22}" + "".join(cells))

    if a.pairs:
        print(f"\n  対応のある差 (step {a.from_step:,} 以降)")
        print(f"    {'組':<34}{'差':>10}{'標準誤差':>11}{'t':>9}{'点':>6}{'相関':>7}")
        for spec in a.pairs.split(","):
            left, right = (s.strip() for s in spec.split("|"))
            ka = tuple(left.split())
            kb = tuple(right.split())
            if ka not in runs or kb not in runs:
                print(f"    {spec:<34}  {'無い run がある'}")
                continue
            out = paired(runs[ka], runs[kb], a.from_step)
            if not out:
                print(f"    {spec:<34}  共通の step が足りない")
                continue
            m, se, n, r1 = out
            t = m / se if se else 0.0
            print(f"    {spec:<34}{m:>+10.4f}{se:>11.5f}{t:>9.1f}{n:>6}{r1:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
