"""細胞種の判別の結果を 1 行 1 測定点の TSV にする。

5 つの系列を 1 つの表に入れる。系列をまたいで値を並べるときに、どれがどれだか
分からなくなるのを防ぐため、series 列を先頭に置く。

``現行本番``      いま走っている BERT の格子。採点は eval_loss_mask
``9エポック``     9 エポックまで走らせた GPT-2
``3エポック``     3 エポックの格子（2026-09-28 に報告した 13 点）
``未学習の幹``    重みを学習させずに同じプローブへ通したもの。4 サイズ x 3 seed
``対照``          モデルを使わない。遺伝子発現をそのまま線形に使う

**系列をまたいだ比較には、その系列の損失が同じ意味を持つか確かめてから入ること。**
BERT の eval_loss_mask はマスクした位置だけを、GPT-2 の val_loss は全トークンを
採点する。loss 列は系列の中で比べるためのものであり、arch をまたいで引き算する
ためのものではない。

損失は曲線の TSV から引く。同じ数が 2 つのファイルに別経路で入ると、必ずどこかで
食い違う。
"""
import argparse
import csv
import glob
import json
import os
import re

COLUMNS = ["series", "arch", "size", "lr", "checkpoint", "loss", "loss_metric",
           "accuracy", "ci_low", "ci_high", "f1_macro", "n_class", "n_cell"]
LR_NUM = {"6e4": 6e-4, "3e4": 3e-4, "1p5e4": 1.5e-4, "1p2e3": 1.2e-3,
          "7p5e5": 7.5e-5, "1p7e4": 1.7e-4, "2p3e4": 2.3e-4}


def load_curve(path):
    """(run, step, metric) -> value。損失を引くための索引。"""
    idx = {}
    if not path or not os.path.exists(path):
        return idx
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            idx[(r["run"], int(r["step"]), r["metric"])] = r["value"]
    return idx


def metrics_of(p):
    j = json.load(open(p))
    m = j["metrics"]
    ci = (j.get("details", {}).get("bootstrap_ci_95") or {}).get("accuracy") or {}
    return (m["accuracy"], ci.get("ci_lo", ""), ci.get("ci_hi", ""), m["f1_macro"],
            m.get("n_classes_scored", ""), m.get("n_cells_scored", ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True, help="eval/ の根")
    ap.add_argument("--curve-tsv", default="", help="損失を引く曲線 TSV")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    curve = load_curve(a.curve_tsv)
    rows = []

    def loss(run, step, metric):
        return curve.get((run, step, metric), "")

    # 現行本番 BERT
    for d in sorted(glob.glob(f"{a.eval_root}/probe-bert-2026-10-22/*/")):
        n = os.path.basename(d.rstrip("/"))
        m = re.match(r"bert-(small|medium|large|xl)-lr([0-9a-z]+)-ck(\d+)$", n)
        if not m:
            continue
        run, step = f"rna-{m.group(1)}-lr{m.group(2)}", int(m.group(3))
        rows.append(["現行本番", "bert", m.group(1), LR_NUM.get(m.group(2), m.group(2)),
                     step, loss(run, step, "eval_loss_mask"), "eval_loss_mask",
                     *metrics_of(os.path.join(d, "metrics.json"))])

    # 9 エポック GPT-2
    for d in sorted(glob.glob(f"{a.eval_root}/probe-9ep-2026-10-22/*/")):
        n = os.path.basename(d.rstrip("/"))
        m = re.match(r"gpt2-(small|medium|large|xl)-lr([0-9a-z]+)-9ep-final$", n)
        if not m:
            continue
        run, step = f"rna-gpt2-{m.group(1)}-lr{m.group(2)}-9ep", 120960
        rows.append(["9エポック", "gpt2", m.group(1), LR_NUM.get(m.group(2), m.group(2)),
                     step, loss(run, step, "val_loss"), "val_loss",
                     *metrics_of(os.path.join(d, "metrics.json"))])

    # 3 エポック GPT-2
    for d in sorted(glob.glob(f"{a.eval_root}/probe/grid40320/*/")):
        n = os.path.basename(d.rstrip("/"))
        m = re.match(r"(small|medium|large|xl)-lr([0-9a-z]+)$", n)
        if not m:
            continue
        run, step = f"rna-gpt2-{m.group(1)}-lr{m.group(2)}", 40320
        rows.append(["3エポック", "gpt2", m.group(1), LR_NUM.get(m.group(2), m.group(2)),
                     step, loss(run, step, "val_loss"), "val_loss",
                     *metrics_of(os.path.join(d, "metrics.json"))])

    # 未学習の幹。学習していないので lr も checkpoint も無い。seed は size の後に置かず、
    # size 列に書き足さない -- 列を増やさずに済ませると、どの seed の行か読めなくなる。
    for d in sorted(glob.glob(f"{a.eval_root}/floor-2026-09-25/*/")):
        n = os.path.basename(d.rstrip("/"))
        m = re.match(r"(small|medium|large|xl)-seed(\d+)$", n)
        if not m:
            continue
        rows.append([f"未学習の幹(seed{m.group(2)})", "gpt2", m.group(1), "", "", "", "",
                     *metrics_of(os.path.join(d, "metrics.json"))])

    # モデルを使わない対照
    for d in sorted(glob.glob(f"{a.eval_root}/control-2026-09-25/*/")):
        n = os.path.basename(d.rstrip("/"))
        rows.append([f"対照({n})", "", "", "", "", "", "",
                     *metrics_of(os.path.join(d, "metrics.json"))])

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(r)
    missing = sum(1 for r in rows if r[6] and not r[5])
    print(f"  {len(rows)} 行を {a.out} に書いた")
    if missing:
        print(f"  損失を引けなかった行 {missing} 件（曲線 TSV に該当 step が無い）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
