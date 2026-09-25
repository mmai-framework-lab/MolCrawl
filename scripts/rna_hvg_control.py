"""モデルを使わない対照。HVG の主成分で細胞種を判別する。

事前学習した埋め込みの数字だけ出しても、それが高いのか低いのか分からない。
遺伝子発現をそのまま線形に使った場合と並べて初めて位置が決まる。

2 条件ある。**同条件**は語彙 25,426 の範囲内から HVG を 1,024 個取ったもので、
モデルが見られる遺伝子と本数を揃えてあり、表現の比較になる。**実用条件**は
全遺伝子から 2,000 個取ったもので、この課題に何を使うべきかの比較になる。
どちらの h5ad も同じ細胞・同じ分割を持ち、HVG の選び方は var と X_pca にしか
効かない。

分類器と採点は評価器のものをそのまま呼ぶ。LogisticRegression の設定や少数
クラスの除き方が違うと、対照と本番が別の採点になり、並べた表が比較にならない。
"""
import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", required=True, help="対照の h5ad")
    ap.add_argument("--out-dir", required=True, help="結果の置き場")
    ap.add_argument("--label", default="cell_type", help="obs の中の細胞種の列")
    ap.add_argument("--split", default="split", help="obs の中の分割の列")
    ap.add_argument("--test-label", default="test", help="分割の列で test を表す値")
    ap.add_argument("--min-cells", type=int, default=20,
                    help="test 側でこの数に満たないクラスは採点から外す")
    a = ap.parse_args(argv)

    import warnings
    warnings.filterwarnings("ignore")
    import anndata as ad
    from sklearn.linear_model import LogisticRegression

    from molcrawl.tasks.evaluation.tabula_sapiens.metrics import (
        cell_type_metrics, drop_rare_labels,
    )

    adata = ad.read_h5ad(a.h5ad)
    if "X_pca" not in adata.obsm:
        raise SystemExit(f"{a.h5ad} に X_pca が無い。HVG から主成分を作る段が要る")
    x = np.asarray(adata.obsm["X_pca"])
    # カテゴリ型は使われていない水準も数えるので、先に落とす。
    labels = adata.obs[a.label]
    if hasattr(labels, "cat"):
        labels = labels.cat.remove_unused_categories()
    y = np.asarray(labels.astype(str))
    split = np.asarray(adata.obs[a.split].astype(str))

    is_test = split == a.test_label
    is_train = ~is_test
    print(f"  {a.h5ad}")
    print(f"  細胞 {len(y):,}  主成分 {x.shape[1]}  "
          f"train {int(is_train.sum()):,}  test {int(is_test.sum()):,}")
    if not is_test.any() or not is_train.any():
        raise SystemExit(f"{a.split} の値が {set(split)} で、train と test に割れていない")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x[is_train], y[is_train])
    pred = clf.predict(x[is_test])

    kept_true, kept_pred, dropped = drop_rare_labels(y[is_test], pred,
                                                     min_count=a.min_cells)
    metrics = cell_type_metrics(np.asarray(kept_true), np.asarray(kept_pred))
    metrics["n_cells_scored"] = int(len(kept_true))
    metrics["n_classes_scored"] = int(len(set(kept_true)))
    metrics["n_classes_excluded"] = int(len(dropped))
    metrics["excluded_labels"] = {str(k): int(v) for k, v in dict(dropped).items()}
    metrics["min_cells_per_label"] = a.min_cells
    metrics["n_components"] = int(x.shape[1])
    metrics["source"] = os.path.abspath(a.h5ad)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "metrics.json")
    with open(out, "w") as fh:
        json.dump({"metrics": metrics}, fh, indent=2, ensure_ascii=False)
    print(f"  accuracy {metrics['accuracy']:.4f}  f1_macro {metrics['f1_macro']:.4f}  "
          f"クラス {metrics['n_classes_scored']}  除外 {metrics['n_classes_excluded']}")
    print(f"  WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
