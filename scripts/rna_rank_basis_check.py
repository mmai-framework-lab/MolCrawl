"""§2.1: 切り出しが何の順序で行われているか、そして「59.7%」が何を測った値かを確かめる。

事前学習側（Geneformer tokenizer.py:200）は
    X_norm = X / n_counts * 10000 / gene_median
つまり「細胞内の総数で割り、遺伝子ごとの中央値で割った値」で並べている。

評価側（prepare_jsonl.py:186 付近）は adata.X をそのまま argsort している。
この 2 つが違う空間なら、評価で渡す系列は学習時と別の順序になる。

測るもの:
  1. 3 つの空間での順序が、上位 1,024 個としてどれだけ一致するか
  2. 上位 1,024 個に入る「発現量」の割合を、空間ごとに
  3. 私が 59.7% と書いた値が実際は何だったか
"""
import pickle
import numpy as np
import scipy.sparse as sp
import cellxgene_census as cc
import argparse

def LOG(*a): print(*a, flush=True)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dataset", required=True, help="調べる census データセットの id")
ap.add_argument("--gene-median-file", required=True,
                help="学習時に使った遺伝子ごとの中央値（pickle）")
ap.add_argument("--top-n", type=int, default=1024, help="切り出し長")
ap.add_argument("--sample", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
LUNG, K, NSAMPLE, SEED = A.dataset, A.top_n, A.sample, A.seed

med = pickle.load(open(A.gene_median_file, "rb"))
LOG(f"gene_median_dictionary: {len(med):,} 遺伝子")

with cc.open_soma(census_version="latest") as c:
    ad = cc.get_anndata(c, organism="Homo sapiens",
                        obs_value_filter=f"dataset_id == '{LUNG}' and is_primary_data == True",
                        obs_column_names=["soma_joinid"],
                        var_column_names=["soma_joinid", "feature_id"])
LOG(f"取得: {ad.n_obs:,} 細胞 x {ad.n_vars:,} 遺伝子")

X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
LOG(f"census の X: dtype={X.dtype} 最大={X.max():.1f} "
    f"整数か={'はい' if np.allclose(X.data, np.rint(X.data)) else 'いいえ'}")
LOG("  → census が返すのは生カウント。prepare_jsonl はこれをそのまま argsort している")

feat = ad.var["feature_id"].astype(str).to_numpy()
in_med = np.array([f in med for f in feat])
norm = np.array([med.get(f, np.nan) for f in feat], dtype=float)
LOG(f"中央値辞書に載る遺伝子: {int(in_med.sum()):,} / {len(feat):,}")

rng = np.random.default_rng(SEED)
rows = rng.choice(X.shape[0], size=min(NSAMPLE, X.shape[0]), replace=False)
LOG(f"標本 {len(rows):,} 細胞（seed={SEED}）で測る\n")

ov_raw_lib, ov_raw_med, ov_lib_med = [], [], []
mass = {"raw": [], "lib": [], "med": []}
ngene_frac, n_all = [], []
for i in rows:
    a, b = X.indptr[i], X.indptr[i + 1]
    cols, vals = X.indices[a:b], X.data[a:b].astype(float)
    keep = in_med[cols]          # 中央値の無い遺伝子は事前学習でも落ちる
    cols, vals = cols[keep], vals[keep]
    if len(vals) < 10:
        continue
    tot = vals.sum()
    v_raw = vals
    v_lib = vals / tot * 1e4
    v_med = v_lib / norm[cols]
    n_all.append(len(vals))
    k = min(K, len(vals))
    tops = {}
    for name, v in [("raw", v_raw), ("lib", v_lib), ("med", v_med)]:
        o = np.argsort(-v)[:k]
        tops[name] = set(cols[o])
        # 「上位 k 個に入る発現量の割合」をその空間で測る
        mass[name].append(v[np.argsort(-v)[:k]].sum() / v.sum())
    ngene_frac.append(k / len(vals))
    ov_raw_lib.append(len(tops["raw"] & tops["lib"]) / k)
    ov_raw_med.append(len(tops["raw"] & tops["med"]) / k)
    ov_lib_med.append(len(tops["lib"] & tops["med"]) / k)

def f(x):

    return f"{np.mean(x)*100:.1f}%"
LOG("=== 1. 上位 1,024 個の顔ぶれが空間でどれだけ一致するか ===")
LOG(f"  生カウント順 と 総数正規化順          : {f(ov_raw_lib)}")
LOG(f"  生カウント順 と 中央値スケール順(学習時): {f(ov_raw_med)}")
LOG(f"  総数正規化順 と 中央値スケール順        : {f(ov_lib_med)}")
LOG("")
LOG("=== 2. 上位 1,024 個に入る「発現量」の割合（各空間で測る）===")
LOG(f"  生カウントで測る                : {f(mass['raw'])}")
LOG(f"  総数正規化で測る                : {f(mass['lib'])}")
LOG(f"  中央値スケールで測る（学習時の空間）: {f(mass['med'])}")
LOG("")
LOG("=== 3. 私が 59.7% と書いた値の正体 ===")
LOG(f"  上位 1,024 個 / 検出遺伝子数 の平均: {f(ngene_frac)}")
LOG(f"  （検出遺伝子 平均 {np.mean(n_all):.0f} 個）")
LOG("  → これは遺伝子の個数の割合であって、発現量の割合ではない")
