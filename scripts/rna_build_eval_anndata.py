"""評価用 AnnData と JSONL を 1 度の census 読みから作る（未承認の作り置き。前提を読むこと）。

================================ 前提ブロック ================================
この版は次を仮定している。上長の判断が違う場合、対応する箇所を直してから走らせる。

 A. 対象は肺アトラス 093d3bfe... の primary 細胞 193,108（全体）
      → 肺実質のみに絞る判断なら --tissue-filter parenchyma
      → 別データセットなら --dataset
 B. test 側 3 名は A41, A43, A47
      → 既定値を置いていない。--test-donors で必ず渡す（8/28 §6.1 の確認待ちのため）
 C. 分割はドナー単位（細胞単位で切らない）
      → これは 8/26 判断で確定済み
 D. JSONL の切り出しは上位 1,024 遺伝子
      → GPT-2 の位置埋め込みが 1,024 なのでモデル側の制約。変える余地はない
      → HVG 対科の遺伝子数は別（--hvg-n）
 E. 語彙は Geneformer の 25,426（token_dictionary.pkl）
      → 学習時と同じもの。変えない

未確定なので既定値を置いていないもの:
 - --test-donors（B）
 - --hvg-n（同条件 1,024 か実用条件 2,000 か、上長の判断待ち。両方作るなら 2 回走らせる）

出力は 2 つ。どちらも cell_id = soma_joinid で対応が取れる。
  eval.h5ad   カウント行列 / obs 4 列 / 語彙内遺伝子の印 / 正規化層 / X_pca
  eval.jsonl  1 行 1 細胞（cell_id, tokens, cell_type, tissue）

実行はまだしていない。承認後、A〜E を突き合わせてから投入すること。
=============================================================================

なぜ 1 度の読みで両方作るか: JSONL と AnnData を別々に取ると、census の版や
フィルタが食い違ったときに気付けない。同じ obs から両方を書けば、cell_id の
対応は構成上ずれない。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def LOG(*a):
    print(*a, flush=True)


AIRWAY = {"bronchus", "trachea"}
OBS_KEEP = ["donor_id", "tissue", "suspension_type", "cell_type"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="093d3bfe-6f0f-4ac0-a7a1-829f94d0a49f")
    ap.add_argument("--test-donors", required=True,
                    help="カンマ区切り。既定を置いていないのは §6.1 の確認待ちのため")
    ap.add_argument("--tissue-filter", choices=["all", "parenchyma"], default="all")
    ap.add_argument("--top-n-genes", type=int, default=1024,
                    help="JSONL の切り出し。GPT-2 の位置埋め込みが 1,024")
    ap.add_argument("--hvg-n", type=int, required=True,
                    help="HVG 対科の遺伝子数。同条件 1024 / 実用条件 2000")
    ap.add_argument("--hvg-within-vocab", action="store_true",
                    help="HVG を語彙 25,426 の範囲内から選ぶ（同条件）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab",
                    default="molcrawl/data/rna/dataset/geneformer/token_dictionary.pkl")
    ap.add_argument("--census-version", default="latest")
    a = ap.parse_args(argv)

    LOG("=== 前提ブロックを読んだか確認すること（ファイル先頭）===")
    test_donors = [d.strip() for d in a.test_donors.split(",") if d.strip()]
    LOG(f"ARGS dataset={a.dataset} test_donors={test_donors} tissue={a.tissue_filter} "
        f"top_n={a.top_n_genes} hvg_n={a.hvg_n} within_vocab={a.hvg_within_vocab}")

    import cellxgene_census as cc
    import scanpy as sc
    import scipy.sparse as sp

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    LOG("--- census から 1 度だけ読む ---")
    with cc.open_soma(census_version=a.census_version) as c:
        ad = cc.get_anndata(
            c, organism="Homo sapiens",
            obs_value_filter=f"dataset_id == '{a.dataset}' and is_primary_data == True",
            obs_column_names=["soma_joinid"] + OBS_KEEP,
            var_column_names=["soma_joinid", "feature_id"],
        )
    LOG(f"CELLS {ad.n_obs:,} GENES {ad.n_vars:,}")

    if a.tissue_filter == "parenchyma":
        keep = ~ad.obs["tissue"].astype(str).isin(AIRWAY)
        ad = ad[keep.to_numpy()].copy()
        LOG(f"CELLS after parenchyma filter: {ad.n_obs:,}")

    # --- 分割はドナー単位。細胞単位で切ると同じドナーが両側に入る ---
    donors = ad.obs["donor_id"].astype(str)
    unknown = sorted(set(test_donors) - set(donors.unique()))
    if unknown:
        raise SystemExit(f"--test-donors に無いドナー: {unknown}\n"
                         f"このデータセットのドナー: {sorted(donors.unique())}")
    ad.obs["split"] = np.where(donors.isin(test_donors), "test", "train")
    LOG("SPLIT " + " ".join(
        f"{k}={int(v):,}" for k, v in ad.obs['split'].value_counts().items()))
    for col in ["suspension_type", "tissue"]:
        LOG(f"  {col} の内訳:\n" +
            "\n".join("    " + line for line in
                      __import__("pandas").crosstab(
                          ad.obs["split"], ad.obs[col].astype(str)).to_string().splitlines()))
    ad.obs["region"] = np.where(ad.obs["tissue"].astype(str).isin(AIRWAY), "airway", "parenchyma")

    # --- 語彙内の印。同条件の HVG 選択と、被覆率の記録に使う ---
    vocab = set(pickle.load(open(a.vocab, "rb")))
    ad.var["in_vocab"] = ad.var["feature_id"].astype(str).isin(vocab).to_numpy()
    LOG(f"VOCAB genes_in_vocab={int(ad.var['in_vocab'].sum()):,} / {ad.n_vars:,}")

    # --- 正規化層と PCA。pcr_comparison と HVG 選択の前段 ---
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.layers["lognorm"] = ad.X.copy()

    # HVG は train 側のドナーだけで選ぶ。全体から選ぶと、選択の段階で
    # test の細胞を見たことになる。
    tr = ad[ad.obs["split"] == "train"]
    pool = tr[:, ad.var["in_vocab"].to_numpy()] if a.hvg_within_vocab else tr
    LOG(f"HVG train_cells={tr.n_obs:,} pool_genes={pool.n_vars:,} n={a.hvg_n}")
    hv = sc.pp.highly_variable_genes(pool.copy(), n_top_genes=a.hvg_n, inplace=False)
    chosen = set(pool.var_names[hv["highly_variable"].to_numpy()])
    ad.var["hvg"] = ad.var_names.isin(chosen)
    LOG(f"HVG selected={int(ad.var['hvg'].sum()):,}")

    sc.pp.pca(ad, n_comps=50, mask_var="hvg")
    LOG(f"PCA X_pca {ad.obsm['X_pca'].shape}")

    h5 = out / "eval.h5ad"
    ad.write_h5ad(h5, compression="gzip")
    LOG(f"WROTE {h5} ({h5.stat().st_size/1e9:.2f} GB)")

    # --- JSONL。順位は生カウントで付ける（正規化前）---
    LOG("--- JSONL ---")
    # 学習時と同じ辞書で ENSG -> トークン id。語彙に無い遺伝子は -1 にして落とす。
    vd = pickle.load(open(a.vocab, "rb"))
    gene_tok = np.array([int(vd.get(str(f), -1)) for f in ad.var["feature_id"]], dtype=np.int64)

    X = ad.layers["counts"]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    ids = ad.obs["soma_joinid"].astype(str).to_numpy()
    cts = ad.obs["cell_type"].astype(str).to_numpy()
    tis = ad.obs["tissue"].astype(str).to_numpy()
    spl = ad.obs["split"].to_numpy()

    jl = out / "eval.jsonl"
    written = skipped = 0
    with jl.open("w", encoding="utf-8") as fh:
        for i in range(X.shape[0]):
            s, e = X.indptr[i], X.indptr[i + 1]
            if e == s:
                skipped += 1
                continue
            cols, vals = X.indices[s:e], X.data[s:e]
            order = np.argsort(-vals)
            toks = [int(t) for t in gene_tok[cols[order]] if t >= 0][:a.top_n_genes]
            if not toks:
                skipped += 1
                continue
            fh.write(json.dumps({"cell_id": ids[i], "tokens": toks,
                                 "cell_type": cts[i], "tissue": tis[i],
                                 "split": str(spl[i])}, ensure_ascii=False) + "\n")
            written += 1
    LOG(f"WROTE {jl} rows={written:,} skipped={skipped:,}")
    LOG("RESULT cell_id は soma_joinid。h5ad の obs と 1 対 1 で対応する")
    return 0


if __name__ == "__main__":
    sys.exit(main())
