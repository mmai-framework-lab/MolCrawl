"""scib-metrics が実データで動くかを確かめる。導入の確認ではなく動作の確認。

指標の名前が API にあることと、この課題のデータで計算が通ることは別である。
評価用 AnnData の一部を取り、X_pca を埋め込みに見立てて Benchmarker を回し、
どの指標が値を返し、どれが落ちるかを一覧にする。

silhouette 系は使わない（2026-09-07 指示）。既定に入っているので明示的に外す。
"""
import argparse
import sys
import traceback


def LOG(*a):
    print(*a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", required=True, help="評価用 AnnData")
    ap.add_argument("--cells", type=int, default=4000, help="標本の細胞数")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--batch-keys", default="donor_id,region,suspension_type",
                    help="batch として順に試す obs 列。カンマ区切り")
    ap.add_argument("--min-cells-per-label", type=int, default=20,
                    help="これ未満の細胞種は細胞種ごとの指標から外す")
    a = ap.parse_args(argv)

    import anndata as ad
    import numpy as np
    import scib_metrics
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    LOG(f"scib-metrics {scib_metrics.__version__ if hasattr(scib_metrics, '__version__') else '?'}")
    adata = ad.read_h5ad(a.h5ad)
    LOG(f"読み込み {adata.n_obs:,} 細胞 x {adata.n_vars:,} 遺伝子  obsm={list(adata.obsm)}")

    # value_counts は categorical の未使用カテゴリも 0 件で返す。census 由来の
    # obs はカテゴリを全体から引き継ぐので、ここを素通しすると「除外した細胞種」
    # が数百件に膨れる。実際に出現するものだけ数える。
    col = adata.obs[a.label_key]
    if hasattr(col, "cat"):
        col = col.cat.remove_unused_categories()
        adata.obs[a.label_key] = col
    vc = col.value_counts()
    vc = vc[vc > 0]
    small = sorted(vc[vc < a.min_cells_per_label].index)
    LOG(f"細胞種 {len(vc)} 種  うち {a.min_cells_per_label} 細胞未満 {len(small)} 種")
    for s in small:
        LOG(f"    除外: {s} ({int(vc[s])} 細胞)")
    keep = adata.obs[a.label_key].isin(vc[vc >= a.min_cells_per_label].index).to_numpy()
    adata = adata[keep].copy()

    rng = np.random.default_rng(0)
    if adata.n_obs > a.cells:
        idx = np.sort(rng.choice(adata.n_obs, a.cells, replace=False))
        adata = adata[idx].copy()
    LOG(f"標本 {adata.n_obs:,} 細胞  細胞種 {adata.obs[a.label_key].nunique()} 種")

    batch_key = None
    for k in a.batch_keys.split(","):
        if k in adata.obs.columns and adata.obs[k].nunique() > 1:
            batch_key = k
            break
    if batch_key is None:
        LOG("RESULT batch に使える obs 列が無い")
        return 1
    LOG(f"batch_key={batch_key}（{adata.obs[batch_key].nunique()} 水準）")

    # silhouette は使わない（塗り丸を前提とし、細胞種の分布はそうではない）
    # silhouette は使わない（2026-09-07 指示）。BioConservation の既定には
    # silhouette_label が入るので外す。BatchCorrection には 0.5.10 時点で
    # silhouette_batch が無いため、外すべきものが無い -- 名前で消しにいくと
    # TypeError になる。
    import dataclasses as _dc
    bio_fields = {f.name for f in _dc.fields(BioConservation)}
    batch_fields = {f.name for f in _dc.fields(BatchCorrection)}
    bio = BioConservation(**{"silhouette_label": False} if "silhouette_label" in bio_fields else {})
    batch = BatchCorrection(**{"silhouette_batch": False} if "silhouette_batch" in batch_fields else {})
    LOG(f"BioConservation の項目 : {sorted(bio_fields)}")
    LOG(f"BatchCorrection の項目 : {sorted(batch_fields)}")
    LOG(f"BioConservation : {bio}")
    LOG(f"BatchCorrection : {batch}")

    bm = Benchmarker(adata, batch_key=batch_key, label_key=a.label_key,
                     embedding_obsm_keys=["X_pca"],
                     bio_conservation_metrics=bio, batch_correction_metrics=batch,
                     n_jobs=-1)
    try:
        bm.benchmark()
    except Exception:
        LOG("RESULT benchmark が落ちた:")
        traceback.print_exc()
        return 1
    df = bm.get_results(min_max_scale=False)
    LOG("")
    LOG("=== 指標ごとの結果 ===")
    ok = bad = 0
    for col in df.columns:
        v = df.loc["X_pca", col] if "X_pca" in df.index else None
        try:
            fv = float(v)
            good = fv == fv  # NaN 判定
        except (TypeError, ValueError):
            good = False
            fv = v
        LOG(f"  {col:34s} {fv if good else '値にならない'}")
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
    LOG("")
    LOG(f"RESULT 値が出た指標 {ok} / 落ちた・値にならない {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
