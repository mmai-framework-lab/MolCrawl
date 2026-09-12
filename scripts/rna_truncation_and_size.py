import pandas as pd
import numpy as np
import h5py
import glob
import os
import argparse

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--obs-pkl", required=True, help="rna_eval_split_candidates が書いた obs")
ap.add_argument("--corpus-h5ad-dir", required=True, help="圧縮率を測るための既存 h5ad")
ap.add_argument("--lengths", default="1024,2000,2048")
ap.add_argument("--n-files", type=int, default=6)
A = ap.parse_args()
obs = pd.read_pickle(A.obs_pkl)
n = obs.nnz.to_numpy()
print(f"細胞 {len(n):,}  検出遺伝子 平均 {n.mean():.0f} 中央 {int(np.median(n)):,}", flush=True)
print("切り出し長ごとの実際の取りこぼし（1 細胞あたり平均）:", flush=True)
for k in [int(x) for x in A.lengths.split(",")]:
    lost = np.maximum(0, n - k)
    print(f"  上位 {k:>5,} 個: 影響を受ける細胞 {(n>k).mean()*100:5.1f}%  "
          f"落ちる遺伝子 平均 {lost.mean():6.0f} 個  "
          f"入力に入る割合 {np.minimum(n,k).sum()/n.sum()*100:5.1f}%", flush=True)
# h5ad の圧縮率を実測（コーパスの既存ファイルから）
fs = sorted(glob.glob(f"{A.corpus_h5ad_dir}/*.h5ad"))[:A.n_files]
tot_raw = tot_disk = 0
for f in fs:
    with h5py.File(f, "r") as h:
        nnz = h["X"]["data"].shape[0]
        ncell = h["X"]["indptr"].shape[0] - 1
    raw = nnz * 8 + ncell * 8
    disk = os.path.getsize(f)
    tot_raw += raw
    tot_disk += disk
    print(f"  {os.path.basename(f)[:34]:36s} nnz={nnz:>11,} 非圧縮 {raw/1e9:.2f}GB "
          f"実ファイル {disk/1e9:.2f}GB 比 {disk/raw:.2f}", flush=True)
r = tot_disk / tot_raw
est = (int(n.sum()) * 8 + len(n) * 8) / 1e9
print(f"\n実測比 {r:.2f} → 肺アトラス h5ad の見込み: 非圧縮 {est:.2f}GB x {r:.2f} = "
      f"{est*r:.2f} GB（.X のみ。PCA と正規化層を足すと +{len(n)*50*4/1e9:.2f}GB 程度）", flush=True)
