"""署名化できた 22,475,847 が、学習した 28,393,154 に足りない理由を特定する。"""
import argparse
import glob
import numpy as np
import pandas as pd
import h5py
def LOG(*a): print(*a, flush=True)

def add_root_args(ap):
    """ツリーの場所は引数で受ける。既定を置かない -- 追跡下のソースに
    このマシンのパスを書かないため（workflows の他のランチャと同じ流儀）。"""
    ap.add_argument("--corpus-h5ad-dir", required=True,
                    help="census から落とした h5ad が並ぶディレクトリ（学習コーパスの実体）")
    ap.add_argument("--obs-id-dir", required=True,
                    help="*.obs_id.tsv が並ぶディレクトリ（学習に使った soma_joinid）")

ap = argparse.ArgumentParser(description=__doc__)
add_root_args(ap)
A = ap.parse_args()
CORPUS, PREP = A.corpus_h5ad_dir, A.obs_id_dir

keep = set()
for t in glob.glob(f"{PREP}/*.obs_id.tsv"):
    keep.update(pd.read_csv(t, header=None)[0].to_numpy().tolist())
LOG(f"学習した細胞（obs_id.tsv の合計）: {len(keep):,}")

files = sorted(glob.glob(f"{CORPUS}/*.h5ad"))
LOG(f"ローカル h5ad: {len(files):,} ファイル")
tot_rows = 0
present = set()
empty_rows = 0
for i, p in enumerate(files, 1):
    with h5py.File(p, "r") as h:
        sj = h["obs"]["soma_joinid"][:]
        ip = h["X"]["indptr"][:]
    tot_rows += len(sj)
    present.update(sj.tolist())
    empty_rows += int((np.diff(ip) == 0).sum())
    if i % 1000 == 0:
        LOG(f"  [{i}/{len(files)}] 行 {tot_rows:,}")
LOG(f"ローカルの総行数        : {tot_rows:,}")
LOG(f"ローカルの一意 soma_joinid: {len(present):,}")
LOG(f"空行（非ゼロ 0 個）      : {empty_rows:,}")
LOG("")
LOG(f"学習した細胞のうちローカルにある: {len(keep & present):,} "
    f"({len(keep & present)/len(keep)*100:.2f}%)")
LOG(f"学習した細胞のうちローカルに無い: {len(keep - present):,}")
LOG(f"ローカルにあるが学習に使っていない: {len(present - keep):,}")
