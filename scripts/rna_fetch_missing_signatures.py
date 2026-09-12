"""照合できていない 5,917,307 細胞を census 2023-12-15 から取り、署名にする。

ローカルの download_dir は学習に使った 28,393,154 細胞のうち 22,475,847
(79.16%) しか持っていない。残りを照合しないまま「重なりなし」とは言えない。
"""
import argparse
import hashlib
import glob
import numpy as np
import pandas as pd
import h5py
import cellxgene_census as cc
import scipy.sparse as sp
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
ap.add_argument("--out-signatures", required=True, help="署名の出力 (.npy)")
ap.add_argument("--out-ids", required=True, help="取りに行った soma_joinid の出力 (.npy)")
ap.add_argument("--chunk", type=int, default=200_000)
A = ap.parse_args()
CORPUS, PREP, OUT, CHUNK = A.corpus_h5ad_dir, A.obs_id_dir, A.out_signatures, A.chunk

keep = set()
for t in glob.glob(f"{PREP}/*.obs_id.tsv"):
    keep.update(pd.read_csv(t, header=None)[0].to_numpy().tolist())
have = set()
for p in sorted(glob.glob(f"{CORPUS}/*.h5ad")):
    with h5py.File(p, "r") as h:
        have.update(h["obs"]["soma_joinid"][:].tolist())
missing = sorted(keep - have)
LOG(f"学習した {len(keep):,} / ローカル {len(have):,} / 取りに行く {len(missing):,}")
np.save(A.out_ids, np.array(missing, dtype=np.int64))

gindex = {}
def gene_key(fids):
    return np.array([gindex.setdefault(str(f).split(".")[0], len(gindex)) for f in fids],
                    dtype=np.int64)
def sig_rows(indptr, indices, data, gk, rows):
    out = np.empty(len(rows), dtype=np.uint64)
    n = 0
    for r in rows:
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b <= a:
            continue
        k = gk[indices[a:b]]
        v = np.rint(data[a:b]).astype(np.int64)
        nz = v != 0
        k, v = k[nz], v[nz]
        if k.size == 0:
            continue
        o = np.argsort(k)
        h = hashlib.blake2b(digest_size=8)
        h.update(k[o].astype(np.int64).tobytes())
        h.update(v[o].tobytes())
        out[n] = int.from_bytes(h.digest(), "big")
        n += 1
    return out[:n]

sigs = []
with cc.open_soma(census_version="2023-12-15") as c:
    for i in range(0, len(missing), CHUNK):
        ids = missing[i:i + CHUNK]
        ad = cc.get_anndata(c, organism="Homo sapiens",
                            obs_value_filter=f"soma_joinid in {ids}",
                            obs_column_names=["soma_joinid"],
                            var_column_names=["soma_joinid", "feature_id"])
        X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
        gk = gene_key(ad.var["feature_id"].astype(str).to_numpy())
        s = sig_rows(X.indptr, X.indices, X.data, gk, np.arange(X.shape[0]))
        sigs.append(s)
        LOG(f"  [{min(i+CHUNK, len(missing)):,}/{len(missing):,}] 署名 {sum(len(x) for x in sigs):,}")
allsig = np.concatenate(sigs) if sigs else np.array([], dtype=np.uint64)
np.save(OUT, allsig)
LOG(f"RESULT missing_signed={len(allsig):,} unique={len(np.unique(allsig)):,}")
LOG(f"RESULT gene_index_size={len(gindex):,}")
LOG(f"WROTE {OUT}")
