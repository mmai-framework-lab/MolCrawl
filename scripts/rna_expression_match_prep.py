"""§1.1 の 4 項目を両側の実物で決める / §1.2 の陽性対照。

§1.1  ハッシュの前に揃えること
   a ENSG の版番号        ENSG00000141510.14 と ENSG00000141510 を同一にするか
   b カウント 0 の項目      疎行列に明示的な 0 が入っているか
   c 遺伝子の並び順        署名を並び順に依存させない
   d カウントの型          整数か浮動小数点か

§1.2  陽性対照
   コーパス側のデータセットを 1 件選び、肺アトラスと同じ読み込み経路
   （census の get_anndata）で取り直して、コーパス側のローカル h5ad から
   作った署名と一致するか。一致しなければ、この方法の検出力はまだ 0。
"""
import glob
import numpy as np
import h5py
import scipy.sparse as sp
import argparse
import cellxgene_census as cc
import hashlib

def LOG(*a): print(*a, flush=True)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--corpus-h5ad-dir", required=True)
ap.add_argument("--query-dataset", required=True)
ap.add_argument("--control-cells", type=int, default=300)
A = ap.parse_args()
CORPUS, LUNG = A.corpus_h5ad_dir, A.query_dataset

# ---------- §1.1 a,b,d : 両側の表現を見る ----------
LOG("=== §1.1 コーパス側（ローカル h5ad、census 2023-12-15 由来）===")
f = sorted(glob.glob(f"{CORPUS}/*.h5ad"))[0]
with h5py.File(f, "r") as h:
    v = h["var"]
    fid = v["feature_id"][:20]
    fid = [x.decode() if isinstance(x, bytes) else str(x) for x in fid]
    d = h["X"]["data"]
    dat = d[:200000]
    ind = h["X"]["indices"][:200000]
    LOG(f"  file: {f.split('/')[-1]}")
    LOG(f"  a 版番号つき feature_id: {sum('.' in x for x in fid)}/{len(fid)}  例 {fid[:3]}")
    LOG(f"  d dtype={d.dtype}  整数か={'はい' if np.allclose(dat, np.rint(dat)) else 'いいえ'}"
        f"  最小={dat.min()} 最大={dat.max()}")
    LOG(f"  b 明示的な 0: {int((dat == 0).sum()):,} / {len(dat):,}")
    LOG(f"  c 列番号が昇順か（CSR 内）: {bool(np.all(np.diff(ind[:5000]) != 0))}"
        f"  ※ 署名側は遺伝子キーで整列するので順序非依存")
    # 1 細胞ぶんの列が昇順に並んでいるか
    ip = h["X"]["indptr"][:3]
    seg = h["X"]["indices"][int(ip[0]):int(ip[1])]
    LOG(f"    先頭細胞の列番号は昇順: {bool(np.all(np.diff(seg) > 0))}")

LOG("\n=== §1.1 肺アトラス側（census latest 経由）===")
with cc.open_soma(census_version="latest") as c:
    ad = cc.get_anndata(c, organism="Homo sapiens",
                        obs_value_filter=f"dataset_id == '{LUNG}' and is_primary_data == True",
                        obs_column_names=["soma_joinid"],
                        var_column_names=["soma_joinid", "feature_id"])
X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
fid2 = ad.var["feature_id"].astype(str).to_numpy()
LOG(f"  a 版番号つき: {sum('.' in x for x in fid2[:20])}/20  例 {list(fid2[:3])}")
LOG(f"  d dtype={X.dtype} 整数か={'はい' if np.allclose(X.data[:200000], np.rint(X.data[:200000])) else 'いいえ'}")
LOG(f"  b 明示的な 0: {int((X.data == 0).sum()):,} / {X.nnz:,}")

# ---------- §1.2 陽性対照 ----------
LOG("\n=== §1.2 陽性対照: 同じ細胞を 2 つの経路で読んで署名が一致するか ===")
def sig(keys, counts):
    o = np.argsort(keys)
    h = hashlib.blake2b(digest_size=8)
    h.update(keys[o].astype(np.int64).tobytes())
    h.update(counts[o].astype(np.int64).tobytes())
    return int.from_bytes(h.digest(), "big")

# コーパス側から、細胞数の小さいデータセットを 1 件選ぶ
with h5py.File(f, "r") as h:
    ds = h["obs"]["dataset_id"]
    # categorical か plain か両対応
    if isinstance(ds, h5py.Group):
        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in ds["categories"][:]]
        codes = ds["codes"][:]
        dsid = np.array(cats)[codes]
    else:
        dsid = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])
    sj = h["obs"]["soma_joinid"][:]
uniq, cnt = np.unique(dsid, return_counts=True)
pick = uniq[np.argmin(cnt)]
LOG(f"  対照に使うデータセット: {pick}  （このファイル内 {cnt.min():,} 細胞）")

# 経路 A: ローカル h5ad から
with h5py.File(f, "r") as h:
    ip = h["X"]["indptr"][:]
    rows = np.where(dsid == pick)[0][:A.control_cells]
    fid_local = [x.decode() if isinstance(x, bytes) else str(x) for x in h["var"]["feature_id"][:]]
    fid_local = np.array([x.split(".")[0] for x in fid_local])
    gidx = {g: i for i, g in enumerate(np.unique(np.concatenate([fid_local, np.array([x.split('.')[0] for x in fid2])])))}
    key_local = np.array([gidx[g] for g in fid_local], dtype=np.int64)
    sigA, sj_sel = {}, []
    for r in rows:
        a, b = int(ip[r]), int(ip[r + 1])
        cols = h["X"]["indices"][a:b]
        vals = np.rint(h["X"]["data"][a:b]).astype(np.int64)
        nz = vals != 0
        sigA[int(sj[r])] = sig(key_local[cols[nz]], vals[nz])
        sj_sel.append(int(sj[r]))
LOG(f"  経路 A（ローカル h5ad）: {len(sigA):,} 細胞を署名化")

# 経路 B: census 2023-12-15 から同じ soma_joinid を取り直す
with cc.open_soma(census_version="2023-12-15") as c:
    ad2 = cc.get_anndata(c, organism="Homo sapiens",
                         obs_value_filter=f"dataset_id == '{pick}'",
                         obs_column_names=["soma_joinid"],
                         var_column_names=["soma_joinid", "feature_id"])
X2 = ad2.X.tocsr() if sp.issparse(ad2.X) else sp.csr_matrix(ad2.X)
fid_b = np.array([x.split(".")[0] for x in ad2.var["feature_id"].astype(str)])
key_b = np.array([gidx.get(g, -1) for g in fid_b], dtype=np.int64)
sj_b = ad2.obs["soma_joinid"].to_numpy()
pos = {int(s): i for i, s in enumerate(sj_b)}
hit = miss = absent = 0
for s in sj_sel:
    if s not in pos:
        absent += 1
        continue
    i = pos[s]
    a, b = X2.indptr[i], X2.indptr[i + 1]
    cols, vals = X2.indices[a:b], np.rint(X2.data[a:b]).astype(np.int64)
    nz = vals != 0
    if sig(key_b[cols[nz]], vals[nz]) == sigA[s]:
        hit += 1
    else:
        miss += 1
LOG(f"  経路 B（census 経由）: 一致 {hit:,} / 不一致 {miss:,} / 見つからず {absent:,}")
LOG("")
if hit and miss == 0 and absent == 0:
    LOG("RESULT positive_control=PASS 同じ細胞を別経路で読んでも同じ署名になる")
else:
    LOG("RESULT positive_control=FAIL 表現を揃える必要がある。上の内訳を見ること")
