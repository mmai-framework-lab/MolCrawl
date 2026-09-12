"""一致 0 という結果の裏取り。

0 件は「重なりがない」でも「読めていなかった」でも同じ見え方になる。本番ログは
一致が出た時しか件数を出さない作りだったので、そこを埋める。

  (1) 286 件それぞれの読み取り件数を census の公称件数と突き合わせる
  (2) 陽性対照: コーパス側のデータセットを問い合わせ側に置き、
      同じ経路で 100% 一致すること（= 一致を検出できること）を確かめる
"""
import hashlib
import glob
import numpy as np
import pandas as pd
import h5py
import fsspec
import cellxgene_census as cc
import argparse


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--obs-id-dir", required=True, help="*.obs_id.tsv が並ぶディレクトリ")
ap.add_argument("--control-datasets", type=int, default=3)
A = ap.parse_args()
PREP = A.obs_id_dir
def log(*a):
    print(*a, flush=True)


def obs_index(url):
    with fsspec.open(url, "rb") as fh:
        with h5py.File(fh, "r") as f:
            g = f["obs"]
            k = g.attrs.get("_index", "_index")
            k = k.decode() if isinstance(k, bytes) else k
            a = g[k][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def h64(arr):
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in arr),
        dtype=np.uint64, count=len(arr))


corpus_ids = set()
for f in glob.glob(f"{PREP}/*.obs_id.tsv"):
    corpus_ids.update(pd.read_csv(f, header=None)[0].to_numpy().tolist())
with cc.open_soma(census_version="2023-12-15") as c:
    old = c["census_data"]["homo_sapiens"].obs.read(
        column_names=["soma_joinid", "dataset_id"]).concat().to_pandas()
    ds = c["census_info"]["datasets"].read().concat().to_pandas()
corpus_ds = sorted(set(old[old["soma_joinid"].isin(corpus_ids)]["dataset_id"]))
del old
declared = ds.set_index("dataset_id")["dataset_total_cell_count"].to_dict()
url_of = {r["dataset_id"]: f"https://datasets.cellxgene.cziscience.com/{r['dataset_version_id']}.h5ad"
          for _, r in ds.iterrows() if r["dataset_id"] in set(corpus_ds)}

log(f"CHECK corpus_datasets={len(corpus_ds)}")
parts, bad, total = [], [], 0
CONTROL = corpus_ds[:A.control_datasets]
control_bc = {}
for i, d in enumerate(corpus_ds, 1):
    try:
        a = obs_index(url_of[d])
    except Exception as e:
        bad.append((d, "read_failed", f"{type(e).__name__}: {e}"))
        log(f"CHECK [{i}/{len(corpus_ds)}] {d} READ FAILED")
        continue
    exp = int(declared.get(d, -1))
    if len(a) == 0:
        bad.append((d, "empty", "0 rows"))
    elif exp > 0 and len(a) != exp:
        bad.append((d, "count_mismatch", f"file={len(a)} census={exp}"))
    total += len(a)
    if d in CONTROL:
        control_bc[d] = a[:5000]
    parts.append(np.unique(h64(a)))
    if i % 50 == 0:
        log(f"CHECK [{i}/{len(corpus_ds)}] read_ok cells_so_far={total:,} anomalies={len(bad)}")

allh = np.unique(np.concatenate(parts))
log("=" * 60)
log(f"RESULT datasets_read={len(parts)} / {len(corpus_ds)}")
log(f"RESULT total_cells_read={total:,}  unique_hashes={len(allh):,}")
log(f"RESULT anomalies={len(bad)}")
for d, kind, detail in bad[:20]:
    log(f"RESULT anomaly {kind} {d} {detail}")

log("--- 陽性対照: コーパス側を問い合わせ側に置く ---")
ok = True
for d, bc in control_bc.items():
    m = np.isin(h64(bc), allh)
    log(f"CONTROL {d} queried={len(bc):,} matched={int(m.sum()):,} ({m.mean()*100:.2f}%)")
    if m.mean() < 0.999:
        ok = False
log(f"RESULT positive_control={'PASS（一致を検出できる）' if ok else 'FAIL（経路が一致を拾えていない）'}")
