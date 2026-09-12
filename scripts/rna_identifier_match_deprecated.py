"""名前（細胞の識別子）による照合。**この方法は使わないこと。**

検出力がない。細胞の識別子は研究ごとに命名の流儀が違う
（`WTDAtest7887999-AAACC...` / `10X389_2:AATCG...` /
`CAAGA..._Donor1_lung_rest` / `AAACC...-1-0`）。同じ細胞を別のグループが再登録して
いても文字列は一致しないので、**汚染の有無によらず「一致 0」を返す。**

2026-09-08 に肺アトラス 193,108 細胞 対 コーパス 286 データセットで 0% が出たが、
これは重なりが無いことを意味しない。陽性対照が通ったのは同じ命名の相手だったためで、
別命名の相手を検出できる証明にはなっていなかった。

**代わりに scripts/rna_expression_match.py を使う。**カウント行列の中身で照合するので
命名に依存しない。

この版を残しているのは、同じ取り違えが 3 度目にならないようにするため。「登録の単位で
照合しても細胞の重なりは分からない」という誤りは、`dataset_id` での照合と合わせて
既に 2 度起きている。
"""

import argparse
import glob
import hashlib
import re

import cellxgene_census as cc
import fsspec
import h5py
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="名前による照合（非推奨）")
ap.add_argument("--query-dataset", required=True)
ap.add_argument("--obs-id-dir", required=True, help="*.obs_id.tsv が並ぶディレクトリ")
A = ap.parse_args()
LUNG = A.query_dataset
BARE10X = re.compile(r"^[ACGT]{14,18}(-\d+)?$")
PREP = A.obs_id_dir


def log(*a):
    print(*a, flush=True)


def obs_index(url):
    """HDF5 の部分読み。ファイル全体は落とさず obs の索引だけ取る。"""
    with fsspec.open(url, "rb") as fh:
        with h5py.File(fh, "r") as f:
            g = f["obs"]
            k = g.attrs.get("_index", "_index")
            k = k.decode() if isinstance(k, bytes) else k
            a = g[k][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def h64(arr):
    """64 bit ハッシュ。4,000 万件でも 320MB に収まり、衝突の期待値は 4e-5 件。"""
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in arr),
        dtype=np.uint64, count=len(arr))


def shape_of(arr, n=4000):
    s = arr[:n]
    bare = sum(1 for x in s if BARE10X.match(x))
    return bare / max(len(s), 1)


# ---------- 学習コーパスの実体（細胞単位）----------
corpus_ids = set()
for f in glob.glob(f"{PREP}/*.obs_id.tsv"):
    corpus_ids.update(pd.read_csv(f, header=None)[0].to_numpy().tolist())
log(f"STAGE0 corpus_cells={len(corpus_ids):,}")

with cc.open_soma(census_version="2023-12-15") as c:
    old = c["census_data"]["homo_sapiens"].obs.read(
        column_names=["soma_joinid", "dataset_id"]).concat().to_pandas()
    ds_old = c["census_info"]["datasets"].read().concat().to_pandas()
old_in = old[old["soma_joinid"].isin(corpus_ids)]
corpus_ds = sorted(set(old_in["dataset_id"]))
per_ds = old_in["dataset_id"].value_counts()
del old, old_in
log(f"STAGE0 corpus_datasets={len(corpus_ds)}")
log(f"STAGE0 datasets_table_columns={list(ds_old.columns)}")

url_of = {}
for _, r in ds_old.iterrows():
    if r["dataset_id"] in set(corpus_ds):
        vid = r.get("dataset_version_id") or r["dataset_id"]
        url_of[r["dataset_id"]] = f"https://datasets.cellxgene.cziscience.com/{vid}.h5ad"
missing = [d for d in corpus_ds if d not in url_of]
log(f"STAGE0 urls_resolved={len(url_of)} unresolved={len(missing)}")
if missing:
    log(f"STAGE0 unresolved_ids={missing[:10]}")

# ---------- 段階 1: 識別子の形式 ----------
with cc.open_soma(census_version="latest") as c:
    ds_new = c["census_info"]["datasets"].read().concat().to_pandas()
row = ds_new[ds_new["dataset_id"] == LUNG].iloc[0]
lung_url = f"https://datasets.cellxgene.cziscience.com/{row.get('dataset_version_id', LUNG)}.h5ad"
log(f"STAGE1 lung_url={lung_url}")
lung = obs_index(lung_url)
log(f"STAGE1 lung_cells={len(lung):,}")
log(f"STAGE1 lung_samples={list(lung[:3])}")
lung_bare = shape_of(lung)
log(f"STAGE1 lung_bare10x_frac={lung_bare:.3f}")

probe = corpus_ds[:5]
for d in probe:
    try:
        a = obs_index(url_of[d])
        log(f"STAGE1 corpus_probe {d} n={len(a):,} bare10x={shape_of(a):.3f} sample={a[0]}")
    except Exception as e:
        log(f"STAGE1 corpus_probe {d} FAILED {type(e).__name__}: {e}")

if lung_bare > 0.5:
    log("STAGE1 WARNING 肺側の識別子は 10x の素のバーコードです。"
        "研究をまたぐと必ず衝突するため、一致は重複の証拠になりません。"
        "以降の数値は上限（偽陽性込み）として読むこと。")

# ---------- 段階 2: 全 286 データセットとの突合 ----------
lung_h = h64(lung)
lung_bare_mask = np.array([bool(BARE10X.match(x)) for x in lung])
log(f"STAGE2 lung_hashed={len(lung_h):,} bare={int(lung_bare_mask.sum()):,} "
    f"named={int((~lung_bare_mask).sum()):,}")

hit_any = np.zeros(len(lung), dtype=bool)
failed, done = [], 0
for i, d in enumerate(corpus_ds, 1):
    try:
        a = obs_index(url_of[d])
    except Exception as e:
        failed.append((d, f"{type(e).__name__}: {e}"))
        log(f"STAGE2 [{i}/{len(corpus_ds)}] {d} FAILED {type(e).__name__}")
        continue
    hs = np.unique(h64(a))
    m = np.isin(lung_h, hs)
    done += 1
    if m.any():
        nb = int((m & ~lung_bare_mask).sum())
        log(f"STAGE2 [{i}/{len(corpus_ds)}] {d} corpus_cells={int(per_ds.get(d,0)):,} "
            f"file_cells={len(a):,} HIT total={int(m.sum()):,} named={nb:,} "
            f"example={lung[np.where(m)[0][0]]}")
        hit_any |= m
    elif i % 25 == 0:
        log(f"STAGE2 [{i}/{len(corpus_ds)}] progress: no hits so far, "
            f"cumulative_hits={int(hit_any.sum()):,}")

log("=" * 60)
log(f"RESULT datasets_checked={done} / {len(corpus_ds)}  failed={len(failed)}")
for d, e in failed:
    log(f"RESULT failed_dataset {d} {e}")
log(f"RESULT lung_cells={len(lung):,}")
log(f"RESULT lung_cells_matching_corpus={int(hit_any.sum()):,} "
    f"({hit_any.sum()/len(lung)*100:.4f}%)")
log(f"RESULT   of which study_scoped_ids={int((hit_any & ~lung_bare_mask).sum()):,}  "
    f"bare_10x_ids={int((hit_any & lung_bare_mask).sum()):,}")
if failed:
    log("RESULT verdict=不完全（取得に失敗したデータセットがあるため、"
        "重なりなしとは言い切れない）")
elif hit_any.sum() == 0:
    log("RESULT verdict=重なりなし（286 データセット全件で一致 0）")
else:
    log("RESULT verdict=一致あり — 上の内訳を読むこと。"
        "bare_10x_ids は偽陽性の可能性が高く、study_scoped_ids が実質の重なり")

