"""§1.3 近い重複の標本確認。完全一致では外れる形を、緩い鍵で 1 回だけ見る。

完全一致は片側で低発現の遺伝子が 1 つ落ちるだけで外れる。そこで
「検出された遺伝子の集合」の重なり（Jaccard）で見る。

28,393,154 x 193,108 の総当りは不可能なので MinHash と帯分割で候補を絞る。
k=64 / r=4 / b=16 なら、重なり 0.8 の対を取り逃す確率は 0.04% 程度。

読み方に注意が要る。似た細胞どうしは、重複でなくても遺伝子集合がよく重なる。
したがって「重なりが高い対がある」こと自体は重複の証拠にならない。
基準として次を並べる。

  陽性対照   コーパスの細胞そのもの        → 最大 Jaccard は 1.0 になるはず
  陰性の基準 別の後発データセット（胚）    → 重複が無い側の分布
  本題       肺アトラス                   → 基準と変わらなければ近い重複なし
"""
import argparse
import glob
import sys
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import cellxgene_census as cc

def LOG(*a): print(*a, flush=True)


def add_root_args(ap):
    """ツリーの場所は引数で受ける。既定を置かない -- 追跡下のソースに
    このマシンのパスを書かないため（workflows の他のランチャと同じ流儀）。"""
    ap.add_argument("--corpus-h5ad-dir", required=True,
                    help="census から落とした h5ad が並ぶディレクトリ（学習コーパスの実体）")
    ap.add_argument("--obs-id-dir", required=True,
                    help="*.obs_id.tsv が並ぶディレクトリ（学習に使った soma_joinid）")
K, R = 64, 4
B = K // R
LUNG = "093d3bfe-6f0f-4ac0-a7a1-829f94d0a49f"
EMBRYO = "58f43044-cd3c-42b8-bc0b-3498eb236359"


def make_perms(n_genes, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, size=K, dtype=np.int64)
    b = rng.integers(0, 2**31 - 1, size=K, dtype=np.int64)
    return a, b


def sketch_rows(indptr, indices, gene_key, rows, a, b, P=(1 << 31) - 1):
    """行ごとに MinHash 署名（k 個の最小値）。カウントは使わない — 集合だけ見る。"""
    out = np.full((len(rows), K), np.iinfo(np.uint32).max, dtype=np.uint32)
    n = 0
    for r in rows:
        s, e = int(indptr[r]), int(indptr[r + 1])
        if e <= s:
            continue
        g = gene_key[indices[s:e]].astype(np.int64)
        h = ((a[None, :] * g[:, None] + b[None, :]) % P)
        out[n] = h.min(axis=0).astype(np.uint32)
        n += 1
    return out[:n]


def band_hash(sk):
    """帯ごとに r 行を 1 つの uint64 に畳む。"""
    n = sk.shape[0]
    out = np.empty((n, B), dtype=np.uint64)
    for j in range(B):
        blk = sk[:, j * R:(j + 1) * R].astype(np.uint64)
        h = np.zeros(n, dtype=np.uint64)
        for c in range(R):
            h = (h * np.uint64(1000003)) ^ blk[:, c]
        out[:, j] = h
    return out


def gene_key_factory():
    idx = {}
    def f(fids):
        return np.array([idx.setdefault(str(x).split(".")[0], len(idx)) for x in fids],
                        dtype=np.int64)
    return f, idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_root_args(ap)
    ap.add_argument("--query-sample", type=int, default=2000)
    ap.add_argument("--corpus-file-limit", type=int, default=None)
    ap.add_argument("--missing-ids", required=True,
                    help="ローカルの h5ad に無い細胞の soma_joinid（.npy）。\ndownload_dir は学習した細胞の一部しか持たないことがある")
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args(argv)
    a, b = make_perms(0)
    gkey, gidx = gene_key_factory()

    keep = set()
    for t in glob.glob(f"{a_.obs_id_dir}/*.obs_id.tsv"):
        keep.update(pd.read_csv(t, header=None)[0].to_numpy().tolist())
    LOG(f"学習した細胞 {len(keep):,}")

    files = sorted(glob.glob(f"{a_.corpus_h5ad_dir}/*.h5ad"))
    if a_.corpus_file_limit:
        files = files[:a_.corpus_file_limit]
    parts = []
    # 陽性対照に使う soma_joinid。署名化した細胞そのものから取る。missing_ids から
    # 取ると、census 取得を飛ばす試走では対照がコーパス側の集合に入っておらず、
    # 方法ではなく試走の組み方のせいで対照が落ちる（2026-09-09 にそうなった）。
    control_ids: list = []
    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as h:
            var = h["var"]["feature_id"][:]
            gk = gkey([x.decode() if isinstance(x, bytes) else str(x) for x in var])
            ip = h["X"]["indptr"][:]
            ind = h["X"]["indices"][:]
            sj = h["obs"]["soma_joinid"][:]
            rows = np.where(np.isin(sj, list(keep)))[0]
            # 非ゼロが 0 の行は署名化されないので、対照には取らない
            nz_rows = [int(r) for r in rows if int(ip[r + 1]) > int(ip[r])]
            parts.append(sketch_rows(ip, ind, gk, rows, a, b))
            if len(control_ids) < 300 and nz_rows:
                control_ids.extend(int(sj[r]) for r in nz_rows[:300 - len(control_ids)])
        if i % 500 == 0:
            LOG(f"CORPUS [{i}/{len(files)}] {sum(len(x) for x in parts):,} 細胞")
    LOG(f"CONTROL ids={len(control_ids)}（署名化した細胞から取得）")
    corpus_sk = np.concatenate(parts)
    del parts
    LOG(f"CORPUS ローカル分 {len(corpus_sk):,}")

    # ローカルに無い分を census から
    if not a_.corpus_file_limit:
        missing = np.load(a_.missing_ids)
        LOG(f"CORPUS census から取る {len(missing):,}")
        more = []
        with cc.open_soma(census_version="2023-12-15") as c:
            for i in range(0, len(missing), 200_000):
                ids = missing[i:i + 200_000].tolist()
                ad = cc.get_anndata(c, organism="Homo sapiens",
                                    obs_value_filter=f"soma_joinid in {ids}",
                                    obs_column_names=["soma_joinid"],
                                    var_column_names=["soma_joinid", "feature_id"])
                X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
                gk = gkey(ad.var["feature_id"].astype(str).to_numpy())
                more.append(sketch_rows(X.indptr, X.indices, gk, np.arange(X.shape[0]), a, b))
                LOG(f"  [{min(i+200_000, len(missing)):,}/{len(missing):,}]")
        corpus_sk = np.concatenate([corpus_sk] + more)
        del more
    LOG(f"CORPUS 合計 {len(corpus_sk):,} 細胞  被覆 "
        f"{len(corpus_sk)/len(keep)*100:.2f}%")

    corpus_bands = band_hash(corpus_sk)
    LOG("CORPUS 帯を作成")
    sorted_bands = [np.sort(corpus_bands[:, j]) for j in range(B)]
    order_bands = [np.argsort(corpus_bands[:, j], kind="stable") for j in range(B)]
    del corpus_bands

    def probe(sk, label):
        """帯で候補を引き、MinHash から Jaccard を見積もって最大を取る。"""
        bands = band_hash(sk)
        best = np.zeros(len(sk))
        ncand = np.zeros(len(sk), dtype=np.int64)
        for j in range(B):
            sb, ob = sorted_bands[j], order_bands[j]
            lo = np.searchsorted(sb, bands[:, j], "left")
            hi = np.searchsorted(sb, bands[:, j], "right")
            for q in np.where(hi > lo)[0]:
                cand = ob[lo[q]:hi[q]]
                ncand[q] += len(cand)
                if len(cand) > 2000:
                    cand = cand[:2000]
                jac = (corpus_sk[cand] == sk[q][None, :]).mean(axis=1)
                m = jac.max()
                if m > best[q]:
                    best[q] = m
        LOG(f"{label} n={len(sk):,} 候補が出た細胞 {int((ncand>0).sum()):,} "
            f"最大 Jaccard: 中央 {np.median(best):.3f} "
            f"95% {np.quantile(best,0.95):.3f} 最大 {best.max():.3f} "
            f"／ 0.8 以上 {int((best>=0.8).sum()):,} 0.95 以上 {int((best>=0.95).sum()):,}")
        return best

    def sketch_census(version, vf, n_limit, label):
        with cc.open_soma(census_version=version) as c:
            ad = cc.get_anndata(c, organism="Homo sapiens", obs_value_filter=vf,
                                obs_column_names=["soma_joinid"],
                                var_column_names=["soma_joinid", "feature_id"])
        X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
        gk = gkey(ad.var["feature_id"].astype(str).to_numpy())
        rng = np.random.default_rng(1)
        rows = np.arange(X.shape[0])
        if n_limit and X.shape[0] > n_limit:
            rows = np.sort(rng.choice(X.shape[0], n_limit, replace=False))
        LOG(f"{label} 取得 {X.shape[0]:,} → 標本 {len(rows):,}")
        return sketch_rows(X.indptr, X.indices, gk, rows, a, b)

    LOG("--- 陽性対照: コーパスの細胞そのもの（署名化した中から）---")
    if not control_ids:
        raise SystemExit("陽性対照の細胞が取れていない。コーパス側が空")
    pos = probe(sketch_census("2023-12-15", f"soma_joinid in {control_ids}", None, "POS"), "POS")
    LOG("--- 基準: 別の後発データセット（胚）---")
    ref = probe(sketch_census("latest",
        f"dataset_id == '{EMBRYO}' and is_primary_data == True", a_.query_sample, "REF"), "REF")
    LOG("--- 本題: 肺アトラス ---")
    q = probe(sketch_census("latest",
        f"dataset_id == '{LUNG}' and is_primary_data == True", a_.query_sample, "QUERY"), "QUERY")

    np.savez(a_.out + ".npz", pos=pos, ref=ref, query=q)
    LOG("=" * 60)
    LOG(f"RESULT 陽性対照 中央 {np.median(pos):.3f}（1.0 に近くなければ方法が効いていない）")
    LOG(f"RESULT 基準(胚)  中央 {np.median(ref):.3f} 95% {np.quantile(ref,0.95):.3f} 最大 {ref.max():.3f}")
    LOG(f"RESULT 肺        中央 {np.median(q):.3f} 95% {np.quantile(q,0.95):.3f} 最大 {q.max():.3f}")
    if np.median(pos) < 0.95:
        LOG("RESULT verdict=判定不能。陽性対照が 1.0 に届かず、緩い鍵が効いていない")
    elif q.max() <= max(ref.max(), 0.0) + 0.02:
        LOG("RESULT verdict=近い重複は見当たらない（基準と同水準）")
    else:
        LOG("RESULT verdict=基準より高い対がある。内訳を見ること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
