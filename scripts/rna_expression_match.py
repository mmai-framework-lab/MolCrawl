"""発現の中身で細胞の同一性を照合する。

================================ 前提ブロック ================================
上長 2026-09-07 判断 §1 により、名前ではなく発現で照合する方式が承認済み。

 A. 判定は「非ゼロ遺伝子の集合とカウント値の組が完全一致」
 B. 評価データは肺アトラス 093d3bfe... の primary 細胞（--query-dataset で変更可）
 C. コーパス側は学習に使った細胞そのもの（obs_id.tsv の soma_joinid で絞る）
 D. ENSG は版番号を落とす / カウント 0 は除く / 遺伝子キーで整列 / 整数に丸める
      → §1.1 として両側の実物で確認済み（2026-09-08）。4 項目とも元から揃っていた

陽性対照は省略できない。コーパス側の細胞を census 経由（= 肺アトラスと同じ
読み込み経路）で取り直し、ローカル h5ad から作った署名と一致するかを毎回見る。
一致しなければ「重なり 0」は方法の無力を意味するだけなので、その場合は
判定を出さずに終わる。

限界: 別グループが生データから再処理していればカウントが変わるため捕まえられない。
捕まえられるのは「同じ処理済み行列の再登録」まで。
=============================================================================
"""
import argparse
import glob
import hashlib
import sys

import h5py
import numpy as np
import pandas as pd


def add_root_args(ap):
    """ツリーの場所は引数で受ける。既定を置かない -- 追跡下のソースに
    このマシンのパスを書かないため（workflows の他のランチャと同じ流儀）。"""
    ap.add_argument("--corpus-h5ad-dir", required=True,
                    help="census から落とした h5ad が並ぶディレクトリ（学習コーパスの実体）")
    ap.add_argument("--obs-id-dir", required=True,
                    help="*.obs_id.tsv が並ぶディレクトリ（学習に使った soma_joinid）")


def LOG(*a):
    print(*a, flush=True)


def sig_rows(indptr, indices, data, gene_key, rows):
    """行ごとに (遺伝子キー, カウント) を 1 個の 64bit に畳む。

    遺伝子キーで整列してから畳むので、列の並び順に依存しない。
    カウントは整数に丸め、0 は除く（§1.1 b/d）。
    """
    out = np.empty(len(rows), dtype=np.uint64)
    n = 0
    for r in rows:
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b <= a:
            continue
        k = gene_key[indices[a:b]]
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


def gene_key_of(feature_ids, index):
    """ENSG から版番号を落として通し番号にする（§1.1 a）。"""
    return np.array([index.setdefault(str(f).split(".")[0], len(index))
                     for f in feature_ids], dtype=np.int64)


def read_var(path):
    with h5py.File(path, "r") as h:
        raw = h["var"]["feature_id"][:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_root_args(ap)
    ap.add_argument("--query-dataset", default="093d3bfe-6f0f-4ac0-a7a1-829f94d0a49f")
    ap.add_argument("--control-cells", type=int, default=300,
                    help="陽性対照に使う細胞数。0 にはできない")
    ap.add_argument("--corpus-scope", choices=["trained", "dataset"], default="trained")
    ap.add_argument("--corpus-file-limit", type=int, default=None, help="試走用")
    ap.add_argument("--extra-signatures", default=None,
                    help="ローカル h5ad に無い細胞の署名を書き出した .npy。"
                         "download_dir は学習した 28,393,154 のうち 22,475,847 しか"
                         "持っておらず、これを足さないと 79.16%% しか照合していない")
    a = ap.parse_args(argv)
    if a.control_cells < 1:
        raise SystemExit("陽性対照は省略できない（§1.2）")
    LOG(f"ARGS query={a.query_dataset} scope={a.corpus_scope} control={a.control_cells}")

    files = sorted(glob.glob(f"{a.corpus_h5ad_dir}/*.h5ad"))
    if a.corpus_file_limit:
        files = files[:a.corpus_file_limit]
    LOG(f"CORPUS files={len(files)}")

    keep = None
    if a.corpus_scope == "trained":
        keep = set()
        for t in glob.glob(f"{a.obs_id_dir}/*.obs_id.tsv"):
            keep.update(pd.read_csv(t, header=None)[0].to_numpy().tolist())
        LOG(f"CORPUS trained_cells={len(keep):,}")

    # var は census の遺伝子軸なので全ファイル共通のはず。1 度作って使い回し、
    # 違うファイルが来たら作り直す（黙って別の軸で畳まないため）。
    gindex = {}
    var0 = read_var(files[0])
    key_cache = {hash(var0.tobytes()): gene_key_of(var0, gindex)}
    LOG(f"CORPUS genes={len(var0):,}")

    parts, total, control_src = [], 0, []
    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as h:
            var = h["var"]["feature_id"][:]
            var = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in var])
            hk = hash(var.tobytes())
            if hk not in key_cache:
                key_cache[hk] = gene_key_of(var, gindex)
                LOG(f"CORPUS [{i}] 別の遺伝子軸を検出。マッピングを追加")
            gk = key_cache[hk]
            indptr = h["X"]["indptr"][:]
            data = h["X"]["data"][:]
            indices = h["X"]["indices"][:]
            sj = h["obs"]["soma_joinid"][:]
            rows = np.arange(len(indptr) - 1) if keep is None \
                else np.where(np.isin(sj, list(keep)))[0]
            s = sig_rows(indptr, indices, data, gk, rows)
        parts.append(s)
        total += len(s)
        if len(control_src) < a.control_cells and len(rows):
            take = rows[:min(a.control_cells - len(control_src), len(rows))]
            control_src.extend(int(sj[r]) for r in take)
        if i % 400 == 0:
            LOG(f"CORPUS [{i}/{len(files)}] cells={total:,}")

    if a.extra_signatures:
        extra = np.load(a.extra_signatures)
        LOG(f"CORPUS extra_signatures={len(extra):,} from {a.extra_signatures}")
        parts.append(extra.astype(np.uint64))
        total += len(extra)
    corpus = np.unique(np.concatenate(parts))
    del parts
    LOG(f"CORPUS cells={total:,} unique_signatures={len(corpus):,}")
    if a.corpus_scope == "trained" and not a.corpus_file_limit:
        trained = len(keep)
        LOG(f"CORPUS coverage={total/trained*100:.2f}% ({total:,} / {trained:,})")
        if total < trained:
            LOG(f"CORPUS WARNING 学習した細胞のうち {trained-total:,} 件を照合していない。"
                "「重なりなし」はこの範囲の話であることを記録に書くこと")

    import cellxgene_census as cc
    import scipy.sparse as sp

    def signed_from_census(version, value_filter, label):
        with cc.open_soma(census_version=version) as c:
            ad = cc.get_anndata(c, organism="Homo sapiens",
                                obs_value_filter=value_filter,
                                obs_column_names=["soma_joinid"],
                                var_column_names=["soma_joinid", "feature_id"])
        X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
        gk = gene_key_of(ad.var["feature_id"].astype(str).to_numpy(), gindex)
        s = sig_rows(X.indptr, X.indices, X.data, gk, np.arange(X.shape[0]))
        LOG(f"{label} cells={X.shape[0]:,} signed={len(s):,}")
        return s, ad.obs["soma_joinid"].to_numpy()

    # --- 陽性対照: コーパスの細胞を「肺アトラスと同じ読み込み経路」で取り直す ---
    LOG("--- 陽性対照 ---")
    ids = ",".join(str(x) for x in control_src[:a.control_cells])
    ctl, _ = signed_from_census("2023-12-15", f"soma_joinid in [{ids}]", "CONTROL")
    ctl_hit = int(np.isin(ctl, corpus).sum())
    LOG(f"CONTROL matched={ctl_hit:,} / {len(ctl):,} "
        f"({ctl_hit/max(len(ctl),1)*100:.2f}%)")
    if len(ctl) == 0 or ctl_hit < len(ctl):
        LOG("RESULT positive_control=FAIL")
        LOG("RESULT verdict=判定を出さない。コーパスにある細胞すら見つけられていない")
        return 1
    LOG("RESULT positive_control=PASS")

    # --- 本番 ---
    LOG("--- 肺アトラス ---")
    q, _ = signed_from_census(
        "latest", f"dataset_id == '{a.query_dataset}' and is_primary_data == True", "QUERY")
    hit = int(np.isin(q, corpus).sum())
    LOG("=" * 60)
    LOG(f"RESULT corpus_cells={total:,} unique={len(corpus):,}")
    LOG(f"RESULT query_cells={len(q):,}")
    LOG(f"RESULT matched={hit:,} ({hit/max(len(q),1)*100:.4f}%)")
    LOG("RESULT verdict=" + ("完全一致による重なりなし（近い重複は §1.3 で別途）"
                             if hit == 0 else "一致あり — 重複の疑い"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
