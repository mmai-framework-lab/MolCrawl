import cellxgene_census as cc
import pandas as pd
import numpy as np
import itertools
import pickle
import argparse
"""§6.1 区分比を含む分割の検討 / §4(2)(3) 系列長と容量の見込み。"""

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dataset", required=True)
ap.add_argument("--vocab", required=True, help="語彙の pickle（token_dictionary）")
ap.add_argument("--airway-tissues", default="bronchus,trachea",
                help="気道として扱う tissue。カンマ区切り")
ap.add_argument("--out-obs", required=True, help="obs の保存先 (.pkl)")
ap.add_argument("--test-donors", type=int, default=3, help="test に回すドナー数")
A = ap.parse_args()
LUNG = A.dataset
def log(*a):
    print(*a, flush=True)
COLS = ["tissue","tissue_general","cell_type","donor_id","assay","suspension_type","nnz","n_measured_vars"]
with cc.open_soma(census_version="latest") as c:
    obs = c["census_data"]["homo_sapiens"].obs.read(
        value_filter=f"dataset_id == '{LUNG}' and is_primary_data == True",
        column_names=COLS).concat().to_pandas()
    var = c["census_data"]["homo_sapiens"].ms["RNA"].var.read(
        column_names=["soma_joinid","feature_id"]).concat().to_pandas()
for c_ in ["tissue","tissue_general","cell_type","donor_id","assay","suspension_type"]:
    obs[c_] = obs[c_].astype(str)
obs.to_pickle(A.out_obs)
log(f"cells={len(obs):,}")

AIR = set(A.airway_tissues.split(","))
obs["区分"] = np.where(obs.tissue.isin(AIR), "気道", "実質")
log("\n=== §6.1 ドナー x 区分 ===")
log(pd.crosstab(obs.donor_id, obs["区分"], margins=True).to_string())

don = sorted(obs.donor_id.unique())

ct = pd.crosstab(obs.cell_type, obs.donor_id)
ge100 = ct.sum(1) >= 100
rows = []
for combo in itertools.combinations(don, A.test_donors):
    m = obs.donor_id.isin(combo)
    te, tr = obs[m], obs[~m]
    rest = [d for d in don if d not in combo]
    if ((ct[list(combo)].sum(1) == 0) & ge100).sum() or ((ct[rest].sum(1) == 0) & ge100).sum():
        continue
    rows.append(dict(test=",".join(combo), frac=len(te)/len(obs),
        te_nuc=(te.suspension_type == "nucleus").mean(), tr_nuc=(tr.suspension_type == "nucleus").mean(),
        te_air=(te["区分"] == "気道").mean(), tr_air=(tr["区分"] == "気道").mean(),
        ge20=int((ct[list(combo)].sum(1) >= 20).sum())))
df = pd.DataFrame(rows)
df["d_frac"] = (df.frac - .3).abs()
df["d_nuc"] = (df.te_nuc - df.tr_nuc).abs()
df["d_air"] = (df.te_air - df.tr_air).abs()
df["score"] = df.d_frac + df.d_nuc + df.d_air
fm = {k: "{:.1%}".format for k in ["frac","te_nuc","tr_nuc","te_air","tr_air"]}
fm.update({k: "{:.3f}".format for k in ["d_frac","d_nuc","d_air","score"]})
log(f"\n=== 3 つを揃えた順（成立する分割 {len(df)} 通り）===")
log(df.sort_values("score").head(8).to_string(index=False, formatters=fm))
log("\n=== 提案していた A26,A37,A42 ===")
log(df[df.test == "A26,A37,A42"].to_string(index=False, formatters=fm))

log("\n=== §4(2) 1 細胞あたりの検出遺伝子数（nnz）===")
q = obs.nnz.quantile([.05,.25,.5,.75,.95])
log(f"  中央 {int(obs.nnz.median()):,}  四分位 {int(q[.25]):,}/{int(q[.75]):,}  5-95% {int(q[.05]):,}〜{int(q[.95]):,}")
for k in [1024, 2000, 2048]:
    log(f"  {k:,} 個を超える細胞: {(obs.nnz > k).mean()*100:5.1f}%  "
        f"（{k} で切ると平均 {max(0, obs.nnz.mean()-k):.0f} 個が落ちる）")
vocab = set(pickle.load(open(A.vocab, "rb")))
log(f"  語彙 25,426 に載る census 遺伝子: {var.feature_id.isin(vocab).sum():,} / {len(var):,}")

log("\n=== §4(3) AnnData の容量見込み ===")
tot = int(obs.nnz.sum())
log(f"  非ゼロ要素 {tot:,}")
log(f"  非圧縮 CSR (float32+int32+indptr): {(tot*8 + len(obs)*8)/1e9:.2f} GB")
log("  gzip h5ad の実測比（既存コーパス 132GB / 非ゼロ推定から）を当てるため、下の実測を参照")
