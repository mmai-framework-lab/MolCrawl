#!/bin/bash
# 評価用の依存を、学習が使う環境とは別のツリーに入れる。
#
# scib-metrics は numpy を 2.x に上げ、scvi-tools は torch と CUDA ランタイムを
# 引き込む。学習側の環境に同居させると、走行中の run の足元が変わる。
#
# aarch64 では scikit-misc（scvi-tools が要求）の wheel が 1 つも公開されておらず
# ソースビルドになる。その際 meson が version_please.py を shebang 経由で起動する
# ため、`python` という名前の実行ファイルが PATH に無いと status 127 で落ちる。
# ノードには python3 しか無いことがあるので、このジョブの PATH の先頭にだけ
# symlink を置く。環境そのものは書き換えない。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=rna-eval-env
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${EVAL_ENV_DIR:?export EVAL_ENV_DIR to the directory the evaluation venv should live in (not under \$HOME if it has a quota)}"
: "${RNA_WORK_DIR:?export RNA_WORK_DIR to a writable directory for the version record}"

if [ -x "./miniconda/bin/python" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
BASE_PY="${PYROOT}/miniconda/bin/python"
[ -x "${BASE_PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${RNA_WORK_DIR}/tmp}"
mkdir -p "${TMPDIR}" "${RNA_WORK_DIR}" workflows/slurm-logs
echo "=== rna-eval-env @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "base python (left untouched): $("${BASE_PY}" -V) at ${PYROOT}/miniconda"
echo "building into              : ${EVAL_ENV_DIR}"

"${BASE_PY}" -m venv "${EVAL_ENV_DIR}"
ENV_PY="${EVAL_ENV_DIR}/bin/python"
"${ENV_PY}" -m pip install --quiet --upgrade pip

SHIM="${TMPDIR}/pyshim-${SLURM_JOB_ID:-$$}"
mkdir -p "${SHIM}"
ln -sf "${ENV_PY}" "${SHIM}/python"
export PATH="${SHIM}:${PATH}"
echo "shim: $(command -v python)"
echo "gfortran: $(command -v gfortran || echo 'MISSING -- scikit-misc will not build')"

echo "--- metrics and the HVG control ---"
"${ENV_PY}" -m pip install --no-input scib-metrics harmonypy scanpy anndata 2>&1 | tail -3
echo "--- scikit-misc, then the scVI control ---"
"${ENV_PY}" -m pip install --no-input scikit-misc 2>&1 | tail -3
"${ENV_PY}" -m pip install --no-input scvi-tools 2>&1 | tail -3

echo "--- pin the versions a report has to cite ---"
"${ENV_PY}" -m pip freeze > "${RNA_WORK_DIR}/eval-env-freeze.txt"
grep -iE "^(scib-metrics|scvi-tools|scikit-misc|jax|jaxlib|harmonypy|scanpy|anndata|numpy|scipy|torch|scikit-learn)==" \
  "${RNA_WORK_DIR}/eval-env-freeze.txt" | sed 's/^/  /'

echo "--- which metrics are actually callable ---"
"${ENV_PY}" - <<'PY'
import dataclasses as dc
import importlib.metadata as md

import scib_metrics as S
from scib_metrics.benchmark import BatchCorrection, BioConservation

print("  scib-metrics", md.version("scib-metrics"))
# A name in a paper is not a name in the package: report what is callable here
# and drop the rest from the metric list rather than carrying it unchecked.
for k in ["bras", "sbee", "cilisi", "CiLISI",
          "nmi_ari_cluster_labels_kmeans", "nmi_ari_cluster_labels_leiden",
          "clisi_knn", "ilisi_knn", "isolated_labels", "graph_connectivity",
          "kbet", "kbet_per_label", "pcr_comparison",
          "silhouette_label", "silhouette_batch"]:
    print(f"    {k:32s} {'callable' if callable(getattr(S, k, None)) else 'absent'}")
print("  BioConservation defaults:", [f.name for f in dc.fields(BioConservation)])
print("  BatchCorrection defaults:", [f.name for f in dc.fields(BatchCorrection)])

import scanpy as sc, numpy as np, anndata as ad
a = ad.AnnData(X=np.random.default_rng(0).poisson(3, (200, 300)).astype("float32"))
sc.pp.highly_variable_genes(a, n_top_genes=50, flavor="seurat_v3")
print(f"  seurat_v3 HVG selected: {int(a.var['highly_variable'].sum())}")
import scvi
print("  scvi", scvi.__version__)
PY
echo "=== done @ $(date -Is) ==="
