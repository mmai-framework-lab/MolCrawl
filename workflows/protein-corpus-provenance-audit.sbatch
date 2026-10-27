#!/bin/bash
#SBATCH --job-name=mc-protein-corpus-prov
#SBATCH --partition=gpu
#SBATCH --account=rkp00024
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Measure the protein_sequence (UniRef50) corpus provenance stage by stage, to
# confirm what the 38,794,121 figure counts and where (if anywhere) sequences are
# dropped. protein-uniref-order-2026-10-27.
#
# No GPU is requested -- it is streaming I/O: a 16.9 GiB FASTA header count, an
# md5 of the 8.7 GiB .gz, 388 raw files, 388 parquet footers, and the packed
# arrow shard row counts. It runs on a compute node, not the login node, because
# that read does not belong on the login node.
#
#   DATA_DIR=<.../protein_sequence> OUT=<out_dir> \
#     sbatch workflows/protein-corpus-provenance-audit.sbatch
set -uo pipefail

: "${DATA_DIR:?set DATA_DIR to the protein_sequence tree}"
: "${OUT:?set OUT to the output directory}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== protein corpus provenance on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT=${OUT}"
rc=0
"${PYTHON}" scripts/audit_protein_corpus_provenance.py \
    --data-dir "${DATA_DIR}" --out "${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
