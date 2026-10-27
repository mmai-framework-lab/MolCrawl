#!/bin/bash
#SBATCH --job-name=mc-corpus-items
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive 2026-10-27 §1: the item counts -- cells, records -- that sit beside the
# window counts audit_data_parameters.py already measures.
#
# No GPU work and no token column is read: a parquet row count comes from the
# footer and an arrow row count from the schema block, so a 76 GiB tree costs a few
# KB per file. The allocation is here because this cluster has one partition and
# nothing heavier than a shell runs on the login node.
#
# Roots come from the environment and the job stops when one is unset, because
# learning_source* is gitignored and so differs per checkout.
#
#   sbatch --export=ALL,RNA_PARQUET_DIR=<...>,MOLNL_ARROW_DIR=<...> \
#          workflows/audit-corpus-item-counts.sbatch
set -eo pipefail

: "${RNA_PARQUET_DIR:?set RNA_PARQUET_DIR to the rna parquet_files directory}"
: "${MOLNL_ARROW_DIR:?set MOLNL_ARROW_DIR to the molecule_nat_lang arrow_splits directory}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

[ -d "${RNA_PARQUET_DIR}" ] || { echo "not a directory: ${RNA_PARQUET_DIR}" >&2; exit 1; }
[ -d "${MOLNL_ARROW_DIR}" ] || { echo "not a directory: ${MOLNL_ARROW_DIR}" >&2; exit 1; }

REPORT="${REPORT:-${REPORT_DIR:-${PYROOT}/tmp}/corpus-item-counts-${SLURM_JOB_ID}.json}"

echo "=== corpus item counts ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "rna   : ${RNA_PARQUET_DIR}"
echo "molnl : ${MOLNL_ARROW_DIR}"
echo "report: ${REPORT}"
echo

"${PYTHON}" scripts/audit_corpus_item_counts.py \
  --parquet-dir "rna_cells=${RNA_PARQUET_DIR}" \
  --arrow "molnl_train=${MOLNL_ARROW_DIR}/train.arrow" \
  --arrow "molnl_valid=${MOLNL_ARROW_DIR}/valid.arrow" \
  --arrow "molnl_test=${MOLNL_ARROW_DIR}/test.arrow" \
  --report "${REPORT}" && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
