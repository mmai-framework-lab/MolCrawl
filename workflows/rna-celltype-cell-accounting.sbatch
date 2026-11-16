#!/bin/bash
#SBATCH --job-name=rna-celltype-accounting
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Count the cells on each side of the cell-type probe's split, and the ones the
# evaluator sets aside before the macro average.
#
# No GPU: it reads one JSONL and counts. It is still not login-node work -- the
# evaluation JSONL is about a gigabyte.
#
# ACCOUNTING_OUT_TSV must name a file beginning with "rt1-": the TSV follows the
# report column convention (docs/07-reports/report_tsv_columns.md) and the script
# refuses a name that does not claim it.
#
#   sbatch --export=ALL,ACCOUNTING_JSONL=<eval.jsonl>,ACCOUNTING_OUT_TSV=<rt1-....tsv> \
#          workflows/rna-celltype-cell-accounting.sbatch
set -eo pipefail
: "${ACCOUNTING_JSONL:?export ACCOUNTING_JSONL to the evaluation JSONL the probe reads}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
    PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== rna celltype accounting on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${PYTHON}" scripts/rna_celltype_cell_accounting.py \
    --jsonl "${ACCOUNTING_JSONL}" \
    ${ACCOUNTING_MIN_CELLS:+--min-cells-per-label "${ACCOUNTING_MIN_CELLS}"} \
    ${ACCOUNTING_OUT_TSV:+--out-tsv "${ACCOUNTING_OUT_TSV}"}
echo "=== done @ $(date -Is) ==="
