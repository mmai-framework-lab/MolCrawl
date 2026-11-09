#!/bin/bash
#SBATCH --job-name=mc-protein-pg-rebuild
#SBATCH --partition=gpu
#SBATCH --account=rkp00024
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Rebuild the protein ProteinGym tables (per-assay, per-model, reached-loss) and the
# train/val loss-curve TSV. protein-judgment-2026-11-09 §5 (seed-42 series).
#
# No GPU: it reads per-assay metrics.json, trainer_state.json and logging_*.csv from
# completed runs and writes TSVs. It runs on a compute node, not the login node.
#
#   OUT=<out_dir> sbatch workflows/protein-proteingym-rebuild.sbatch
set -uo pipefail

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

echo "=== protein ProteinGym/curve rebuild on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "OUT=${OUT}"
rc=0
# protein_proteingym_tables.py writes per-assay, per-model and the reached-loss table
# (the reconcile TSV/script it replaced was retired in protein-judgment-2026-11-09 §5).
"${PYTHON}" scripts/protein_proteingym_tables.py "${OUT}" || rc=$?
echo "--- tables rc=${rc} ---"
"${PYTHON}" scripts/protein_loss_curves_trainval.py "${OUT}" || rc=$?
echo "--- curves rc=${rc} ---"
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
