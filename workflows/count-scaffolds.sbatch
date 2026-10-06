#!/bin/bash
#SBATCH --job-name=mc-scaffold-count
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Count distinct Bemis-Murcko scaffolds in a SMILES parquet. No GPU and no model:
# it is RDKit over the corpus, parallel across the allocated cores.
#
#   sbatch --export=ALL,SCAFFOLD_PARQUET=<file.parquet>,SCAFFOLD_OUT=<file.json> \
#          workflows/count-scaffolds.sbatch
set -uo pipefail
: "${SCAFFOLD_PARQUET:?export SCAFFOLD_PARQUET to the parquet holding the SMILES}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== scaffold count on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "parquet: ${SCAFFOLD_PARQUET}"
"${PYTHON}" scripts/count_scaffolds.py --parquet "${SCAFFOLD_PARQUET}" \
    --column "${SCAFFOLD_COLUMN:-smiles}" \
    --processes "${SLURM_CPUS_PER_TASK:-32}" \
    ${SCAFFOLD_LIMIT:+--limit "${SCAFFOLD_LIMIT}"} \
    ${SCAFFOLD_OUT:+--out "${SCAFFOLD_OUT}"}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
