#!/bin/bash
#SBATCH --job-name=mc-sxs-scan
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Find the tables that put two architectures side by side. No GPU: it reads markdown
# and TSV. It runs here rather than on the login node because the deliverable
# directory holds TSVs of a hundred thousand rows.
#
#   sbatch --export=ALL,SXS_ARGS='--root DIR --out FILE [--modality RE]' \
#          workflows/arch-side-by-side-scan.sbatch
set -uo pipefail
: "${SXS_ARGS:?set SXS_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== arch side-by-side scan on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 -- SXS_ARGS is an argument list on purpose
"${PYTHON}" scripts/arch_side_by_side_scan.py ${SXS_ARGS}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
