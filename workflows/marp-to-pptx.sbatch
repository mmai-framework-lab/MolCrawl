#!/bin/bash
#SBATCH --job-name=mc-marp-pptx
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Convert a Marp deck to .pptx on a compute node. No GPU is requested.
#
# The markdown and the output directory are BOTH outside any checkout: the deck
# carries internal numbers and this repository is public.
#
#   sbatch --export=ALL,DECK_MD=<deck.md>,DECK_OUT_DIR=<dir>[,DECK_FOOTER=<text>] \
#          workflows/marp-to-pptx.sbatch
set -eo pipefail

: "${DECK_MD:?set DECK_MD to the Marp markdown file}"
: "${DECK_OUT_DIR:?set DECK_OUT_DIR to the directory the .pptx is written into}"
[ -f "${DECK_MD}" ] || { echo "DECK_MD=${DECK_MD} is not a file" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# scripts/ holds the renderer that marp_to_pptx imports.
export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== marp -> pptx on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "md:      ${DECK_MD}"
echo "outputs: ${DECK_OUT_DIR}"
rc=0
"${PYTHON}" scripts/marp_to_pptx.py --md "${DECK_MD}" --out-dir "${DECK_OUT_DIR}" \
    --base-dir "$(dirname "${DECK_MD}")" --footer "${DECK_FOOTER:-}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
