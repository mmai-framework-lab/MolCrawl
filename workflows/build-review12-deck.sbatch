#!/bin/bash
#SBATCH --job-name=mc-build-deck
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Render the progress deck on a compute node.
#
# No GPU is requested -- the job only renders matplotlib PNGs and writes .pptx --
# but compute does not belong on the login node here, and the partition accepts a
# job without a GPU.
#
# The content file and the output directory are BOTH outside any checkout: the
# deck carries internal numbers and this repository is public. Neither path is
# written into this file; both must be passed in, and the job fails loudly when
# they are not.
#
#   sbatch --export=ALL,DECK_CONTENT=<file.json>,DECK_OUT_DIR=<dir> \
#          workflows/build-review12-deck.sbatch
set -eo pipefail

: "${DECK_CONTENT:?set DECK_CONTENT to the JSON content file (kept outside any checkout)}"
: "${DECK_OUT_DIR:?set DECK_OUT_DIR to the directory the .pptx files are written into}"
[ -f "${DECK_CONTENT}" ] || { echo "DECK_CONTENT=${DECK_CONTENT} is not a file" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# A git worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== deck build on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "content: ${DECK_CONTENT}"
echo "outputs: ${DECK_OUT_DIR}"
rc=0
"${PYTHON}" scripts/build_review12_deck.py \
    --content "${DECK_CONTENT}" --out-dir "${DECK_OUT_DIR}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
