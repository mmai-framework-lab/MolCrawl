#!/bin/bash
#SBATCH --job-name=mc-cmp-curve-tsv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Build the unified curve TSV on a compute node. No GPU: it reads trainer_state.json
# and nanoGPT logs and writes a TSV. It runs here rather than on the login node
# because the nanoGPT logs are hundreds of megabytes.
#
#   sbatch --export=ALL,CURVE_ARGS='--bert dir=...,size=...,... --gpt2-logs ... --out ...' \
#          workflows/compounds-curve-tsv.sbatch
set -uo pipefail
: "${CURVE_ARGS:?set CURVE_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds curve TSV on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# Split CURVE_ARGS on whitespace but do NOT expand globs in it: the log pattern is
# meant to reach the script, which globs it itself. Left to the shell it expands
# into hundreds of file names, argparse takes them as stray positionals, and the
# run fails on a missing --out that was in fact supplied.
set -f
# shellcheck disable=SC2086 -- CURVE_ARGS is an argument list on purpose
"${PYTHON}" scripts/build_compounds_curve_tsv.py ${CURVE_ARGS}
rc=$?
set +f
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
