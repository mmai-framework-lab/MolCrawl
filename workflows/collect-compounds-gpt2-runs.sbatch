#!/bin/bash
#SBATCH --job-name=mc-cmp-gpt2-runs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# List every compounds GPT-2 run from the logs the runs wrote, and dump every
# evaluation point. No GPU: it reads text and writes a TSV. It runs here rather
# than on the login node because the logs are hundreds of megabytes.
#
#   sbatch --export=ALL,RUN_LOGS='<dir>/mc-cmp-lrsweep-*.out <dir>/mc-cmp-seedvar-*.out',\
# RUN_TSV=<file.tsv> workflows/collect-compounds-gpt2-runs.sbatch
set -uo pipefail
: "${RUN_LOGS:?set RUN_LOGS to the log glob(s), space separated}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds GPT-2 run inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 -- RUN_LOGS is a list of globs on purpose
"${PYTHON}" scripts/collect_compounds_gpt2_runs.py --logs ${RUN_LOGS} ${RUN_TSV:+--tsv "${RUN_TSV}"}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
