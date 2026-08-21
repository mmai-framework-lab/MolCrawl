#!/bin/bash
#SBATCH --job-name=mc-unit-tests
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Run the unit tests on a compute node (nothing heavier than pytest runs on the
# login node). Pass TESTS to narrow the selection:
#
#   sbatch --export=ALL,TESTS="tests/unit/test_pack_order.py" workflows/unit-tests.sbatch
set -eo pipefail
TESTS="${TESTS:-tests/unit}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

# A git worktree holds only tracked files, so the interpreter is not inside it. Point
# PYTHON at the main checkout's miniconda when running from one.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

echo "=== unit tests ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tests: ${TESTS}"
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${PYTHON}" -m pytest ${TESTS} -q && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only ever be 0, and a failing
# pytest would otherwise leave nothing at the end of the log.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
