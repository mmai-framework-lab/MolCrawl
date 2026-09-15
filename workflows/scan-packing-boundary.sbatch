#!/bin/bash
#SBATCH --job-name=mc-boundary-scan
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Does the id document masking would split on actually appear in the packed data?
#
#   sbatch --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/compounds/bert_small.py,\
# LEARNING_SOURCE_DIR=... workflows/scan-packing-boundary.sbatch
#
# main.py asks the same question, but only after the job has queued, a node has been
# allocated and the dataset has loaded. Asking here costs minutes instead of a
# reservation, and reports what the packer did write when the resolved id is absent.
# Read-only: nothing is trained and nothing is written outside the log.
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:?set CONFIG to the BERT config to check}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root the config reads}"
export LEARNING_SOURCE_DIR
SPLIT="${SPLIT:-train}"
ROWS="${ROWS:-2000}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}" || exit 1
mkdir -p workflows/slurm-logs

# A git worktree holds only tracked files, so the interpreter is not inside it. Point
# PYTHON at the main checkout's miniconda when running from one.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

echo "=== packing boundary scan ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${PYTHON}" scripts/scan_packing_boundary.py --config "${CONFIG}" --split "${SPLIT}" --rows "${ROWS}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
