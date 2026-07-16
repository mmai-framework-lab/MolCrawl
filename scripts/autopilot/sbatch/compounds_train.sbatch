#!/bin/bash
#SBATCH --job-name=cmp-train
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-cmp-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-cmp-%x-%j.out

# Compounds pretrain (BERT or GPT-2, any size) via existing workflow shell.
#
# Usage:
#   sbatch --job-name=cmp-bert-small \
#          --export=WORKFLOW=03c-compounds-train-bert-small.sh \
#          tmp/scripts/autopilot/sbatch/compounds_train.sbatch
#
# WORKFLOW: workflows/<script>.sh (must exist).
# LEARNING_SOURCE_DIR is exported below unconditionally so autopilot runs land
# under a fresh output tree; the existing common_functions.sh check passes.

set -euo pipefail

cd /lustre/home/matsubara/riken-dataset-fundational-model

: "${WORKFLOW:?WORKFLOW must be set via --export=WORKFLOW=03a-*.sh}"

WORKFLOW_PATH="workflows/${WORKFLOW}"
if [ ! -f "${WORKFLOW_PATH}" ]; then
    echo "ERROR: workflow not found: ${WORKFLOW_PATH}" >&2
    exit 2
fi

# autopilot output tree — separate from any hand-driven runs.
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_autopilot_v2}"
export NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "${LEARNING_SOURCE_DIR}/compounds/logs"

echo "=== autopilot compounds train ==="
echo "node       : $(hostname)"
echo "SLURM_JOB  : ${SLURM_JOB_ID}"
echo "workflow   : ${WORKFLOW_PATH}"
echo "LSD        : ${LEARNING_SOURCE_DIR}"
echo "NUM_GPUS   : ${NUM_GPUS}"
echo "date       : $(date -Iseconds)"

bash "${WORKFLOW_PATH}"

echo "=== workflow finished ==="
echo "exit=$?, date=$(date -Iseconds)"
