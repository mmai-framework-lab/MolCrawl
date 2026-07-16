#!/bin/bash
#SBATCH --job-name=sub-train
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-sub-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-sub-%x-%j.out

# Genome subset small training (21 subset × BERT/GPT-2 = 42 run).
#
# Env inputs:
#   WORKFLOW       — workflows/03[ac]-genome_sequence-train-*-small-subset.sh
#   GENOME_SUBSET  — mammal_centered / eukaryote_matched_random_seedN / global_random_seedN
#   NUM_GPUS       — default 4
#
# LEARNING_SOURCE_DIR must point at the G2 v2 output tree so the config resolves
# training_ready_hf_dataset_{bert,gpt2} correctly.

set -euo pipefail

cd /lustre/home/matsubara/riken-dataset-fundational-model

: "${WORKFLOW:?WORKFLOW must be set}"
: "${GENOME_SUBSET:?GENOME_SUBSET must be set}"

WORKFLOW_PATH="workflows/${WORKFLOW}"
if [ ! -f "${WORKFLOW_PATH}" ]; then
    echo "ERROR: workflow not found: ${WORKFLOW_PATH}" >&2
    exit 2
fi

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_genome_v2}"
export NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs"

echo "=== autopilot subset train ==="
echo "node         : $(hostname)"
echo "SLURM_JOB    : ${SLURM_JOB_ID}"
echo "workflow     : ${WORKFLOW_PATH}"
echo "GENOME_SUBSET: ${GENOME_SUBSET}"
echo "LSD          : ${LEARNING_SOURCE_DIR}"
echo "NUM_GPUS     : ${NUM_GPUS}"
echo "date         : $(date -Iseconds)"

bash "${WORKFLOW_PATH}"

echo "=== workflow finished ==="
echo "exit=$?, date=$(date -Iseconds)"
