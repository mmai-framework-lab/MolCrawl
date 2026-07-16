#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-smoke-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-smoke-%x-%j.out

# subset smoke — SMOKE_MAX_STEPS=5 で 5 iter (early_stop OFF、 warmup=2) だけ回して
# training path が通ることを readiness で証跡化する。
# 全 42 config は同じ template を使うので、 arch × subset の代表 1 本ずつで十分。
#
# Usage:
#   sbatch --job-name=smoke-bert-seed6 \
#          --export=WORKFLOW=03c-genome_sequence-train-bert-small-subset.sh,\
#                   GENOME_SUBSET=global_random_seed6,\
#                   SMOKE_MAX_STEPS=5,NUM_GPUS=4 \
#          tmp/scripts/autopilot/sbatch/subset_smoke.sbatch

set -euo pipefail
cd /lustre/home/matsubara/riken-dataset-fundational-model

: "${WORKFLOW:?WORKFLOW must be set}"
: "${GENOME_SUBSET:?GENOME_SUBSET must be set}"
: "${SMOKE_MAX_STEPS:=5}"

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_genome_v2}"
export NUM_GPUS="${NUM_GPUS:-4}"
export SMOKE_MAX_STEPS

mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs"

echo "=== autopilot subset SMOKE ==="
echo "node             : $(hostname)"
echo "SLURM_JOB        : ${SLURM_JOB_ID}"
echo "workflow         : workflows/${WORKFLOW}"
echo "GENOME_SUBSET    : ${GENOME_SUBSET}"
echo "SMOKE_MAX_STEPS  : ${SMOKE_MAX_STEPS}"
echo "LSD              : ${LEARNING_SOURCE_DIR}"
echo "date             : $(date -Iseconds)"

bash "workflows/${WORKFLOW}"

echo "=== smoke finished ==="
echo "exit=$?, date=$(date -Iseconds)"
