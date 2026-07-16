#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-sweep-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-sweep-%x-%j.out

# genome subset small LR sweep — charter 2026-07-14 reply §「mini LR sweep」.
# Uses SUBSET_{BERT,GPT2}_LR env vars (subset template reads via os.environ)
# and SMOKE_MAX_STEPS to cap iteration count. BERT_LR_TAG / GPT2_LR_TAG make
# checkpoint dirs distinct across LR values so concurrent sweep runs on the
# same subset do not race onto the same output tree.
#
# Usage:
#   sbatch --job-name=sweep-bert-mammal-1e-4 \
#     --export=ALL,\
#          WORKFLOW=03c-genome_sequence-train-bert-small-subset.sh,\
#          GENOME_SUBSET=mammal_centered,\
#          SUBSET_BERT_LR=0.0001,BERT_LR_TAG=lr1e-4,\
#          SMOKE_MAX_STEPS=800,NUM_GPUS=4 \
#     tmp/scripts/autopilot/sbatch/subset_lr_sweep.sbatch

set -euo pipefail
cd /lustre/home/matsubara/riken-dataset-fundational-model

: "${WORKFLOW:?WORKFLOW must be set}"
: "${GENOME_SUBSET:?GENOME_SUBSET must be set}"
: "${SMOKE_MAX_STEPS:=800}"

# LR override: exactly one of {SUBSET_BERT_LR, SUBSET_GPT2_LR} + matching tag.
# Passed via --export so we don't need to set anything else here.
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_genome_v2}"
export NUM_GPUS="${NUM_GPUS:-4}"
export SMOKE_MAX_STEPS

mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs"

echo "=== autopilot subset LR sweep ==="
echo "node             : $(hostname)"
echo "SLURM_JOB        : ${SLURM_JOB_ID}"
echo "workflow         : workflows/${WORKFLOW}"
echo "GENOME_SUBSET    : ${GENOME_SUBSET}"
echo "SUBSET_BERT_LR   : ${SUBSET_BERT_LR:-<unset>}"
echo "BERT_LR_TAG      : ${BERT_LR_TAG:-<unset>}"
echo "SUBSET_GPT2_LR   : ${SUBSET_GPT2_LR:-<unset>}"
echo "GPT2_LR_TAG      : ${GPT2_LR_TAG:-<unset>}"
echo "SMOKE_MAX_STEPS  : ${SMOKE_MAX_STEPS}"
echo "date             : $(date -Iseconds)"

bash "workflows/${WORKFLOW}"

echo "=== sweep run finished ==="
echo "exit=$?, date=$(date -Iseconds)"
