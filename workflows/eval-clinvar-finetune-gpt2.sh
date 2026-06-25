#!/bin/bash
# SLURM wrapper: fine-tune ClinVar pathogenicity classifier on every
# GPT-2 subset checkpoint. Mirrors eval-clinvar-finetune-bert.sh.
#
# Env knobs (forwarded to scripts/run_clinvar_finetune_gpt2_subsets.py):
#   SEEDS / SUBSETS / NUM_TRAIN_STEPS / LEARNING_RATE /
#   PER_DEVICE_BATCH_SIZE / MAX_LENGTH / MAX_TRAIN_ROWS / MAX_TEST_ROWS /
#   TEST_CHROMS / CLINVAR_DATA / DEVICE / OUT_TAG
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   sbatch --gres=gpu:1 workflows/eval-clinvar-finetune-gpt2.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/workflows/common_functions.sh" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/workflows"
fi
source "${SCRIPT_DIR}/common_functions.sh"
check_learning_source_dir

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEEDS=${SEEDS:-1,2,3}
SUBSETS=${SUBSETS:-}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-192}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-20000}
MAX_TEST_ROWS=${MAX_TEST_ROWS:-4000}
TEST_CHROMS=${TEST_CHROMS:-8,X,Y}
CLINVAR_DATA=${CLINVAR_DATA:-}
DEVICE=${DEVICE:-cuda}
OUT_TAG=${OUT_TAG:-}

LOG_DIR="${LEARNING_SOURCE_DIR}/genome_sequence/logs"
mkdir -p "$LOG_DIR"
LOG_TAG_SUFFIX=""
[ -n "$OUT_TAG" ] && LOG_TAG_SUFFIX="-${OUT_TAG}"
LOG_FILE="${LOG_DIR}/clinvar-finetune-gpt2${LOG_TAG_SUFFIX}-$(date +%Y-%m-%d_%H-%M-%S).log"

EXTRA_ARGS=()
[ -n "$SUBSETS" ]      && EXTRA_ARGS+=(--subsets "$SUBSETS")
[ -n "$CLINVAR_DATA" ] && EXTRA_ARGS+=(--clinvar-data "$CLINVAR_DATA")
[ -n "$OUT_TAG" ]      && EXTRA_ARGS+=(--out-tag "$OUT_TAG")

echo "========================================"
echo "ClinVar fine-tune GPT-2 classifier"
echo "========================================"
echo "GPU:               ${CUDA_VISIBLE_DEVICES}"
echo "seeds:             ${SEEDS}"
echo "subsets:           ${SUBSETS:-auto-detect}"
echo "num_train_steps:   ${NUM_TRAIN_STEPS}"
echo "learning_rate:     ${LEARNING_RATE}"
echo "batch_size:        ${PER_DEVICE_BATCH_SIZE}"
echo "max_length:        ${MAX_LENGTH}"
echo "test_chroms:       ${TEST_CHROMS}"
echo "clinvar_data:      ${CLINVAR_DATA:-<runner default>}"
echo "out_tag:           ${OUT_TAG:-<none>}"
echo "log:               ${LOG_FILE}"
echo

run_training_background "$LOG_FILE" \
    scripts/run_clinvar_finetune_gpt2_subsets.py \
    --seeds "${SEEDS}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --max-train-rows "${MAX_TRAIN_ROWS}" \
    --max-test-rows "${MAX_TEST_ROWS}" \
    --test-chroms "${TEST_CHROMS}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
