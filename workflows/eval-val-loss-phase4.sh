#!/bin/bash
# SLURM wrapper: re-evaluate val_loss for every (subset, model) ckpt
# against the post-Phase-4-fix Arrow valid split.
#
# Env knobs:
#   EVAL_ITERS    — default 200 (batches per ckpt)
#   MODELS        — default "bert,gpt2"
#   SUBSETS       — default "" (= all 21)
#   DEVICE        — default "cuda"
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   sbatch --gres=gpu:1 workflows/eval-val-loss-phase4.sh
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

EVAL_ITERS=${EVAL_ITERS:-200}
MODELS=${MODELS:-bert,gpt2}
SUBSETS=${SUBSETS:-}
DEVICE=${DEVICE:-cuda}

LOG_DIR="${LEARNING_SOURCE_DIR}/genome_sequence/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/val-loss-phase4-$(date +%Y-%m-%d_%H-%M-%S).log"

EXTRA_ARGS=()
[ -n "$SUBSETS" ] && EXTRA_ARGS+=(--subsets "$SUBSETS")

echo "========================================"
echo "val_loss re-eval on Phase 4 fixed Arrow"
echo "========================================"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "eval_iters: ${EVAL_ITERS}"
echo "models:     ${MODELS}"
echo "subsets:    ${SUBSETS:-all 21}"
echo "device:     ${DEVICE}"
echo "log:        ${LOG_FILE}"
echo

run_training_background "$LOG_FILE" \
    scripts/eval_val_loss_phase4_arrow.py \
    --eval-iters "${EVAL_ITERS}" \
    --models "${MODELS}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
