#!/bin/bash
#SBATCH --job-name=mc-ladder-ppl
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Deterministic whole-split loss / perplexity for a GPT-2 size ladder.
#
# Use it for two things:
#   SPLIT=test   the held-out number (checkpoints were selected on valid)
#   SPLIT=valid  a noise-free replacement for best_val, which is a minimum over
#                eval points estimated from a size-dependent number of sequences
#                and therefore carries a size-dependent selection bias
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,MODALITY=compounds,SPLIT=valid \
#     workflows/eval-ladder-perplexity-gpt2.sbatch
#
# RNA reads the flat uint16 memmap its training runs use, so it needs RNA_BIN_DIR:
#   LEARNING_SOURCE_DIR=<dir> RNA_BIN_DIR=<dir> \
#     sbatch --export=ALL,MODALITY=rna,SPLIT=valid,SIZES="small medium large" ... 
set -e
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR before submitting}"
MODALITY="${MODALITY:-compounds}"
SIZES="${SIZES:-small medium large ex-large}"
SPLIT="${SPLIT:-test}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"   # 0 = whole split
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/experiment_data/eval/ladder-perplexity}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT}" workflows/slurm-logs

echo "=== ${MODALITY} GPT-2 ${SPLIT} perplexity ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "sizes: ${SIZES} | split=${SPLIT} | max_blocks=${MAX_BLOCKS}"
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${REPO}/miniconda/bin/python" scripts/eval_gpt2_perplexity.py \
  --modality "${MODALITY}" --sizes ${SIZES} --split "${SPLIT}" \
  --max-blocks "${MAX_BLOCKS}" --output-dir "${OUT}" \
  ${RNA_BIN_DIR:+--rna-bin-dir "${RNA_BIN_DIR}"} \
  ${RNA_VOCAB_FILE:+--rna-vocab-file "${RNA_VOCAB_FILE}"} \
  ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"} \
  ${CKPT_TEMPLATE:+--checkpoint-template "${CKPT_TEMPLATE}"}
echo "=== rc=$? @ $(date -Is) ==="
