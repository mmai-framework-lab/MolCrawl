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
#   SPLIT=valid  a noise-free replacement for best_val, which is a minimum over eval
#                points each estimated from a with-replacement sample, so it sits
#                below the value it estimates
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,MODALITY=compounds,SPLIT=valid \
#     workflows/eval-ladder-perplexity-gpt2.sbatch
#
# RNA reads the flat uint16 memmap its training runs use, so it needs RNA_BIN_DIR:
#   LEARNING_SOURCE_DIR=<dir> RNA_BIN_DIR=<dir> sbatch \
#     --export=ALL,MODALITY=rna,SPLIT=valid,SIZES="small medium large" \
#     workflows/eval-ladder-perplexity-gpt2.sbatch
#
# protein and genome exclude ambiguous tokens from the training loss, and the script
# excludes the same ones, so those numbers stay on the scale of their own val losses.
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR before submitting}"
MODALITY="${MODALITY:-compounds}"
SIZES="${SIZES:-small medium large ex-large}"
SPLIT="${SPLIT:-test}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"   # 0 = whole split
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/experiment_data/eval/ladder-perplexity}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT}" workflows/slurm-logs

# A git worktree holds only tracked files, so the interpreter is not inside it. Point
# PYTHON at the main checkout's miniconda when running from one.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

echo "=== ${MODALITY} GPT-2 ${SPLIT} perplexity ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "sizes: ${SIZES} | split=${SPLIT} | max_sequences=${MAX_SEQUENCES}"
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${PYTHON}" scripts/eval_gpt2_perplexity.py \
  --modality "${MODALITY}" --sizes ${SIZES} --split "${SPLIT}" \
  --max-sequences "${MAX_SEQUENCES}" --output-dir "${OUT}" \
  ${RNA_BIN_DIR:+--rna-bin-dir "${RNA_BIN_DIR}"} \
  ${RNA_VOCAB_FILE:+--rna-vocab-file "${RNA_VOCAB_FILE}"} \
  ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"} \
  ${CKPT_TEMPLATE:+--checkpoint-template "${CKPT_TEMPLATE}"} \
  ${OUTPUT_NAME:+--output-name "${OUTPUT_NAME}"} \
  ${PAD_TOKEN_ID_FOR_LOSS:+--pad-token-id-for-loss "${PAD_TOKEN_ID_FOR_LOSS}"} \
  ${GENOME_TOKENIZER_MODEL:+--genome-tokenizer-model "${GENOME_TOKENIZER_MODEL}"} && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only ever be 0, because a
# failure never reaches the line.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
