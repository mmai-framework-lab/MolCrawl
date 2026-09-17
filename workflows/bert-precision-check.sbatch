#!/bin/bash
#SBATCH --job-name=mc-bert-precision
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# One arm of the fp32 / bf16 loss check on rna BERT small (all-bert-order-2026-09-17 §9).
#
# The two arms differ in precision only: same config, same seed (so the same data order
# and the same masked positions), four dataloader workers and pinning in both. They run
# the 3-epoch schedule -- max_steps 40,320, warmup 4,032 -- so both pass the learning-rate
# peak at step 4,032 and train 4,000 steps beyond it, and they end at step 8,000 through
# stop_at_step. max_steps is NOT shortened to 8,000: HF decays the learning rate over
# max_steps, so that would be a different schedule.
#
# The config is named first and the overrides after it. configurator.py applies argv in
# order, so a config file placed after an override would silently take its value back.
#
#   sbatch --export=ALL,PRECISION=bf16,LEARNING_SOURCE_DIR=<rna input root>,\
# MODEL_OUTPUT_ROOT=<output root> workflows/bert-precision-check.sbatch
set -eo pipefail
: "${PRECISION:?PRECISION=fp32 or bf16}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the root outputs go under, outside the input tree}"
case "${PRECISION}" in fp32) BF16=False;; bf16) BF16=True;; *) echo "PRECISION must be fp32 or bf16" >&2; exit 1;; esac
[ -d "${LEARNING_SOURCE_DIR}/rna" ] || { echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no rna/" >&2; exit 1; }
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }

CFG=molcrawl/tasks/pretrain/configs/rna/bert_small.py
OUT="${MODEL_OUTPUT_ROOT%/}/rna/bert-precision-check/bert_small-${PRECISION}"
# Everything but precision is fixed here and identical for both arms.
ARGS=(--model_path="${OUT}" --max_steps=40320 --warmup_steps=4032 --stop_at_step=8000 --bf16="${BF16}")

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-rna}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs

"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what="model_path")' "${OUT}" || exit 1
if [ -n "$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | head -1)" ]; then
  echo "REFUSING: ${OUT} already holds checkpoints; this check starts from step 0. Move it aside." >&2
  exit 1
fi
mkdir -p "${OUT}"

echo "=== rna bert_small precision check: ${PRECISION} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG}   overrides: ${ARGS[*]}"
echo "outputs: ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 molcrawl/models/bert/main.py "${CFG}" "${ARGS[@]}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
