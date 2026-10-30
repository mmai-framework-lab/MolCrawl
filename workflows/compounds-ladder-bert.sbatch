#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# compounds BERT ladder on the v4 packed data (10 epochs, global batch 2,560).
# Submit with:  sbatch --job-name=mc-cmp-bert-<size> --time=<hh:mm:ss> \
#                      --export=ALL,SIZE=<size> workflows/compounds-ladder-bert.sbatch
# model_path comes from the config (get_bert_output_path -> LEARNING_SOURCE_DIR),
# which is a fresh dir under the packed learning source => trains from scratch.
# Eval runs on valid (bert/main.py resolve_eval_split default) — test stays held out.
set -e
: "${SIZE:?SIZE must be set (small|medium|large)}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$PWD"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the packed compounds learning_source root (learning_source_20260805_compounds_packed in the main checkout)}"
export LEARNING_SOURCE_DIR
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH}"
export HF_DATASETS_CACHE="${REPO}/tmp/hf-cache-compounds"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}"
echo "=== compounds LADDER bert-${SIZE} (v4 packed) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "branch: $(git -C "${REPO}" rev-parse --abbrev-ref HEAD)  head: $(git -C "${REPO}" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
"${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "molcrawl/tasks/pretrain/configs/compounds/bert_${SIZE}.py"
echo "=== rc=$? @ $(date -Is) ==="
