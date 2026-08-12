#!/bin/bash
# RNA BERT pretraining launcher (rikyu / ai.r-ccs).
#
#   SIZE=small sbatch workflows/rna-bert.sbatch                      # production
#   SIZE=small SMOKE_MAX_STEPS=200 sbatch workflows/rna-bert.sbatch  # smoke
#
# Env knobs:
#   SIZE            required: small | medium | large
#   TAG             output-dir suffix for smoke runs (default: smoke)
#   SMOKE_MAX_STEPS if set, runs a short job into runs/smoke-* instead of the
#                   production model_path
#
# --gpus=4 is NOT a free choice here. HF Trainer's effective batch is
# per_device (8) x grad_accum (80) x world_size, so the 2560 spec only holds at
# 4 GPUs; changing the GPU count silently changes the amount of data the model
# sees. GPT-2 has no such coupling (nanoGPT divides grad_accum by world_size).
#SBATCH --account=rkp00024
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --job-name=rna-bert
#SBATCH --output=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/%x-%j.out
#SBATCH --error=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/%x-%j.out
set -e

# Code and data are separate: REPO is the checkout to run (a worktree when
# testing a branch), MAIN holds the conda env, the learning source and the
# log directory. Pinning both to the same path silently ran whatever branch
# the shared checkout happened to be on.
MAIN="/data1/rkp00024/matsubara/MolCrawl"
REPO="${REPO:-$MAIN}"
SIZE="${SIZE:?SIZE is required (small|medium|large)}"
TAG="${TAG:-smoke}"
CONFIG="molcrawl/tasks/pretrain/configs/rna/bert_${SIZE}.py"

export LEARNING_SOURCE_DIR="${MAIN}/learning_source_20260803_rna_train"

# Keep caches and temporaries off $HOME (quota) — they all go under /data1.
export HF_HOME="${MAIN}/tmp/hf-rna"
export HF_DATASETS_CACHE="${MAIN}/tmp/hf-rna/datasets"
export XDG_CACHE_HOME="${MAIN}/tmp/hf-rna/xdg"
export TMPDIR="${MAIN}/tmp/hf-rna/tmp"
export LD_LIBRARY_PATH="${MAIN}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1

cd "${REPO}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
mkdir -p "${MAIN}/workflows/slurm-logs" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

OVERRIDES=()
if [ -n "${SMOKE_MAX_STEPS:-}" ]; then
  OUT="${LEARNING_SOURCE_DIR}/rna/runs/${TAG}-bert-${SIZE}"
  OVERRIDES+=(--max_steps="${SMOKE_MAX_STEPS}" --save_steps=100)
else
  OUT="$("${MAIN}"/miniconda/bin/python -c "
from molcrawl.core.paths import get_bert_output_path
print(get_bert_output_path('rna', '${SIZE}'))")"
fi
OVERRIDES+=(--model_path="${OUT}")
mkdir -p "${OUT}"

echo "=== rna bert-${SIZE} ${SMOKE_MAX_STEPS:+SMOKE max_steps=${SMOKE_MAX_STEPS}} @ $(date -Is) on ${SLURMD_NODENAME} ==="
echo "repo:   ${REPO}"
echo "branch: $(git -C "${REPO}" rev-parse --abbrev-ref HEAD)  head: $(git -C "${REPO}" rev-parse --short HEAD)"
"${MAIN}"/miniconda/bin/python -c "import molcrawl; print('molcrawl from:', molcrawl.__file__)"
echo "config: ${CONFIG}"
echo "data:   ${LEARNING_SOURCE_DIR}/rna/training_ready_hf_dataset"
echo "out:    ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Sanity anchors to grep for in this log (see §5 of the instruction):
#   vocab_size 25,432   (= (25,426 + [UNK] + [PAD]) rounded up to a multiple of 8)
#   eval runs on the valid split — test stays held out (PR #98)
rc=0
"${MAIN}"/miniconda/bin/torchrun --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CONFIG}" "${OVERRIDES[@]}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
ls -d "${OUT}"/checkpoint-* 2>/dev/null | tail -3
exit $rc
