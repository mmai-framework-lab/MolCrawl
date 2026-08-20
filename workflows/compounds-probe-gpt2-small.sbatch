#!/bin/bash
#SBATCH --job-name=mc-cmp-probe
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/mc-cmp-probe-%j.out
#SBATCH --error=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/mc-cmp-probe-%j.out
# 手順3 probe: gpt2-small on the v4 packed data, 60 iters only.
# Checks: tokens per iteration == 2,621,440 / s-per-iter for the GPU-h estimate /
# no divergence at LR 6e-4. Writes to a throwaway out_dir, never the ladder dir.
set -e
REPO="/data1/rkp00024/matsubara/MolCrawl"
NEW="${REPO}/learning_source_20260805_compounds_packed"
OUT="${NEW}/runs/probe-gpt2-small"
export LEARNING_SOURCE_DIR="${NEW}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/miniconda/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"
mkdir -p "${OUT}"
echo "=== compounds probe (gpt2-small, v4 packed) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
"${REPO}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py molcrawl/tasks/pretrain/configs/compounds/gpt2_small.py \
  --init_from=scratch --max_iters=60 --lr_decay_iters=1558 --eval_interval=20 \
  --eval_sequences=320 \
  --out_dir="${OUT}" --compile=False --tensorboard=False
echo "=== rc=$? @ $(date -Is) ==="
