#!/bin/bash
#SBATCH --job-name=smoke-bert-seed10-debug
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/docs_tmp_local/seed10_smoke_out_%j.log
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/docs_tmp_local/seed10_smoke_err_%j.log

# BERT eukaryote_matched_random_seed10 の 2 連続 NODE_FAIL (23030 gpu03, 23121 gpu01)
# の startup crash 原因を pinpoint するための debug smoke。
# heavy verbose (NCCL_DEBUG=INFO, PYTHONFAULTHANDLER=1, python -u) で
# stderr/stdout を別ファイルに落として、 crash 直前の状態を捕捉する。
#
# 走行時間は最大 30 分に制限、 SMOKE_MAX_STEPS=100 で早期完了 or 早期 crash を狙う。

set -x  # trace 全 shell command
set -euo pipefail

echo "=== SMOKE DEBUG for BERT eukaryote_matched_random_seed10 ==="
echo "node          : $(hostname)"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID}"
echo "date          : $(date -Iseconds)"
echo "cwd           : $(pwd)"

# System-level diagnostics before Python
echo "--- system diagnostics ---"
nvidia-smi -L
free -h
df -h /lustre/home/matsubara 2>&1 | head -5
ulimit -a | head -20
echo "SLURM allocated GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM allocated node: ${SLURM_JOB_NODELIST:-unset}"
echo ""

cd /lustre/home/matsubara/riken-dataset-fundational-model
source /lustre/home/matsubara/miniforge3/etc/profile.d/conda.sh
conda activate molcrawl

# Python + torch quick check
python -c "
import sys, torch, os
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda_avail:', torch.cuda.is_available())
print('cuda_devices:', torch.cuda.device_count())
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'))
"

# Dataset load standalone check (must succeed before starting training)
echo "--- dataset load standalone check ---"
python -X faulthandler -u -c "
from datasets import load_from_disk
d = load_from_disk('/lustre/home/matsubara/learning_source_20260710_genome_v2/genome_sequence/eukaryote_matched_random_seed10/training_ready_hf_dataset_bert')
print('OK loaded:', list(d.keys()))
for k in d: print(f'  {k}: {len(d[k]):,} rows')
print('sample row keys:', list(d['train'][0].keys()))
print('input_ids length:', len(d['train'][0]['input_ids']))
"

# Debug env
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Smoke mode: force short run to fail fast if there's a startup issue
export SMOKE_MAX_STEPS=100
export SMOKE_WARMUP_STEPS=10

# Actual training env (mirrors production but with SMOKE overrides)
export LEARNING_SOURCE_DIR=/lustre/home/matsubara/learning_source_20260710_genome_v2
export NUM_GPUS=4
export GENOME_SUBSET=eukaryote_matched_random_seed10

echo ""
echo "--- launching BERT small subset training with 4 GPU torchrun ---"
echo "SMOKE_MAX_STEPS=$SMOKE_MAX_STEPS, SMOKE_WARMUP_STEPS=$SMOKE_WARMUP_STEPS"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "date before workflow: $(date -Iseconds)"

# stderr を stdout に合流 + 全 stdout を tee で file にも書く
bash workflows/03c-genome_sequence-train-bert-small-subset.sh 2>&1

echo ""
echo "--- workflow finished ---"
echo "exit=$?, date=$(date -Iseconds)"
