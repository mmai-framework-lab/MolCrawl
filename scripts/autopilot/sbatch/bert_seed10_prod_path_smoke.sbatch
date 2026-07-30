#!/bin/bash
#SBATCH --job-name=diag-bert-seed10-prodpath
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/docs_tmp_local/seed10_prodpath_out_%j.log
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/docs_tmp_local/seed10_prodpath_err_%j.log

# BERT seed10 の production 経路 diagnostic:
# SMOKE_MAX_STEPS は使わず (=production code path)、 HARD_MAX_STEPS_OVERRIDE=5000 で
# max_steps だけ短縮 (warmup_steps は production の ~2% = ~1940 のまま残る)。
#
# これで:
# - 起動経路: production と完全一致 (warmup 計算・eval_interval も production 値)
# - max_steps: 5000 (production 97,116 の ~5%、 wall time ~15 分に time-box)
#
# もし production と同じく 1-2 分で NODE_FAIL するなら systematic (config 起因ではない可能性大、
# 起動 racing / node prolog 系)。 5000 step 完走なら production も transient と示唆される。
#
# 用途: 23170 (retry3) と並行させ、 別ノードで独立 data point 取得

set -x
set -euo pipefail

echo "=== DIAG: BERT seed10 production-path (HARD_MAX_STEPS_OVERRIDE=5000) ==="
echo "node          : $(hostname)"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID}"
echo "date          : $(date -Iseconds)"

nvidia-smi -L
free -h
echo "SLURM allocated node: ${SLURM_JOB_NODELIST:-unset}"
echo ""

cd /lustre/home/matsubara/riken-dataset-fundational-model
source /lustre/home/matsubara/miniforge3/etc/profile.d/conda.sh
conda activate molcrawl

# 起動直後 python + torch check
python -c "
import sys, torch, os
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda_avail:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
"

# Debug env
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Production 経路: SMOKE_MAX_STEPS は unset のまま、 HARD_MAX_STEPS_OVERRIDE のみ設定
export HARD_MAX_STEPS_OVERRIDE=5000
# unset SMOKE_MAX_STEPS SMOKE_WARMUP_STEPS  # 念のため

# Actual training env (mirrors production exactly except max_steps time-box)
export LEARNING_SOURCE_DIR=/lustre/home/matsubara/learning_source_20260710_genome_v2
export NUM_GPUS=4
export GENOME_SUBSET=eukaryote_matched_random_seed10
export BERT_LR_TAG=diagprodpath  # 出力を別 dir に

echo ""
echo "--- launching BERT small subset training (production path, max_steps=5000) ---"
echo "HARD_MAX_STEPS_OVERRIDE=$HARD_MAX_STEPS_OVERRIDE"
echo "BERT_LR_TAG=$BERT_LR_TAG"
echo "date before workflow: $(date -Iseconds)"

bash workflows/03c-genome_sequence-train-bert-small-subset.sh 2>&1

echo ""
echo "--- workflow finished ---"
echo "exit=$?, date=$(date -Iseconds)"
