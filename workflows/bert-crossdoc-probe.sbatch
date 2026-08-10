#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#SBATCH --time=06:00:00
# Cross-document attention probe.
#
#   V1  v3 padded      seq 128, one molecule per row  -> no cross-document context
#   V2  packed slices  seq 128, packed stream cut     -> cross-document context kept
#   --  packed 1024 (production shape)                -> [MASK] 2.43-2.45
#
# Verdict is read off eval_loss_mask ([MASK] positions only), against baselines
# measured on a random sample of the split (compounds packed / v3 padded agree):
#   unigram 2.476 / left-neighbour 1.932 / both-neighbours 1.380
#   < 1.93  -> the model uses context; cross-document attention is the cause
#   > 2.48  -> it fails here too; the post-LN architecture is the remaining suspect
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --job-name=<name> \
#     --export=ALL,TAG=<tag>,DATA=<dataset>,STEPS=<n>,WARM=<n>,EVERY=<n> \
#     workflows/bert-crossdoc-probe.sbatch
set -e
: "${TAG:?}" "${DATA:?}" "${STEPS:?}" "${WARM:?}" "${EVERY:?}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR before submitting}"
LR="${LR:-1e-4}"
TORCHRUN="${TORCHRUN:-torchrun}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/runs/crossdoc-${TAG}}"
export LEARNING_SOURCE_DIR
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO}/tmp/hf-cache-compounds}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs
echo "=== crossdoc probe [${TAG}] ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "data=${DATA}"
echo "seq 128 | LR=${LR} | warmup=${WARM} | steps=${STEPS} | global batch 8x80x4 = 2560"
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/compounds/bert_small.py \
  --model_path="${OUT}" --dataset_dir="${DATA}" --max_length=128 \
  --learning_rate=${LR} --warmup_steps=${WARM} --max_steps=${STEPS} \
  --log_interval=${EVERY} --save_steps=${STEPS}
echo "=== rc=$? @ $(date -Is) ==="
