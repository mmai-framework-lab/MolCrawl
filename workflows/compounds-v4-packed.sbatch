#!/bin/bash
#SBATCH --job-name=mc-cmp-v4packed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/mc-cmp-v4packed-%j.out
#SBATCH --error=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/mc-cmp-v4packed-%j.out
# compounds training_ready v4: packed 1024-token blocks (supersedes v3 seq128+pad).
# CPU-only work; --gpus=1 only because `gpu` is the sole partition on this cluster.
set -e
REPO="/data1/rkp00024/matsubara/MolCrawl"
NEW="${REPO}/learning_source_20260805_compounds_packed"
export MOLCRAWL_REPO="${REPO}"
export ORGANIX13_DIR="${NEW}/compounds/organix13"
export LEARNING_SOURCE_DIR="${NEW}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/miniconda/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export V4_WORKERS=16
cd "${REPO}"
echo "=== compounds v4 packed build ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "input : ${ORGANIX13_DIR}/OrganiX13.parquet"
echo "output: ${ORGANIX13_DIR}/training_ready_hf_dataset_{gpt2,bert}"
"${REPO}/miniconda/bin/python" scripts/run_phase1_2_training_ready_v4_packed.py
echo "=== rc=$? @ $(date -Is) ==="
