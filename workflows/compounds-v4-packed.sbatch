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
echo "output: ${ORGANIX13_DIR}/training_ready_hf_dataset_{gpt2,bert}${PREP_OUT_SUFFIX}"
echo "order : ${PACK_ORDER:-shuffled}"

# Refuse to write over an existing build. The finished ladders trained on what is
# already at these paths, and a rebuild that changes block contents cannot be undone.
# Set PREP_OUT_SUFFIX to write elsewhere, then rename deliberately.
for d in "${ORGANIX13_DIR}/training_ready_hf_dataset_gpt2${PREP_OUT_SUFFIX}" \
         "${ORGANIX13_DIR}/training_ready_hf_dataset_bert${PREP_OUT_SUFFIX}"; do
  if [ -e "${d}" ]; then
    echo "refusing to overwrite existing build: ${d}"
    echo "set PREP_OUT_SUFFIX to a fresh name, or move the existing one aside first"
    exit 1
  fi
done

"${REPO}/miniconda/bin/python" scripts/run_phase1_2_training_ready_v4_packed.py && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? could only ever print 0.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
