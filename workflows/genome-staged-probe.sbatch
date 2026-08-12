#!/bin/bash
#SBATCH --job-name=mc-gen-staged
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Probe a split staged on node-local NVMe instead of read over Lustre.
#
# Established for GPT-2: 5.826 -> 0.708 s/iter, MFU 3.1% -> 25.7%,
# an 8.2x speedup, with staging costing 111 s against a 10.7 h run. The split is
# read as thousands of scattered row offsets per step, so the cost was per-read
# latency; local NVMe removes the network round trip. Holding the split in page
# cache was tried first and does not work -- the pages are dropped as fast as
# they are read.
#
# MODEL=gpt2|bert selects the trainer. Effective global batch is 2,560 either
# way, but the two frameworks reach it differently: nanoGPT divides grad_accum
# by world_size (so micro x accum = 2560, accum divisible by 4), while HF
# Trainer treats it as per-device (so per_device x accum x 4 = 2560).

set -uo pipefail
# Submitted from the repo root by the submit-* wrappers; SLURM records that as
# SLURM_SUBMIT_DIR. Falling back to the current directory keeps a direct sbatch
# from the repo root working too.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

MODEL="${MODEL:-gpt2}"
STEPS="${STEPS:-60}"
case "${MODEL}" in
    gpt2) MB="${MB:-128}"; GA="${GA:-20}" ;;
    bert) MB="${MB:-32}";  GA="${GA:-20}" ;;
    *) echo "MODEL must be gpt2 or bert"; exit 1 ;;
esac

SRC="${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}/genome_sequence/mammal_centered"
LOCAL_ROOT=/tmp/genome-local
DST="${LOCAL_ROOT}/genome_sequence/mammal_centered"
SPLIT="training_ready_hf_dataset_${MODEL}"

export GENOME_SUBSET=mammal_centered
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) model=${MODEL} micro=${MB} accum=${GA} steps=${STEPS} ==="

echo "########## staging ${SPLIT} to local NVMe ##########"
mkdir -p "${DST}"
t0=$(date +%s)
cp -r "${SRC}/${SPLIT}" "${DST}/" || { echo "stage failed"; exit 1; }
t1=$(date +%s)
echo "=== staged $(du -sh ${DST}/${SPLIT} 2>/dev/null | cut -f1) in $((t1 - t0)) s ==="

echo ""
echo "########## probe from local NVMe ##########"
if [ "${MODEL}" = "gpt2" ]; then
    LEARNING_SOURCE_DIR="${LOCAL_ROOT}" SMOKE_MAX_STEPS="${STEPS}" \
    ./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
        --out_dir="$PWD/learning_source_genome_runs/staged-${MODEL}" \
        --tensorboard_dir="$PWD/learning_source_genome_runs/staged-${MODEL}" \
        --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
        --compile=False --log_interval=10
else
    LEARNING_SOURCE_DIR="${LOCAL_ROOT}" SMOKE_MAX_STEPS="${STEPS}" \
    ./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py \
        --model_path="$PWD/learning_source_genome_runs/staged-${MODEL}" \
        --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
        --log_interval="${STEPS}" --save_steps="${STEPS}"
fi
echo "---- ${MODEL} rc=$? ----"

rm -rf "${LOCAL_ROOT}"
echo "=== done @ $(date -Is) ==="
