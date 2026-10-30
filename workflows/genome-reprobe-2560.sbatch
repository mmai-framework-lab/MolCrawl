#!/bin/bash
#SBATCH --job-name=mc-gen-reprobe
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out

# A2 re-probe at the corrected effective global batch of 2,560.
#
# Purpose is wall-clock feasibility, not tuning: the gpu partition caps a job at
# 4 days, and at the shipped 8 x 320 split GPT-2 mammal_centered needs ~141 h.
# Only the micro-batch / grad_accum split varies here; the product -- and so the
# optimizer trajectory -- is held at 2,560 in every point.
#
# GPT-2 (nanoGPT): grad_accum is an all-rank total and train.py asserts it is
# divisible by world_size, so 2560/micro must stay a multiple of 4.
#   micro 128 -> 20, micro 160 -> 16, micro 320 -> 8   (256 -> 10 would fail)
# BERT (HF Trainer): grad_accum is per-device, so per_device * accum * 4 = 2560.
#   per_device 32 -> 20, 64 -> 10, 128 -> 5
#
# Timing only. SMOKE_MAX_STEPS shortens the run, which compresses the LR
# schedule; that does not matter for s/iter and these checkpoints are discarded.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
# Outputs go to the main checkout, not to a worktree that is meant to be deleted.
RUNS_ROOT="${RUNS_ROOT:-${PYROOT}/learning_source_genome_runs}"
echo "=== outputs under: ${RUNS_ROOT} ==="

: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
export GENOME_SUBSET=mammal_centered
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

PY="${PYROOT}/miniconda/bin/python"
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
OUT_BASE="${RUNS_ROOT}/reprobe-2560"
STEPS="${REPROBE_STEPS:-50}"

echo "=== node: $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
echo "=== steps per point: ${STEPS} ==="

############################ GPT-2 ############################
for pair in "128 20" "160 16" "320 8"; do
    set -- $pair
    MB=$1; GA=$2
    echo ""
    echo "########## GPT-2  micro=${MB}  grad_accum=${GA}  (product=$((MB * GA))) ##########"
    SMOKE_MAX_STEPS="${STEPS}" "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
        --out_dir="${OUT_BASE}/gpt2-mb${MB}" \
        --tensorboard_dir="${OUT_BASE}/gpt2-mb${MB}" \
        --batch_size="${MB}" \
        --gradient_accumulation_steps="${GA}" \
        --compile=False \
        --log_interval=10
    echo "---- rc=$? for gpt2 micro=${MB} ga=${GA} ----"
done

############################ BERT #############################
for pair in "32 20" "64 10" "128 5"; do
    set -- $pair
    MB=$1; GA=$2
    echo ""
    echo "########## BERT  per_device=${MB}  grad_accum=${GA}  (global=$((MB * GA * 4))) ##########"
    SMOKE_MAX_STEPS="${STEPS}" "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py \
        --model_path="${OUT_BASE}/bert-pd${MB}" \
        --batch_size="${MB}" \
        --gradient_accumulation_steps="${GA}" \
        --log_interval="${STEPS}" \
        --save_steps="${STEPS}"
    echo "---- rc=$? for bert per_device=${MB} ga=${GA} ----"
done

echo ""
echo "=== reprobe done @ $(date -Is) ==="
