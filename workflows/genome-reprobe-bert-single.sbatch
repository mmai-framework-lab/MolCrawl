#!/bin/bash
#SBATCH --job-name=mc-gen-rpb
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# A2, BERT half: one micro-batch point per job.
#
# One point per job for the same reason as the GPT-2 half -- running the points
# together let page cache carry over between them and produced a reading that
# was not reproducible in isolation.
#
# HF Trainer treats gradient_accumulation_steps as per-device, so the global
# batch is per_device * accum * world_size and must come to 2,560:
#   32 -> 20, 64 -> 10, 128 -> 5   (all valid; HF imposes no divisibility rule)
#
# This half was blocked until torch_compile could be turned off: BERT compiled
# unconditionally and inductor crashed before the first step. Runs eager here,
# which the startup line confirms.
#
# Required: MB, GA via --export.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
# Outputs go to the main checkout, not to a worktree that is meant to be deleted.
RUNS_ROOT="${RUNS_ROOT:-${PYROOT}/learning_source_genome_runs}"
echo "=== outputs under: ${RUNS_ROOT} ==="

: "${MB:?set MB via --export}"
: "${GA:?set GA via --export}"
STEPS="${STEPS:-60}"

: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
export GENOME_SUBSET=mammal_centered
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

cache_kb() { awk '/^Cached:/{print $2}' /proc/meminfo; }

echo "=== node=$(hostname) per_device=${MB} accum=${GA} global=$((MB * GA * 4)) steps=${STEPS} ==="
echo "=== page cache BEFORE: $(( $(cache_kb) / 1024 / 1024 )) GiB ==="

SMOKE_MAX_STEPS="${STEPS}" "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py \
    --model_path="${RUNS_ROOT}/rpb-pd${MB}" \
    --batch_size="${MB}" \
    --gradient_accumulation_steps="${GA}" \
    --log_interval="${STEPS}" \
    --save_steps="${STEPS}"
rc=$?

echo "=== page cache AFTER: $(( $(cache_kb) / 1024 / 1024 )) GiB ==="
echo "---- rc=${rc} for per_device=${MB} ga=${GA} ----"
