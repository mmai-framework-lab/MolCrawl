#!/bin/bash
#SBATCH --job-name=mc-gen-bert-isolate
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out

# Controlled test: is the BERT dynamo/inductor failure caused by PR #102, or by
# micro-batch shape?
#
# Both points run on the SAME commit (current main, which INCLUDES #102) and
# differ only in batch_size. The effective global batch is held at 2,560:
#   batch 8  x accum 80 x 4 GPU = 2560   <- the shipped config; succeeded on 08-03
#   batch 32 x accum 20 x 4 GPU = 2560   <- failed in the A2 re-probe
#
# If 8 passes and 32 fails with #102 present in both, then #102 is not what
# breaks startup and the trigger is the shape. Five steps is enough: the failure
# occurs while compiling the first training step, before any step completes.

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

TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
OUT_BASE="${RUNS_ROOT}/bert-compile-isolate"
STEPS=5

echo "=== commit under test: $(git rev-parse --short HEAD) ==="
echo "=== #102 (8a12f63) present: $(git merge-base --is-ancestor 8a12f63 HEAD && echo YES || echo NO) ==="
echo "=== node: $(hostname) ==="

for pair in "8 80" "32 20"; do
    set -- $pair
    MB=$1; GA=$2
    echo ""
    echo "########## BERT per_device=${MB} accum=${GA} (global=$((MB * GA * 4))) ##########"
    SMOKE_MAX_STEPS="${STEPS}" "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py \
        --model_path="${OUT_BASE}/pd${MB}" \
        --batch_size="${MB}" \
        --gradient_accumulation_steps="${GA}" \
        --log_interval="${STEPS}" \
        --save_steps="${STEPS}"
    echo "---- rc=$? for per_device=${MB} ----"
done

echo ""
echo "=== isolate done @ $(date -Is) ==="
