#!/bin/bash
#SBATCH --job-name=mc-gen-rp1
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out

# A2 re-measurement, one micro-batch point per job.
#
# The first attempt ran all three points inside one job and produced a
# non-monotonic result (160 was 9x faster than both 128 and 320), which is not
# physically sensible. The suspected cause is page-cache carry-over between
# points: the first point warms the cache, the second rides it, the third
# evicts it. One point per job removes that coupling, and the cache counters
# below make the effect visible instead of inferred.
#
# Effective global batch is held at 2,560 (micro x accum, GPU-count-independent
# under nanoGPT's grad_accum //= world_size). 2560/micro must stay a multiple
# of world_size 4:  128 -> 20, 160 -> 16, 320 -> 8.
#
# Required: MB (micro-batch) and GA (grad accum), passed via --export.

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

echo "=== node=$(hostname)  micro=${MB}  grad_accum=${GA}  product=$((MB * GA))  steps=${STEPS} ==="
echo "=== page cache BEFORE: $(( $(cache_kb) / 1024 / 1024 )) GiB ==="
free -g | sed 's/^/    /'

SMOKE_MAX_STEPS="${STEPS}" "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
    --out_dir="${RUNS_ROOT}/rp1-mb${MB}" \
    --tensorboard_dir="${RUNS_ROOT}/rp1-mb${MB}" \
    --batch_size="${MB}" \
    --gradient_accumulation_steps="${GA}" \
    --compile=False \
    --log_interval=10
rc=$?

echo "=== page cache AFTER: $(( $(cache_kb) / 1024 / 1024 )) GiB ==="
free -g | sed 's/^/    /'
echo "---- rc=${rc} for micro=${MB} ga=${GA} ----"
