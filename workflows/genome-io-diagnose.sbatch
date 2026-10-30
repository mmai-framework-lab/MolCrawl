#!/bin/bash
#SBATCH --job-name=mc-gen-io
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# #6 diagnosis: is the GPT-2 step time bound by random-read latency against
# Lustre, and does holding the shards in page cache remove it?
#
# What the numbers so far say:
#   - 5.83 s/iter at MFU ~3%, unchanged across micro-batch splits
#   - the iteration right after a 313 s checkpoint pause took 709 ms, i.e. the
#     loader gets far ahead whenever compute stops
#   - train split is 180 GB over 386 shards; the node has ~1680 GB of RAM, so
#     the whole split fits in page cache many times over
#   - yet a probe run only grew the cache from 7 to 35 GiB
#
# get_batch draws batch_size random row indices per micro-step, so a step
# touches thousands of scattered offsets across 180 GB. If each of those costs a
# Lustre round trip, the step is latency-bound rather than bandwidth-bound, and
# nothing about batch shape can help -- which matches A2.
#
# Two phases in one job, same node, so the only difference is cache residency:
#   phase 1  cold  -> expected to reproduce ~5.8 s/iter
#   phase 2  warm  -> shards read once sequentially, then the same measurement
#
# A large gap means residency is the lever and the fix is about keeping the
# working set resident. No gap means the cost is elsewhere and #6 needs a
# different hypothesis -- which is equally worth knowing before proposing work.

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

DS="${GENOME_SOURCE_ROOT}"/genome_sequence/mammal_centered/training_ready_hf_dataset_gpt2/train
STEPS=60
MB=128
GA=20

cache_gib() { awk '/^Cached:/{printf "%.0f", $2/1024/1024}' /proc/meminfo; }

run_probe() {
    local tag=$1
    echo ""
    echo "########## phase: ${tag}  (cache ${1:+}$(cache_gib) GiB) ##########"
    SMOKE_MAX_STEPS="${STEPS}" "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
        --out_dir="${RUNS_ROOT}/io-${tag}" \
        --tensorboard_dir="${RUNS_ROOT}/io-${tag}" \
        --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
        --compile=False --log_interval=10
    echo "---- ${tag} rc=$? cache=$(cache_gib) GiB ----"
}

echo "=== node=$(hostname) ==="
free -g | sed 's/^/    /'
echo "=== dataset: $(du -sh ${DS} 2>/dev/null | cut -f1) over $(ls ${DS}/*.arrow 2>/dev/null | wc -l) shards ==="

run_probe cold

echo ""
echo "########## warming page cache: reading all shards sequentially ##########"
t0=$(date +%s)
cat ${DS}/*.arrow > /dev/null 2>&1
t1=$(date +%s)
echo "=== warm read took $((t1 - t0)) s; cache now $(cache_gib) GiB ==="

run_probe warm

echo ""
echo "=== io diagnose done @ $(date -Is) ==="
