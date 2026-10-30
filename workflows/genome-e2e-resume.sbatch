#!/bin/bash
#SBATCH --job-name=mc-gen-e2e-resume
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# End-to-end check that a resumed run continues rather than restarts.
#
# Chaining is no longer needed now that a subset fits in one job, but the
# launcher still resumes from out_dir, and that path is what turns a node
# failure into a continuation. It has never been exercised on real hardware.
#
# Checkpoints carry RNG state so the streams continue instead of restarting at
# seed + rank; without it the data order diverges, since get_batch draws its
# indices from np.random. Verified in isolation, but not through an actual
# interrupt-and-resume.
#
# The test does exactly that:
#   phase A  fresh run to STEPS, checkpointing at eval_interval
#   phase B  resume from the last checkpoint and run the same tail again
#
# If the streams really continue, phase B replays phase A's tail on the same
# data and the per-iteration losses match. If RNG were not restored the batches
# would differ and so would the losses -- which is what makes this a test rather
# than a smoke run. Comparing losses is the check; matching iter_num alone would
# pass even with the data order broken.

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

SUBSET="${GENOME_SUBSET:-mammal_centered}"
STEPS="${STEPS:-120}"
EVAL_EVERY="${EVAL_EVERY:-50}"   # checkpoints land here; last one is at 100
MB="${MB:-320}"
GA="${GA:-8}"

LOCAL_ROOT=/tmp/genome-e2e
DST="${LOCAL_ROOT}/genome_sequence/${SUBSET}"
OUT="${RUNS_ROOT}/e2e-resume-${SUBSET}"

export GENOME_SUBSET="${SUBSET}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

rm -rf "${OUT}"

echo "=== node=$(hostname) subset=${SUBSET} steps=${STEPS} eval_every=${EVAL_EVERY} ==="
mkdir -p "${DST}"
mkdir -p "${LOCAL_ROOT}/genome_sequence"
# The config builds the tokenizer on demand when it is missing, and staging
# repoints LEARNING_SOURCE_DIR, so on a fresh local root all four ranks race to
# create it. Copy the 16 KB tokenizer across so there is nothing to build.
for aux in custom_tokenizer_bert_single_nuc; do
    [ -d "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" ] && cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" "${LOCAL_ROOT}/genome_sequence/" || true
done
cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${SUBSET}/training_ready_hf_dataset_gpt2" "${DST}/" || {
    echo "stage failed"; exit 1; }
export LEARNING_SOURCE_DIR="${LOCAL_ROOT}"

run_phase() {
    local init=$1
    SMOKE_MAX_STEPS="${STEPS}" SMOKE_EVAL_INTERVAL="${EVAL_EVERY}" \
    "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py \
        molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
        --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
        --init_from="${init}" --compile=False \
        --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
        --log_interval=10
}

echo ""
echo "########## phase A: fresh run to ${STEPS} ##########"
run_phase scratch
echo "---- phase A rc=$? ----"

echo ""
echo "=== checkpoints written ==="
ls -d "${OUT}"/checkpoint-*/ 2>/dev/null | sed 's|.*checkpoint-||; s|/||' | sort -n | sed 's/^/    step /'
echo "=== per-rank RNG files in the last checkpoint ==="
last=$(ls -d "${OUT}"/checkpoint-*/ 2>/dev/null | sed 's|.*checkpoint-||; s|/||' | sort -n | tail -1)
ls "${OUT}/checkpoint-${last}"/rng_state_*.pth 2>/dev/null | sed 's|.*/|    |' || echo "    none -- resume cannot restore the streams"

echo ""
echo "########## phase B: resume from checkpoint-${last} ##########"
run_phase resume
echo "---- phase B rc=$? ----"

rm -rf "${LOCAL_ROOT}"
echo ""
echo "=== compare the tail: phase B should replay phase A iteration for iteration ==="
echo "=== e2e resume done @ $(date -Is) ==="
