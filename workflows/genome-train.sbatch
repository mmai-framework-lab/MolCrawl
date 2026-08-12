#!/bin/bash
#SBATCH --job-name=mc-gen-train
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=3-20:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Production run for one genome subset.
#
# Stages the split onto node-local NVMe before training. Read over Lustre the
# step is bound by per-read latency -- get_batch draws thousands of scattered
# row offsets per step across 180 GB -- and measured 5.826 s/iter at MFU 3.1%.
# From local NVMe the same work takes 0.708 s/iter at MFU 25.7%, an 8.2x
# difference, for a staging cost of about 111 s against a 10.7 h run. BERT
# gains less (1.423 -> 1.082 s/iter) because HF's DataLoader was already hiding
# some of the latency, but still gains.
#
# Holding the split in page cache was tried first and does not work: the pages
# are dropped as fast as they are read, with 1.6 TB allocated and 180 GB to
# hold, so this is not a capacity or cgroup problem.
#
# With staging, the largest subset needs 10.7 h (GPT-2) or 33.4 h (BERT) against
# the partition's 4-day limit, so a subset now fits in a single job and chaining
# is no longer required. The resume logic below is kept anyway: it costs nothing
# on a clean run and is what makes a node failure resumable rather than a
# restart from scratch. Links can still be pre-connected with
# --dependency=afterany via submit-genome-chain.sh if that is ever wanted.
#
# Requires GENOME_SUBSET. Optional: MODEL (gpt2|bert), MB / GA (micro-batch
# split; the product must stay 2560, and for gpt2 2560/MB must be a multiple of
# world_size 4), STAGE=0 to read from Lustre instead.

set -uo pipefail
# Submitted from the repo root by the submit-* wrappers; SLURM records that as
# SLURM_SUBMIT_DIR. Falling back to the current directory keeps a direct sbatch
# from the repo root working too.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${GENOME_SUBSET:?set GENOME_SUBSET via --export}"
MODEL="${MODEL:-gpt2}"
STAGE="${STAGE:-1}"

SRC_ROOT="${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
LOCAL_ROOT=/tmp/genome-local
SPLIT="training_ready_hf_dataset_${MODEL}"

export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

CFG="molcrawl/tasks/pretrain/configs/genome_sequence/${MODEL}_small_subset.py"
OUT="$PWD/learning_source_genome_runs/${MODEL}-small-${GENOME_SUBSET}"
DONE_MARKER="${OUT}/.run_complete"

if [ -f "${DONE_MARKER}" ]; then
    echo "=== ${GENOME_SUBSET}/${MODEL}: already complete ($(cat "${DONE_MARKER}")). Nothing to do. ==="
    exit 0
fi

# Refuse to train on a pre-v2 export. The two generations share a layout and
# subset names, so a wrong LEARNING_SOURCE_DIR would not fail on its own.
echo "=== verifying data source ==="
LEARNING_SOURCE_DIR="${SRC_ROOT}" ./miniconda/bin/python scripts/verify_genome_training_source.py \
    --subset "${GENOME_SUBSET}" --model "${MODEL}" --schema-only || {
    echo "data source verification failed; refusing to launch"; exit 1; }

if [ "${STAGE}" = "1" ]; then
    DST="${LOCAL_ROOT}/genome_sequence/${GENOME_SUBSET}"
    echo "=== staging ${SPLIT} to local NVMe ==="
    mkdir -p "${DST}"
    t0=$(date +%s)
    cp -r "${SRC_ROOT}/genome_sequence/${GENOME_SUBSET}/${SPLIT}" "${DST}/" || {
        echo "staging failed; refusing to fall back to Lustre silently"; exit 1; }
    echo "=== staged $(du -sh ${DST}/${SPLIT} 2>/dev/null | cut -f1) in $(( $(date +%s) - t0 )) s ==="
    export LEARNING_SOURCE_DIR="${LOCAL_ROOT}"
else
    echo "=== STAGE=0: reading over Lustre (expect ~8x slower for gpt2) ==="
    export LEARNING_SOURCE_DIR="${SRC_ROOT}"
fi
export GENOME_SUBSET

# Resume when a checkpoint carrying training state exists. train.py picks the
# highest step among out_dir/checkpoint-*/training_state.bin.
INIT_FROM=scratch
if compgen -G "${OUT}/checkpoint-*/training_state.bin" > /dev/null; then
    INIT_FROM=resume
    echo "=== resuming from $(ls -d ${OUT}/checkpoint-*/ | sed 's|.*checkpoint-||; s|/||' | sort -n | tail -1) ==="
fi

EXTRA=()
[ -n "${MB:-}" ] && EXTRA+=("--batch_size=${MB}")
[ -n "${GA:-}" ] && EXTRA+=("--gradient_accumulation_steps=${GA}")

echo "=== node=$(hostname) job=${SLURM_JOB_ID} model=${MODEL} subset=${GENOME_SUBSET} init=${INIT_FROM} ==="

if [ "${MODEL}" = "gpt2" ]; then
    ./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py "${CFG}" \
        --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
        --init_from="${INIT_FROM}" --compile=False "${EXTRA[@]}"
else
    # BERT resumes from out_dir on its own; torch.compile stays off, where it
    # measured 2.11x faster than on.
    ./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py "${CFG}" \
        --model_path="${OUT}" "${EXTRA[@]}"
fi
rc=$?

if [ "${rc}" -eq 0 ]; then
    date -Is > "${DONE_MARKER}"
    echo "=== ${GENOME_SUBSET}/${MODEL}: COMPLETE ==="
else
    echo "=== ${GENOME_SUBSET}/${MODEL}: exited rc=${rc}; rerun resumes from the last checkpoint ==="
fi

rm -rf "${LOCAL_ROOT}"
exit "${rc}"
