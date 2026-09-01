#!/bin/bash
#SBATCH --job-name=mc-gen-bert
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=4-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# genome BERT production, one subset per job.
#
# LR is 1e-4: three levels were probed at production schedule shape over 8,000
# steps (2026-08-24, jobs 35233 / 35234 / 35235) and 1e-4 was clearly best,
# while 1e-5 -- the value the pre-G2 note favoured -- stalled just under the
# degenerate line. The collapse detector stays off; every level crossed that
# line within 700 steps, so it carries no information here.
#
# WHY THIS STAGES TO NODE-LOCAL NVMe. Throughput is not set by this config; it
# is set by shared-Lustre load, which moves with the time of day:
#
#   cache-warm (no Lustre reads)      0.533 s/step
#   real run, three jobs at once      2.259
#   real run, alone, midday           2.817   <- 3.6 days for one subset
#
# The two concurrent runs on different nodes agreed to three decimals, which is
# what points at a shared bottleneck rather than the node. At 2.817 s/step a
# subset needs 3.6 days against a 4-day MaxTime, leaving 0.4 days of margin on a
# figure that drifts -- so an unstaged run is one busy afternoon away from being
# killed at 95% complete. Staging removes Lustre from the loop entirely.
#
# Fixing the micro-batch is what made this worth doing. At the old 8 x 80 the
# job was compute-bound (3.033 s/step even cache-warm), which is why staging
# measured only 1.3x for BERT before. At 160 x 4 compute costs 0.533 and I/O
# now dominates, so the picture matches genome GPT-2, where NVMe gave 8.2x.
#
# The run is resumable: checkpoints land every 1,000 steps and a re-submit picks
# up the newest one, so a job that dies mid-subset costs the staging plus the
# steps since the last checkpoint, not the run.
#
# Required: GENOME_SOURCE_ROOT, GENOME_SUBSET.
# Optional: LR, MB, GA, STAGE (set STAGE=0 to read straight from Lustre).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
: "${GENOME_SUBSET:?set GENOME_SUBSET to the subset to train}"

LR="${LR:-0.0001}"
MB="${MB:-160}"
GA="${GA:-4}"
STAGE="${STAGE:-1}"
SUBSET="${GENOME_SUBSET}"
CONFIG=molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py

# The output root is stated rather than inherited from wherever this was
# submitted. A worktree has its own gitignored learning_source_genome_runs, so
# $PWD silently sends a production run somewhere that is meant to be disposable.
# RUN_TAG keeps a probe from colliding with the production directory of the same
# subset -- the 512 and 1,024 builds of a subset share a name otherwise.
RUNS_ROOT="${RUNS_ROOT:-$PWD/learning_source_genome_runs}"
OUT="${RUNS_ROOT}/bert-small-${SUBSET}${RUN_TAG:+-${RUN_TAG}}"
echo "=== outputs: ${OUT} ==="
DONE="${OUT}/.run_complete"
LOCAL_ROOT=/tmp/genome-bert-${SLURM_JOB_ID:-0}

[ -f "${DONE}" ] && { echo "=== ${SUBSET} already complete -- nothing to do ==="; exit 0; }

# A run that would resume into a finished tree it was not meant to touch. The
# 512 and 1,024 builds of a subset share a directory name unless RUN_TAG
# separates them, and main.py resumes from whatever checkpoints it finds, so a
# mismatched window would silently continue someone else's run.
if [ -d "${OUT}" ] && ls -d "${OUT}"/checkpoint-* >/dev/null 2>&1; then
    _seen=$(ls "${OUT}"/checkpoint-*/run_manifest.json 2>/dev/null | head -1)
    _len=$("${PYROOT:-.}/miniconda/bin/python" -c "
import json,sys
try: print(json.load(open('${OUT}/run_manifest.json'))['batch']['seq_len'])
except Exception: print('')" 2>/dev/null)
    if [ -n "${_len}" ] && [ "${_len}" != "${SUBSET_BERT_MAX_LENGTH:-512}" ]; then
        echo "REFUSING: ${OUT} holds a run at seq_len ${_len}, this one is ${SUBSET_BERT_MAX_LENGTH:-512}." >&2
        echo "          Set RUN_TAG so the two do not share a directory." >&2
        exit 1
    fi
fi

export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export SUBSET_BERT_LR="${LR}"
export GENOME_SUBSET="${SUBSET}"

# The interpreter lives in the main checkout, not in a worktree, so a relative
# ./miniconda breaks whenever this is submitted from one. Resolve it through
# git's common dir rather than writing an absolute path into the repo.
if [ -x "./miniconda/bin/torchrun" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda -- set PYROOT"; exit 1; }

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subset=${SUBSET} LR=${LR} MB=${MB} GA=${GA} stage=${STAGE} ==="
echo "=== torchrun: ${TORCHRUN} ==="
echo "=== commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
echo "=== global batch = ${MB} x ${GA} x 4 GPU = $((MB * GA * 4)) ==="

########## stage to node-local NVMe ##########
if [ "${STAGE}" = "1" ]; then
    DST="${LOCAL_ROOT}/genome_sequence"
    mkdir -p "${DST}" || { echo "cannot create ${DST}"; exit 1; }
    # The tokenizer path follows LEARNING_SOURCE_DIR. Copying it up front is what
    # stops four ranks racing to build it on a fresh root -- that killed four
    # GPT-2 subsets before.
    cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/custom_tokenizer_bert_single_nuc" "${DST}/" 2>/dev/null
    # Only the BERT arrow tree. Copying the whole subset directory pulls in the
    # GPT-2 dataset and every intermediate as well -- 499G against the 230G that
    # is actually read.
    mkdir -p "${DST}/${SUBSET}"
    t0=$(date +%s)
    cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${SUBSET}/training_ready_hf_dataset_bert" \
          "${DST}/${SUBSET}/" || { echo "staging failed -- falling back to Lustre"; STAGE=0; }
    [ "${STAGE}" = "1" ] && echo "=== staged in $(( $(date +%s) - t0 ))s: $(du -sh ${DST}/${SUBSET} 2>/dev/null | cut -f1) ==="
fi
if [ "${STAGE}" = "1" ]; then
    export LEARNING_SOURCE_DIR="${LOCAL_ROOT}"
else
    export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
fi

########## resume ##########
# main.py finds the newest checkpoint under model_path by itself and resumes
# from it; there is no flag to pass. Reported here so the log says which way
# this job went.
mkdir -p "${OUT}"
LAST=$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "${LAST}" ] && echo "=== will resume from step ${LAST} ===" || echo "=== no checkpoint -- starting from scratch ==="

########## train ##########
t1=$(date +%s)
"${TORCHRUN}" --standalone --nproc_per_node=4 \
    molcrawl/models/bert/main.py "${CONFIG}" \
    --model_path="${OUT}" \
    --batch_size="${MB}" \
    --gradient_accumulation_steps="${GA}"
rc=$?
elapsed=$(( $(date +%s) - t1 ))

rm -rf "${LOCAL_ROOT}"

if [ ${rc} -eq 0 ]; then
    touch "${DONE}"
    echo "=== ${SUBSET} COMPLETE in ${elapsed}s ==="
else
    echo "=== ${SUBSET} rc=${rc} after ${elapsed}s -- re-submit to resume from the newest checkpoint ==="
fi
echo "=== done @ $(date -Is) ==="
