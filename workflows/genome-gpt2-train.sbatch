#!/bin/bash
#SBATCH --job-name=mc-gen-gpt2
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=4-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# genome GPT-2 production, one subset per job.
#
# WHY THIS EXISTS BESIDE genome-train.sbatch. That one writes to
# <runs>/gpt2-small-<subset> with no tag, so a second series of the same subsets
# lands on top of the first and main.py's resume picks up from its checkpoints.
# The 21 three-epoch runs are the campaign's record and are not to be written
# over; RUN_TAG keeps a second series beside them.
#
# WHY IT STAGES TO NODE-LOCAL NVMe. Throughput here is set by shared-Lustre
# load, not by this config. The three-epoch runs measured 0.81 s/step staged;
# unstaged the same work runs several times slower and drifts with the time of
# day.
#
# Nine epochs is three times the three-epoch schedule: 122,685 to 163,902 steps
# across the 21 subsets, which at the measured 0.82 s/step is 28 to 37 hours.
# That fits one 4-day allocation with margin, so there is no chaining and no
# resume seam.
#
# Required: GENOME_SOURCE_ROOT, GENOME_SUBSET.
# Optional: LR, EPOCHS, EVAL_INTERVAL, MB, GA, RUN_TAG, RUNS_ROOT, STAGE.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
: "${GENOME_SUBSET:?set GENOME_SUBSET to the subset to train}"

LR="${LR:-0.0001}"
MB="${MB:-320}"
GA="${GA:-8}"
STAGE="${STAGE:-1}"
SUBSET="${GENOME_SUBSET}"
CONFIG=molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py

# The interpreter and the output root both resolve to the main checkout:
# ./miniconda does not exist in a worktree, and learning_source* is gitignored
# so every checkout has its own copy of it.
if [ -x "./miniconda/bin/torchrun" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
RUNS_ROOT="${RUNS_ROOT:-${PYROOT}/learning_source_genome_runs}"
OUT="${RUNS_ROOT}/gpt2-small-${SUBSET}${RUN_TAG:+-${RUN_TAG}}"
DONE="${OUT}/.run_complete"
LOCAL_ROOT=/tmp/genome-gpt2-${SLURM_JOB_ID:-0}

[ -f "${DONE}" ] && { echo "=== ${SUBSET}${RUN_TAG:+ (${RUN_TAG})} already complete ==="; exit 0; }

# A run that would resume into a tree trained on a different schedule. train.py
# resumes from whatever checkpoints it finds, so a 9-epoch job pointed at a
# 3-epoch tree would silently continue someone else's run on a schedule that
# ended long ago. Resume needs positive evidence, not the lack of a refusal.
if [ -d "${OUT}" ] && ls -d "${OUT}"/checkpoint-* >/dev/null 2>&1; then
    _prev=$("${PYROOT}/miniconda/bin/python" -c "
import json
try: print(json.load(open('${OUT}/run_manifest.json'))['schedule']['max_iters'])
except Exception: print('')" 2>/dev/null)
    if [ -z "${_prev}" ]; then
        echo "REFUSING: ${OUT} holds checkpoints with no run_manifest.json, so the" >&2
        echo "          schedule they were trained on cannot be established." >&2
        echo "          Set RUN_TAG to write beside them, or move them aside." >&2
        exit 1
    fi
fi

export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export GENOME_SUBSET="${SUBSET}"
export SUBSET_GPT2_LR="${LR}"
export SUBSET_GPT2_EPOCHS="${EPOCHS:-3}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subset=${SUBSET} LR=${LR} ==="
echo "=== epochs=${SUBSET_GPT2_EPOCHS} eval_interval=${EVAL_INTERVAL:-config} MB=${MB} GA=${GA} ==="
echo "=== torchrun: ${TORCHRUN} ==="
echo "=== outputs:  ${OUT} ==="
echo "=== commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
echo "=== global batch = ${MB} x ${GA} = $((MB * GA)) (nanoGPT divides grad_accum by world size) ==="

########## stage to node-local NVMe ##########
if [ "${STAGE}" = "1" ]; then
    DST="${LOCAL_ROOT}/genome_sequence"
    mkdir -p "${DST}/${SUBSET}" || { echo "cannot create ${DST}"; exit 1; }
    # The tokenizer is copied up front: the config builds it on demand when it
    # is missing, and on a fresh local root all four ranks would race to do so.
    cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/custom_tokenizer_bert_single_nuc" "${DST}/" 2>/dev/null
    t0=$(date +%s)
    cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${SUBSET}/training_ready_hf_dataset_gpt2" \
          "${DST}/${SUBSET}/" || { echo "staging failed -- falling back to Lustre"; STAGE=0; }
    [ "${STAGE}" = "1" ] && echo "=== staged in $(( $(date +%s) - t0 ))s ==="
fi
if [ "${STAGE}" = "1" ]; then export LEARNING_SOURCE_DIR="${LOCAL_ROOT}"
else export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"; fi

mkdir -p "${OUT}"
t1=$(date +%s)
"${TORCHRUN}" --standalone --nproc_per_node=4 \
    molcrawl/models/gpt2/train.py "${CONFIG}" \
    --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
    --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
    ${EVAL_INTERVAL:+--eval_interval="${EVAL_INTERVAL}"} \
    --compile=False --log_interval=100
rc=$?
elapsed=$(( $(date +%s) - t1 ))
rm -rf "${LOCAL_ROOT}"
[ ${rc} -eq 0 ] && touch "${DONE}"
echo "---- ${SUBSET}${RUN_TAG:+ (${RUN_TAG})} rc=${rc} elapsed=$((elapsed/3600))h$(((elapsed%3600)/60))m ----"
exit ${rc}
