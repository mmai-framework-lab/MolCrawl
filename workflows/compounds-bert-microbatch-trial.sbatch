#!/bin/bash
#SBATCH --job-name=mc-cmp-bmb
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Find the largest per-device batch that fits at one BERT size, and what a step costs
# there. The compounds ladder ships 8 x 80; on genome the same shipped 8 x 80 measured
# 5.7x slower than 160 x 4 at the identical global batch, so the shape is worth
# measuring rather than inheriting.
#
#   sbatch --export=ALL,SIZE=large,LEARNING_SOURCE_DIR=...,MODEL_OUTPUT_ROOT=... \
#          workflows/compounds-bert-microbatch-trial.sbatch
#
# What is measured is one micro-batch, with gradient_accumulation_steps=1. The real
# step is 640/batch_size micro-batches, and at xl that is about three minutes, so a
# 50-step trial at the production shape would cost hours per candidate and tell us
# nothing the micro-batch does not. Memory is a per-micro-batch property, and the step
# time is the micro-batch time times the accumulation count.
#
# Candidates descend from a size-dependent ceiling. Every one of them divides 640, so
# batch_size * gradient_accumulation_steps * 4 GPUs = 2,560 sequences holds whichever
# is adopted -- the trial changes the shape, never the global batch.
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the packed compounds learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}" || exit 1
mkdir -p workflows/slurm-logs

TORCHRUN="${TORCHRUN:-${REPO}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun"; exit 1; }

CONFIG="molcrawl/tasks/pretrain/configs/compounds/bert_${SIZE}.py"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

# Ceilings by size. Starting every size at 320 spends most of the trial on candidates
# that cannot fit; these are one step above what the parameter count makes plausible.
case "${SIZE}" in
    small)  CANDIDATES="320 160 128 64 32 16 8" ;;
    medium) CANDIDATES="160 128 64 32 16 8" ;;
    large)  CANDIDATES="128 64 32 16 8" ;;
    xl)     CANDIDATES="64 32 16 8" ;;
    *) echo "unknown SIZE ${SIZE}" >&2; exit 1 ;;
esac

TRIAL_ROOT="${MODEL_OUTPUT_ROOT}/microbatch-trial-${SIZE}-${SLURM_JOB_ID}"
mkdir -p "${TRIAL_ROOT}"

echo "=== compounds BERT micro-batch trial ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code       : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "size       : ${SIZE}"
echo "config     : ${CONFIG}"
echo "candidates : ${CANDIDATES}"
echo "output     : ${TRIAL_ROOT}"
echo

RESULTS="${TRIAL_ROOT}/results.tsv"
printf 'size\tbatch_size\tgrad_accum\tsteps\tfit\tmicro_s\tstep_s\ttok_per_s\n' > "${RESULTS}"

for BS in ${CANDIDATES}; do
    GA=$(( 640 / BS ))                       # production accumulation for this shape
    # Roughly constant tokens per candidate, so a large batch is not charged for the
    # same step count as a small one. Bounded so the timing still averages over enough
    # steps to be readable.
    STEPS=$(( 1280 / BS )); [ "${STEPS}" -lt 10 ] && STEPS=10; [ "${STEPS}" -gt 40 ] && STEPS=40
    OUT="${TRIAL_ROOT}/bs${BS}"
    LOG="${TRIAL_ROOT}/bs${BS}.log"
    mkdir -p "${OUT}"

    echo "--- batch_size=${BS} (production grad_accum ${GA}) x ${STEPS} steps @ $(date -Is)"
    # log_interval and save_steps are past max_steps on purpose: no evaluation and no
    # checkpoint, so what is timed is the training step and nothing else.
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py "${CONFIG}" \
        --model_path="${OUT}" \
        --batch_size="${BS}" \
        --gradient_accumulation_steps=1 \
        --max_steps="${STEPS}" \
        --warmup_steps=1 \
        --log_interval=$(( STEPS + 1000 )) \
        --save_steps=$(( STEPS + 1000 )) \
        > "${LOG}" 2>&1
    RC=$?

    if [ "${RC}" -ne 0 ]; then
        if grep -qi "out of memory\|CUDA out of memory\|OutOfMemoryError" "${LOG}"; then
            echo "    does not fit (OOM)"
            printf '%s\t%s\t%s\t%s\tOOM\t\t\t\n' "${SIZE}" "${BS}" "${GA}" "${STEPS}" >> "${RESULTS}"
        else
            echo "    failed rc=${RC} for a reason other than memory; see ${LOG}"
            printf '%s\t%s\t%s\t%s\tFAIL(rc=%s)\t\t\t\n' "${SIZE}" "${BS}" "${GA}" "${STEPS}" "${RC}" >> "${RESULTS}"
            tail -20 "${LOG}"
        fi
        rm -rf "${OUT}"
        continue
    fi

    # train_runtime excludes model init and dataset load, which dominate a trial this
    # short, so it is the only number here worth dividing.
    RUNTIME=$(grep -o "'train_runtime': [0-9.]*" "${LOG}" | tail -1 | sed "s/.*: //")
    if [ -z "${RUNTIME}" ]; then
        echo "    ran, but train_runtime was not printed; see ${LOG}"
        printf '%s\t%s\t%s\t%s\tNO-TIMING\t\t\t\n' "${SIZE}" "${BS}" "${GA}" "${STEPS}" >> "${RESULTS}"
        rm -rf "${OUT}"
        continue
    fi

    read -r MICRO STEP_S TOKS <<EOF
$(awk -v r="${RUNTIME}" -v s="${STEPS}" -v ga="${GA}" -v bs="${BS}" \
      'BEGIN { m = r / s; printf "%.4f %.2f %.0f", m, m * ga, bs * 4 * 1024 / m }')
EOF
    echo "    fits. micro-step ${MICRO}s -> production step ${STEP_S}s at ${GA} accum, ${TOKS} tok/s"
    printf '%s\t%s\t%s\t%s\tfits\t%s\t%s\t%s\n' "${SIZE}" "${BS}" "${GA}" "${STEPS}" "${MICRO}" "${STEP_S}" "${TOKS}" >> "${RESULTS}"
    rm -rf "${OUT}"
done

echo
echo "=== results (${RESULTS}) ==="
cat "${RESULTS}"

# The trial above times training only. Evaluation is 150 points over the run against
# 15 saves, on a 10,000-row subset at per_device_eval_batch_size, and that count was
# inherited rather than measured. At xl it is the difference between 1% of the run and
# something worth reorganising, so one eval is timed at the shape that was adopted.
ADOPTED=$(awk -F'\t' 'NR>1 && $5=="fits" { print $2; exit }' "${RESULTS}")
if [ -n "${ADOPTED:-}" ]; then
    echo
    echo "=== one evaluation at batch_size=${ADOPTED} @ $(date -Is) ==="
    OUT="${TRIAL_ROOT}/eval-probe"
    LOG="${TRIAL_ROOT}/eval-probe.log"
    mkdir -p "${OUT}"
    # log_interval = max_steps puts exactly one evaluation at the last step.
    # save_on_improve would then write a checkpoint -- 17 GB at xl -- for a number
    # nothing reads, so it is off here.
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py "${CONFIG}" \
        --model_path="${OUT}" \
        --batch_size="${ADOPTED}" \
        --gradient_accumulation_steps=1 \
        --max_steps=2 \
        --warmup_steps=1 \
        --log_interval=2 \
        --save_steps=1000 \
        --save_on_improve=False \
        > "${LOG}" 2>&1
    EVAL_RC=$?
    EVAL_RUNTIME=$(grep -o "'eval_runtime': [0-9.]*" "${LOG}" | tail -1 | sed "s/.*: //")
    if [ "${EVAL_RC}" -eq 0 ] && [ -n "${EVAL_RUNTIME}" ]; then
        awk -v e="${EVAL_RUNTIME}" \
            'BEGIN { printf "one eval %.1fs -> 150 eval points = %.2f h wall = %.1f GPU-h\n", e, e*150/3600, e*150/3600*4 }'
    else
        echo "eval probe did not report a runtime (rc=${EVAL_RC}); see ${LOG}"
        tail -20 "${LOG}"
    fi
    rm -rf "${OUT}"
fi

echo
echo "=== largest that fits, and the 15,000-step cost it implies ==="
awk -F'\t' 'NR>1 && $5=="fits" { print; exit }' "${RESULTS}" |
    awk -F'\t' '{ printf "batch_size=%s gradient_accumulation_steps=%s  step %ss  15,000 steps = %.1f h wall = %.1f GPU-h\n", $2, $3, $7, $7*15000/3600, $7*15000/3600*4 }'
echo "=== done @ $(date -Is) ==="
