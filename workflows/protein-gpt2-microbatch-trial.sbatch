#!/bin/bash
#SBATCH --job-name=mc-prot-gmb
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# protein GPT-2 batch-size grid, section 4: pick the 1-GPU micro-batch per size by
# MEASURING iteration time, the nanoGPT counterpart of the BERT micro-batch trial.
#
#   sbatch --export=ALL,SIZE=small,LEARNING_SOURCE_DIR=...,MODEL_OUTPUT_ROOT=... \
#          workflows/protein-gpt2-microbatch-trial.sbatch
#
# Candidates = the grid's allowed micro-batches {4,8,16,32,64}: micro x 4 GPUs must
# divide the smallest global batch 256. The chosen micro is the same across all five
# batches of a size (directive section 4).
#
# nanoGPT world-size handling (molcrawl/models/gpt2/train.py): the DECLARED
# gradient_accumulation_steps is divided by world_size (train.py line 365), and the
# expected_global_batch guard then checks (GA/world) x world x batch_size = GA x
# batch_size. To time ONE micro-forward per GPU per iter (pure micro-batch timing) we
# declare gradient_accumulation_steps = world_size (= 4), so GA/4 = 1 per rank, and set
# expected_global_batch = micro x 4 to satisfy the guard. The global batch during the
# probe is micro x 4; the step time at a grid global batch G is iter_s x G/(micro x 4).
#
# Measurement discipline: real config + real train.py, one node, candidates sequential,
# eval all but disabled (eval_interval past max_iters; init_from=scratch; the one
# start-of-run eval is held tiny with eval_sequences=64, since the config's 3,200
# forbids setting eval_iters directly), no checkpoint
# (always_save_checkpoint False), and the first iterations are dropped so dataset open
# and the first-step overhead are excluded from the timing.
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the protein learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda" >&2; exit 1; }
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}" || exit 1
mkdir -p workflows/slurm-logs

CONFIG="molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_${SIZE}.py"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

CANDIDATES="64 32 16 8 4"
ITERS=25          # iterations per candidate; the first WARM are dropped from timing
WARM=10

TRIAL_ROOT="${MODEL_OUTPUT_ROOT}/microbatch-trial-protein-gpt2-${SIZE}-${SLURM_JOB_ID}"
mkdir -p "${TRIAL_ROOT}"

echo "=== protein GPT-2 micro-batch trial ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "code       : $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"
echo "size       : ${SIZE}"
echo "config     : ${CONFIG}"
echo "candidates : ${CANDIDATES}"
echo "output     : ${TRIAL_ROOT}"
echo

RESULTS="${TRIAL_ROOT}/results.tsv"
printf 'size\tmicro_batch\titers\twarm\tfit\titer_s\ttok_per_s\n' > "${RESULTS}"

for BS in ${CANDIDATES}; do
    OUT="${TRIAL_ROOT}/mb${BS}"
    LOG="${TRIAL_ROOT}/mb${BS}.log"
    mkdir -p "${OUT}"

    echo "--- micro_batch=${BS} x ${ITERS} iters (accum 4/world=1 per rank, global ${BS}x4=$(( BS * 4 ))) @ $(date -Is)"
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/gpt2/train.py "${CONFIG}" \
        --out_dir="${OUT}" \
        --batch_size="${BS}" \
        --gradient_accumulation_steps=4 \
        --expected_global_batch=$(( BS * 4 )) \
        --max_iters="${ITERS}" \
        --init_from=scratch \
        --eval_interval=$(( ITERS + 1000 )) \
        --eval_sequences=64 \
        --log_interval=1 \
        --always_save_checkpoint=False \
        > "${LOG}" 2>&1
    RC=$?

    if [ "${RC}" -ne 0 ]; then
        if grep -qi "out of memory\|CUDA out of memory\|OutOfMemoryError" "${LOG}"; then
            echo "    does not fit (OOM)"
            printf '%s\t%s\t%s\t%s\tOOM\t\t\n' "${SIZE}" "${BS}" "${ITERS}" "${WARM}" >> "${RESULTS}"
        else
            echo "    failed rc=${RC} (not memory); see ${LOG}"
            printf '%s\t%s\t%s\t%s\tFAIL(rc=%s)\t\t\n' "${SIZE}" "${BS}" "${ITERS}" "${WARM}" "${RC}" >> "${RESULTS}"
            tail -20 "${LOG}"
        fi
        rm -rf "${OUT}"
        continue
    fi

    # Median of the per-iter "time Xms" after the warm-up iters (iter index >= WARM).
    # dt is one full optimizer step = one micro-forward per GPU here. POSIX awk: the
    # "iter N: loss L, time Xms, ..." line splits so $2 is "N:" and $6 is "Xms".
    read -r ITER_S TOKS <<EOF
$(grep -E 'iter [0-9]+: loss [0-9.]+, time [0-9.]+ms' "${LOG}" | \
  awk -v warm="${WARM}" -v bs="${BS}" '
        { it=$2; sub(/:/,"",it); tm=$6; sub(/ms/,"",tm);
          if (it+0 >= warm) { v[n++]=tm+0 } }
        END {
            if (n == 0) { print "NA NA"; exit }
            for (i=0;i<n;i++) for (j=i+1;j<n;j++) if (v[j]<v[i]) { t=v[i]; v[i]=v[j]; v[j]=t }
            med = (n%2) ? v[(n-1)/2] : (v[n/2-1]+v[n/2])/2
            s = med/1000.0
            printf "%.4f %.0f", s, bs*4*1024/s
        }')
EOF
    if [ "${ITER_S}" = "NA" ]; then
        echo "    ran, but no per-iter timing past warm-up; see ${LOG}"
        printf '%s\t%s\t%s\t%s\tNO-TIMING\t\t\n' "${SIZE}" "${BS}" "${ITERS}" "${WARM}" >> "${RESULTS}"
        rm -rf "${OUT}"
        continue
    fi
    echo "    fits. iter ${ITER_S}s (median past warm ${WARM}), ${TOKS} tok/s"
    printf '%s\t%s\t%s\t%s\tfits\t%s\t%s\n' "${SIZE}" "${BS}" "${ITERS}" "${WARM}" "${ITER_S}" "${TOKS}" >> "${RESULTS}"
    rm -rf "${OUT}"
done

echo
echo "=== results (${RESULTS}) ==="
cat "${RESULTS}"
echo
echo "=== fastest micro-batch (max tok/s) for gpt2 ${SIZE} ==="
awk -F'\t' 'NR>1 && $5=="fits" { if ($7+0 > best+0) { best=$7; bs=$2; is=$6 } } END { if (bs) printf "micro_batch=%s  iter_s=%s  tok_per_s=%s\n", bs, is, best; else print "none fit" }' "${RESULTS}"
echo "=== done @ $(date -Is) ==="
