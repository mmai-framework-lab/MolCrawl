#!/bin/bash
#SBATCH --job-name=mc-prot-bmb
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# protein BERT batch-size grid, section 4: pick the 1-GPU micro-batch per size by
# MEASURING step time, not inheriting it. The 26 current protein BERT arms all run
# micro 8; on genome/compounds the shipped 8 x 80 shape was 5.7x slower than the best
# shape at the identical global batch, so this is measured here rather than assumed.
#
#   sbatch --export=ALL,SIZE=large,LEARNING_SOURCE_DIR=...,MODEL_OUTPUT_ROOT=... \
#          workflows/protein-bert-microbatch-trial.sbatch
#
# Candidates = the grid's allowed micro-batches only. The grid's smallest global batch
# is 256, and with 4 GPUs a micro-batch must satisfy micro x 4 | 256, so micro is one of
# {4,8,16,32,64}. 160 (used by the old throughput configs) is excluded: 160 x 4 = 640
# does not divide 256. The chosen micro is the same across all five batches of a size
# (directive section 4), so only the micro-batch time matters here, not any one batch.
#
# Each candidate runs with gradient_accumulation_steps=1 (pure micro-batch timing); the
# step time at a global batch G is then micro_s x G/(micro x 4). main.py stops a run
# whose effective batch does not match the declaration, so the trial declares what it
# actually runs (expected_global_batch = micro x 4) rather than being exempted.
#
# Measurement discipline (micro-benchmarks have misdiagnosed this workstream before):
#   - real config, real training entry point, not a synthetic loop
#   - one node, candidates run sequentially, nothing alongside
#   - train_runtime (which excludes model init and dataset open) is the only number divided
#   - log_interval and save_steps are past max_steps, so no eval and no checkpoint fire:
#     training throughput alone is timed
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the protein learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

# The interpreter and torchrun live in the MAIN checkout; a worktree has no ./miniconda.
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

CONFIG="molcrawl/tasks/pretrain/configs/protein_sequence/bert_${SIZE}.py"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

# The grid's allowed micro-batches, largest first. Same set for every size; OOM at a
# given size is reported, not assumed.
CANDIDATES="64 32 16 8 4"

TRIAL_ROOT="${MODEL_OUTPUT_ROOT}/microbatch-trial-protein-${SIZE}-${SLURM_JOB_ID}"
mkdir -p "${TRIAL_ROOT}"

echo "=== protein BERT micro-batch trial ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "code       : $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"
echo "size       : ${SIZE}"
echo "config     : ${CONFIG}"
echo "candidates : ${CANDIDATES}"
echo "output     : ${TRIAL_ROOT}"
echo

RESULTS="${TRIAL_ROOT}/results.tsv"
printf 'size\tmicro_batch\tsteps\tfit\tmicro_s\ttok_per_s\n' > "${RESULTS}"

for BS in ${CANDIDATES}; do
    # Roughly constant tokens per candidate, bounded so timing averages over enough steps.
    STEPS=$(( 1280 / BS )); [ "${STEPS}" -lt 10 ] && STEPS=10; [ "${STEPS}" -gt 40 ] && STEPS=40
    OUT="${TRIAL_ROOT}/mb${BS}"
    LOG="${TRIAL_ROOT}/mb${BS}.log"
    mkdir -p "${OUT}"

    echo "--- micro_batch=${BS} x ${STEPS} steps (accum 1, global ${BS}x4=$(( BS * 4 ))) @ $(date -Is)"
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py "${CONFIG}" \
        --model_path="${OUT}" \
        --batch_size="${BS}" \
        --gradient_accumulation_steps=1 \
        --expected_global_batch=$(( BS * 4 )) \
        --max_steps="${STEPS}" \
        --warmup_steps=1 \
        --log_interval=$(( STEPS + 1000 )) \
        --save_steps=$(( STEPS + 1000 )) \
        > "${LOG}" 2>&1
    RC=$?

    if [ "${RC}" -ne 0 ]; then
        if grep -qi "out of memory\|CUDA out of memory\|OutOfMemoryError" "${LOG}"; then
            echo "    does not fit (OOM)"
            printf '%s\t%s\t%s\tOOM\t\t\n' "${SIZE}" "${BS}" "${STEPS}" >> "${RESULTS}"
        else
            echo "    failed rc=${RC} (not memory); see ${LOG}"
            printf '%s\t%s\t%s\tFAIL(rc=%s)\t\t\n' "${SIZE}" "${BS}" "${STEPS}" "${RC}" >> "${RESULTS}"
            tail -20 "${LOG}"
        fi
        rm -rf "${OUT}"
        continue
    fi

    RUNTIME=$(grep -o "'train_runtime': [0-9.]*" "${LOG}" | tail -1 | sed "s/.*: //")
    if [ -z "${RUNTIME}" ]; then
        echo "    ran, but train_runtime not printed; see ${LOG}"
        printf '%s\t%s\t%s\tNO-TIMING\t\t\n' "${SIZE}" "${BS}" "${STEPS}" >> "${RESULTS}"
        rm -rf "${OUT}"
        continue
    fi

    read -r MICRO TOKS <<EOF
$(awk -v r="${RUNTIME}" -v s="${STEPS}" -v bs="${BS}" \
      'BEGIN { m = r / s; printf "%.4f %.0f", m, bs * 4 * 1024 / m }')
EOF
    echo "    fits. micro-step ${MICRO}s, ${TOKS} tok/s"
    printf '%s\t%s\t%s\tfits\t%s\t%s\n' "${SIZE}" "${BS}" "${STEPS}" "${MICRO}" "${TOKS}" >> "${RESULTS}"
    rm -rf "${OUT}"
done

echo
echo "=== results (${RESULTS}) ==="
cat "${RESULTS}"
echo
echo "=== fastest micro-batch (max tok/s) for ${SIZE} ==="
awk -F'\t' 'NR>1 && $4=="fits" { if ($6+0 > best+0) { best=$6; bs=$2; ms=$5 } } END { if (bs) printf "micro_batch=%s  micro_s=%s  tok_per_s=%s\n", bs, ms, best; else print "none fit" }' "${RESULTS}"
echo "=== done @ $(date -Is) ==="
