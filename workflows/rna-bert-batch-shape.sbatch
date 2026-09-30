#!/bin/bash
# Does a given batch shape fit, and how fast is it, for one rna BERT size.
#
# The effective batch has to stay at 2,560 = per_device x grad_accum x world_size.
# large runs 8 x 80 on four GPUs. xl is 1,561.2M parameters against large's 605.7M,
# so 8 per device may not fit; if it does not, the per-device number comes down and
# the accumulation goes up by the same factor, and the product is unchanged.
#
# This is a probe, not a ladder run. It writes to its own tree so it cannot be
# mistaken for one or land beside the adopted checkpoints, and max_steps is short --
# the s/it printed here is a startup-inclusive figure, not a production rate.
#
# Usage:
#   LEARNING_SOURCE_DIR=<rna learning_source root> MODEL_OUTPUT_ROOT=<probe root> \
#     sbatch --export=ALL,SIZE=xl,PD=8,GA=80 workflows/rna-bert-batch-shape.sbatch
#SBATCH --job-name=rna-bert-bshape
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${PD:?PD=per_device_train_batch_size}"
: "${GA:?GA=gradient_accumulation_steps}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a probe root outside every input tree}"
[ -d "${LEARNING_SOURCE_DIR}/rna" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no rna/ under it" >&2; exit 1; }

SIZE="${SIZE:-xl}"
STEPS="${STEPS:-30}"
EVERY="${EVERY:-5}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="molcrawl/tasks/pretrain/configs/rna/bert_${SIZE}.py"
[ -f "${REPO}/${CFG}" ] || { echo "no config at ${CFG}" >&2; exit 1; }

if [ -x "${REPO}/miniconda/bin/torchrun" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

OUT="${MODEL_OUTPUT_ROOT}/rna/bert-probe-batchshape/${SIZE}-${PD}x${GA}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME="${MODEL_OUTPUT_ROOT}/_cache/hf" XDG_CACHE_HOME="${MODEL_OUTPUT_ROOT}/_cache/xdg"
export TMPDIR="${MODEL_OUTPUT_ROOT}/_cache/tmp"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" workflows/slurm-logs

# Ask the guard itself rather than restating its rule here, so the two cannot drift.
# Seconds, and before the allocation is put to work.
"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "model_path" || exit 1

echo "=== rna bert-${SIZE} batch-shape probe ${PD}x${GA} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "per_device=${PD}  grad_accum=${GA}  world_size=4  -> effective $(( PD * GA * 4 ))"
echo "outputs: ${OUT}"
echo "steps=${STEPS}  log every ${EVERY}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CFG}" \
  --model_path="${OUT}" \
  --batch_size="${PD}" --per_device_eval_batch_size="${PD}" \
  --gradient_accumulation_steps="${GA}" \
  --max_steps="${STEPS}" --warmup_steps=10 --log_interval="${EVERY}" \
  --save_steps="${STEPS}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/peak-ish /'
exit $rc
