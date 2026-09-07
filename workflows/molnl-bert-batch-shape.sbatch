#!/bin/bash
#SBATCH --job-name=mc-molnl-bshape
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Throughput of the batch decomposition, at a fixed effective global batch.
#
# genome BERT moved from per_device 8 x grad_accum 80 to 160 x 4 and got 5.7x for it.
# 32 of the 33 BERT configs in the plan are still 8 x 80, so the question is whether
# the rest are leaving the same throughput on the table.
#
# For mol_nl the answer was no: 8x80 / 32x20 / 64x10 came in at 35.57 / 35.16 / 35.15
# s/it, a 1.2% spread, with eval_loss_mask matching to 0.04 at step 200. genome's gain
# came from amortising I/O latency (MFU 3%, Lustre discarding the page cache); mol_nl's
# corpus is 2.4G and already compute-bound, so a bigger micro-batch buys nothing.
# Whether a modality gains is therefore a question about its I/O, and this script is
# how to answer it for the next one rather than assume either way.
#
# Submit from the repository root, once per decomposition:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,PD=8,GA=80 \
#     workflows/molnl-bert-batch-shape.sbatch
#
# PD x GA x world_size must equal what the config declares as expected_global_batch,
# or main.py refuses to start. That check is the point: it lets the decomposition be
# varied without letting the run being measured drift into a different model.
set -eo pipefail
: "${PD:?PD=per_device_train_batch_size}"
: "${GA:?GA=gradient_accumulation_steps}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
SIZE="${SIZE:-medium}"
STEPS="${STEPS:-200}"
EVERY="${EVERY:-25}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_${SIZE}.py"

if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || {
  echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun" >&2; exit 1; }

# Its own tree. This is a probe, not a ladder run, and must not be mistaken for one
# or land beside the adopted checkpoints.
RUNS_ROOT="${RUNS_ROOT:-${LEARNING_SOURCE_DIR}/molecule_nat_lang}"
OUT="${RUNS_ROOT}/bert-probe-batchshape/${SIZE}-${PD}x${GA}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs

echo "=== mol_nl bert-${SIZE} batch-shape probe ${PD}x${GA} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "per_device=${PD}  grad_accum=${GA}  world_size=4  -> effective $(( PD * GA * 4 ))"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "steps=${STEPS}  eval every ${EVERY}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CFG}" \
  --model_path="${OUT}" \
  --batch_size="${PD}" --per_device_eval_batch_size="${PD}" \
  --gradient_accumulation_steps="${GA}" \
  --max_steps="${STEPS}" --log_interval="${EVERY}" --save_steps="${STEPS}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
