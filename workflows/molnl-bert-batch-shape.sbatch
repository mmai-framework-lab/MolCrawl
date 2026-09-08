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
#
# NOT the way to measure a production s/it. STEPS is passed as max_steps, so a 200
# step probe runs a 200 step schedule -- and warmup_steps is a constant 200, so the
# whole probe sits inside warmup while production leaves it at step 200 and decays
# from there. That is harmless for comparing decompositions against each other, which
# is all this is for: the three runs are wrong in the same way and the ratio holds.
# For an absolute number, run the production schedule and let the time limit stop it:
#
#   LEARNING_SOURCE_DIR=<dir> RUNS_ROOT=<probe root> sbatch --time=<short> \
#     --export=ALL,SIZE=large,STEPS=12000,SEGMENT=1,EVERY=100 \
#     workflows/molnl-bert-ladder.sbatch
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
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# Its own tree. This is a probe, not a ladder run, and must not be mistaken for one
# or land beside the adopted checkpoints.
# Inputs and outputs are separate trees. core/output_guard.py refuses a model_path
# that lands inside LEARNING_SOURCE_DIR, and mol_nl used to point both at the same
# place: LEARNING_SOURCE_DIR named learning_source_20260803_molnl_train, which holds
# the outputs and two symlinks to the corpus. So the default here was a path the
# guard is guaranteed to reject, and the failure came two minutes into a 4-GPU job.
#
# LEARNING_SOURCE_DIR now names the corpus itself (learning_source), which is what
# the symlinks pointed at, so the data read is the same bytes. MOLNL_ARTIFACT_ROOT
# names where mol_nl's own artifacts already live, and nothing moves.
ARTIFACT_ROOT="${MOLNL_ARTIFACT_ROOT:-${PYROOT}/learning_source_20260803_molnl_train}"
RUNS_ROOT="${RUNS_ROOT:-${ARTIFACT_ROOT}/molecule_nat_lang}"
OUT="${RUNS_ROOT}/bert-probe-batchshape/${SIZE}-${PD}x${GA}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs

# Ask the guard itself rather than re-deriving its rule here, so the two cannot drift
# and so nanoGPT -- which does not call it anywhere -- is covered too. Seconds, and
# before the allocation is put to work.
"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "model_path" || exit 1

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
