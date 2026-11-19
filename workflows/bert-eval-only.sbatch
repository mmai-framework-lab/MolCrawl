#!/bin/bash
#SBATCH --job-name=mc-bert-evalonly
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Score one existing checkpoint under the grid's evaluation. No training.
#
# The compounds batch-size grid fixes the masked positions; the existing 2,560 arms
# were measured with the positions redrawn at every evaluation. The two series do not
# join, so the existing checkpoints are measured again here under the grid's
# evaluation -- same GPU count, same scoring rows, same subset, fixed mask.
#
#   sbatch --export=ALL,LEARNING_SOURCE_DIR=<input root>,MODEL_OUTPUT_ROOT=<output root>,\
# CONFIG=<the arm's config>,CKPT=<checkpoint dir>,OUT=<a separate output dir>,\
# OUT_JSON=<where the metrics go>[,EVAL_BATCH=32] workflows/bert-eval-only.sbatch
#
# OUT must be outside the run being measured: core/output_guard.py refuses a
# model_path inside LEARNING_SOURCE_DIR, and models/bert/_eval_only.py refuses one
# inside the checkpoint's own run.
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the root outputs go under}"
: "${CONFIG:?set CONFIG to the config of the arm being measured}"
: "${CKPT:?set CKPT to the checkpoint directory to score}"
: "${OUT:?set OUT to an output directory outside the run being measured}"
: "${OUT_JSON:?set OUT_JSON to the file the metrics are written to}"
EVAL_BATCH="${EVAL_BATCH:-32}"
MASK_SEED="${MASK_SEED:-42}"

export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A git worktree holds only tracked files, so miniconda is not inside it.
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || {
  echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun" >&2; exit 1; }

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-compounds}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs

[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG} (relative to ${REPO})" >&2; exit 1; }
[ -d "${CKPT}" ] || { echo "no checkpoint directory at ${CKPT}" >&2; exit 1; }
if [ -e "${OUT_JSON}" ]; then
  # A measurement already written is the record something was reported from.
  echo "refusing to overwrite ${OUT_JSON}; move it aside or name another file" >&2
  exit 1
fi

echo "=== bert eval-only on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:       ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config:     ${CONFIG}"
echo "checkpoint: ${CKPT}"
echo "output:     ${OUT}"
echo "metrics:    ${OUT_JSON}"
echo "eval batch: ${EVAL_BATCH}   mask seed: ${MASK_SEED}   GPUs: 4"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CONFIG}" \
  --model_path="${OUT}" \
  --eval_only=True \
  --eval_only_checkpoint="${CKPT}" \
  --eval_only_out="${OUT_JSON}" \
  --fixed_eval_mask=True \
  --eval_mask_seed="${MASK_SEED}" \
  --eval_subset_random=True \
  --per_device_eval_batch_size="${EVAL_BATCH}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
