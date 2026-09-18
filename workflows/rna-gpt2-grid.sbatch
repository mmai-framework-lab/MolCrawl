#!/bin/bash
# One link of one point of the rna GPT-2 grid.
#
# The config names itself: GRID_CONFIG carries the file, and the config derives
# its own out_dir from MODEL_OUTPUT_ROOT and its own name. Nothing about the
# learning rate or the output location is passed on the command line, so the
# value a run trained at is the value committed in the config, and a log can be
# matched to a point without trusting how it was launched.
#
# init_from stays "resume" from the base config. Link 1 finds an empty out_dir
# and starts from scratch; later links continue. A link whose run has already
# reached max_iters exits at once, which is what makes over-provisioning links
# cheap.
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${GRID_CONFIG:?export GRID_CONFIG to the grid config to run}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside any input tree}"

# miniconda lives in the main checkout; a worktree holds only tracked files.
if [ -x "./miniconda/bin/torchrun" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HOME="${MODEL_OUTPUT_ROOT}/_cache/hf" XDG_CACHE_HOME="${MODEL_OUTPUT_ROOT}/_cache/xdg"
export TMPDIR="${MODEL_OUTPUT_ROOT}/_cache/tmp"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" workflows/slurm-logs

echo "=== ${SLURM_JOB_NAME:-link} @ $(date -Is) on ${SLURMD_NODENAME} ==="
# A link can start days after the one before it, from a checkout someone may
# have moved in between. Refuse to continue a run on different code rather than
# finding out from the loss curve.
if [ -n "${EXPECT_COMMIT:-}" ]; then
    ACTUAL=$(git rev-parse --short HEAD)
    [ "${ACTUAL}" = "${EXPECT_COMMIT}" ] || {
        echo "FATAL: repo is at ${ACTUAL}, expected ${EXPECT_COMMIT}" >&2; exit 1; }
    echo "commit guard: ${ACTUAL} == ${EXPECT_COMMIT}"
fi
echo "config: ${GRID_CONFIG}"
echo "branch: $(git rev-parse --abbrev-ref HEAD)  head: $(git rev-parse --short HEAD)"
"${PYROOT}"/miniconda/bin/python -c "import molcrawl; print('molcrawl from:', molcrawl.__file__)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Anchors to grep for:
#   "tokens per iteration will be: 2,621,440"  = 2560 x 1024, invariant to GPUs
#   "number of parameters: ..."                 the size this link actually built
rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${GRID_CONFIG}" \
  --compile=False --early_stopping=False || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit $rc
