#!/bin/bash
#SBATCH --job-name=mc-molnl-gpt2-ckpt-state
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the stopping counters out of nanoGPT checkpoints.
#
#   sbatch --export=ALL,CKPTS="<dir> <dir> ..." workflows/molnl-gpt2-ckpt-state.sbatch
#
# Space separated, not comma separated: --export splits a value on commas and the list
# would be cut short.
#
# --gpus=0 asks for no accelerator and does not keep the job off one -- the gpu partition
# is the default and bills by GPU (job 305167 was allocated gres/gpu=1 under this header).
# Nothing here uses a GPU: torch.load with mmap=True reads a few scalars.
set -eo pipefail
: "${CKPTS:?CKPTS=space separated checkpoint directories or training_state.bin paths}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
echo "=== gpt2 checkpoint state ${SLURM_JOB_ID:-local} @ $(date -Is) ==="
# shellcheck disable=SC2086
"${PYTHON}" scripts/molnl_gpt2_checkpoint_state.py ${CKPTS}
echo "=== rc=$? @ $(date -Is) ==="
