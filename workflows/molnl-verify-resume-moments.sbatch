#!/bin/bash
#SBATCH --job-name=mc-molnl-verify-moments
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Whether the mol_nl extensions restored the optimizer's moments at their resume.
#
#   sbatch --export=ALL,CKPTS="<dir> <dir> ..." workflows/molnl-verify-resume-moments.sbatch
#
# Space separated: --export splits a value on commas.
#
# --gpus=0 asks for no accelerator and does not keep the job off one (job 305167 was
# allocated gres/gpu=1 under this header). Nothing here uses a GPU.
set -eo pipefail
: "${CKPTS:?CKPTS=space separated checkpoint directories}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
echo "=== resume moment check ${SLURM_JOB_ID:-local} @ $(date -Is) ==="
# shellcheck disable=SC2086
"${PYTHON}" scripts/verify_resume_optimizer_state.py ${CKPTS}
echo "=== rc=$? @ $(date -Is) ==="
