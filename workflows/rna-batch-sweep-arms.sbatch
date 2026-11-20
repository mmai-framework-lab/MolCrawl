#!/bin/bash
#SBATCH --job-name=mc-rna-bsweep-arms
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Derive the rna batch-size grid's arms and check the directive's tables.
#
#   MICRO_SET=current OUT=... sbatch workflows/rna-batch-sweep-arms.sbatch
#
# Env:
#   MICRO_SET      which one-GPU block count table to build with (default current)
#   OUT        TSV to write the arm inventory to
#              (default: beside this job's log, which outlives the job;
#              a session scratchpad under /tmp is node-local and is lost)
#              (default: beside this job's log, which outlives the job;
#              a session scratchpad under /tmp is node-local and is lost)
#   GRACE_BLOCKS   with DETECTOR=1, the block position a healthy arm is past the
#                  collapse threshold by; each arm's grace window is this / its batch
#   DETECTOR       1 to write the collapse detector's values (a proposal)
#
# Arithmetic only; one GPU because the partition requires one.

set -u
OUT="${OUT:-workflows/slurm-logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-0}.tsv}"
MICRO_SET="${MICRO_SET:-current}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== derived:  $(date -Is) ==="
echo
PYTHONPATH="$PWD" "${PYTHON}" scripts/rna_batch_sweep_arms.py \
  --micro-batch-set "${MICRO_SET}" \
  ${DETECTOR:+--collapse-detection} \
  ${GRACE_BLOCKS:+--grace-blocks "${GRACE_BLOCKS}"} \
  ${OUT:+--out "${OUT}"}
rc=$?
echo
echo "=== arms rc=${rc} ==="
exit ${rc}
