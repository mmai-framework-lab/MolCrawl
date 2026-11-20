#!/bin/bash
#SBATCH --job-name=mc-rna-bsweep-premises
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Check the rna batch-size directive's own premises against the existing grid.
#
#   CURVES=... SUMMARY=... TRAIN_BIN=... sbatch workflows/rna-batch-sweep-premises.sbatch
#
# Env:
#   CURVES     the rt1 rna curve TSV (long form, one row per point)
#   SUMMARY    the rna BERT grid summary TSV (carries collapsed and turned_step)
#   TRAIN_BIN  the train split's .bin, for the block count
#   OUT        optional TSV to write the threshold crossings to
#
# Reads finished files only. One GPU because the partition requires one; this
# parses two TSVs and stats one file.

set -u
: "${CURVES:?export CURVES to the rt1 rna curve TSV}"
: "${SUMMARY:?export SUMMARY to the rna BERT grid summary TSV}"
: "${TRAIN_BIN:?export TRAIN_BIN to the train split bin file}"
for f in "${CURVES}" "${SUMMARY}" "${TRAIN_BIN}"; do
  [ -f "${f}" ] || { echo "no such file: ${f}" >&2; exit 1; }
done

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== measured: $(date -Is) ==="
echo
PYTHONPATH="$PWD" "${PYTHON}" scripts/rna_batch_sweep_premises.py \
  --curves "${CURVES}" --summary "${SUMMARY}" --train-bin "${TRAIN_BIN}" \
  ${OUT:+--out "${OUT}"}
rc=$?
echo
echo "=== premises rc=${rc} ==="
exit ${rc}
