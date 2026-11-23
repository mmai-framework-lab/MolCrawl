#!/bin/bash
#SBATCH --job-name=mc-rna-tput-table
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Turn the rna micro-batch probe's logs into §4's chosen block count and §9's
# accumulation table.
#
#   RECORD=... sbatch --dependency=afterany:<last bert arm>:<last gpt2 arm> \
#     workflows/rna-microbatch-probe-table.sbatch
#
# Queued with the probe rather than run afterwards by hand: the chains take hours
# to days behind the queue, and a session that submitted them is not there to read
# the logs. afterany, because an arm that ran out of memory is an answer about that
# shape and must not keep the table from being written.
#
# Env:
#   RECORD   the probe's submit record TSV (names every arm and its job id)
#   LOGS     directory holding the arms' logs (default: this checkout's slurm-logs)
#   OUT      TSV to write (default: beside this job's log)
#
# Reads finished logs only. One GPU because the partition requires one.

set -u
: "${RECORD:?export RECORD to the submit record TSV of the probe}"
[ -f "${RECORD}" ] || { echo "no such record: ${RECORD}" >&2; exit 1; }
LOGS="${LOGS:-workflows/slurm-logs}"
OUT="${OUT:-workflows/slurm-logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-0}.tsv}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== read:     $(date -Is) ==="
echo "=== record:   ${RECORD} ==="
echo
PYTHONPATH="$PWD" "${PYTHON}" scripts/rna_microbatch_probe_table.py \
  --record "${RECORD}" --logs "${LOGS}" --out "${OUT}"
rc=$?
echo
echo "=== table rc=${rc} (non-zero: an arm has no rate and did not report FAILED) ==="
exit ${rc}
