#!/bin/bash
#SBATCH --job-name=mc-rna-ext-progress
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Where the rna 2026-11-18 extension stands, on a compute node.
#
#   BERT_ROOT=... GPT2_ROOT=... sbatch workflows/rna-extension-progress.sbatch
#   BERT_ROOT=... GPT2_ROOT=... sbatch --begin=now+6hours workflows/...
#
# --begin is the point of having this as a job rather than a shell in a session:
# Slurm holds it, costs nothing while pending, and writes the answer to a log that
# outlives whoever queued it. Four compounds runs once died on a Friday and were
# noticed on Monday because the only watcher was inside the session that started
# them.
#
# Env:
#   BERT_ROOT  the runs root the BERT configs resolve against
#   GPT2_ROOT  the runs root the GPT-2 configs resolve against (the gpt2-grid one)
#
# Reads finished files only. Nothing is written into a run directory, and the
# extension keeps running while this reads.
#
# One GPU because the partition requires one; this parses JSON and CSV.

set -u
: "${BERT_ROOT:?export BERT_ROOT to the runs root the BERT configs resolve against}"
: "${GPT2_ROOT:?export GPT2_ROOT to the runs root the GPT-2 configs resolve against}"
for d in "${BERT_ROOT}" "${GPT2_ROOT}"; do
  [ -d "${d}" ] || { echo "no such runs root: ${d}" >&2; exit 1; }
done

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

echo "=== checkout:  $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== measured:  $(date -Is) ==="
echo "=== BERT_ROOT: ${BERT_ROOT} ==="
echo "=== GPT2_ROOT: ${GPT2_ROOT} ==="
echo
PYTHONPATH="$PWD" "${PYTHON}" scripts/rna_extension_progress.py
rc=$?
echo
echo "=== progress rc=${rc} ==="
exit ${rc}
