#!/bin/bash
# Write the rna loss curves as one TSV, one row per evaluation point.
#
# No GPU: it reads logging_*.csv and trainer_state.json and writes a table. It
# still goes through Slurm, because several hundred megabytes of run logs is not
# something to read on the login node.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=rna-curve-table
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -eo pipefail
: "${GPT2_RUNS_ROOT:?export GPT2_RUNS_ROOT to the directory holding the rna-gpt2-* run directories}"
: "${BERT_RUNS_ROOT:?export BERT_RUNS_ROOT to the directory holding the rna-<size>-lr* run directories}"
: "${CURVE_OUT:?export CURVE_OUT to the TSV to write, outside every input tree}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
mkdir -p workflows/slurm-logs "$(dirname "${CURVE_OUT}")"
echo "=== rna-curve-table @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
_args=()
for d in ${CURVE_LOG_DIRS:-}; do _args+=(--log-dir "$d"); done
"${PY}" scripts/rna_curve_table.py \
  --gpt2-root "${GPT2_RUNS_ROOT}" \
  --bert-root "${BERT_RUNS_ROOT}" \
  --out "${CURVE_OUT}" "${_args[@]}"
# The downstream table takes its loss column out of the curve TSV just written,
# so the same number cannot appear differently in the two files.
if [ -n "${DOWNSTREAM_OUT:-}" ]; then
    : "${EVAL_ROOT:?export EVAL_ROOT to the eval/ root when DOWNSTREAM_OUT is set}"
    "${PY}" scripts/rna_downstream_table.py \
      --eval-root "${EVAL_ROOT}" \
      --curve-tsv "${CURVE_OUT}" \
      --out "${DOWNSTREAM_OUT}"
fi
echo "=== done @ $(date -Is) ==="
