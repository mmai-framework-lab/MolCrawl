#!/bin/bash
#SBATCH --job-name=mc-gen-params
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The non-embedding parameter count, per architecture, from the checkpoints.
#
# Counted rather than quoted. A compute figure needs the non-embedding count,
# and what counts as an embedding differs between the two paths: BERT's three
# input tables are all subtracted, while nanoGPT's wte is tied to the output
# head and is not.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RUNS_ROOT. Optional: OUT, BERT_RUN, GPT2_RUN.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== runs: ${RUNS_ROOT} ==="

"${PY}" scripts/genome_parameter_counts.py \
    --runs-root "${RUNS_ROOT}" \
    ${BERT_RUN:+--bert-run "${BERT_RUN}"} \
    ${GPT2_RUN:+--gpt2-run "${GPT2_RUN}"} \
    ${OUT:+--out "${OUT}"}
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
