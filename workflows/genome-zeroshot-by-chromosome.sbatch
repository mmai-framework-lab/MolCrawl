#!/bin/bash
#SBATCH --job-name=mc-gen-zs-bychrom
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Zero-shot AUROC per chromosome, and an overall over the three folds.
#
# The evaluator's own overall covers every row it scored, which is chr21, 22, X
# and Y; the fold analysis uses three of those. The two therefore describe
# different populations, and that was never written down. Both come out here,
# with the counts beside them.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: OUT.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT="${OUT:-${RESULTS_ROOT}/analyses/ep9/zeroshot-by-chromosome.tsv}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_zeroshot_by_chromosome.py \
    --scores-dir "bert-w1026:${RESULTS_ROOT}/clinvar-zeroshot-bert-w1026" \
    --scores-dir "gpt2-ep9:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-ep9" \
    --scores-dir "gpt2:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-w1024" \
    --out-tsv "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
