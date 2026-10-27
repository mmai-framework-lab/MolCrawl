#!/bin/bash
#SBATCH --job-name=mc-gen-pos-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Which base the zero-shot single-position score actually reads.
#
# The evaluator builds its position list from flank alone and the adapter
# filters raw token-stream indices, while the tokeniser encodes with
# add_special_tokens=True. The linear probe adds that offset explicitly; the
# zero-shot path does not. This encodes a real row and reports what sits where.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: CSV, TOKENIZER. Optional: FLANK.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CSV:?set CSV to the scored clinvar table}"
: "${TOKENIZER:?set TOKENIZER to the tokenizer directory}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
"${PY}" scripts/check_clinvar_scored_position.py \
    --csv "${CSV}" --tokenizer "${TOKENIZER}" --flank "${FLANK:-512}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
