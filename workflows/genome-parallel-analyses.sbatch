#!/bin/bash
#SBATCH --job-name=mc-gen-analyses
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The two analyses the 21-run results cannot be read without, run beside the
# training rather than after it:
#
#   1. per-subset window loss and contig length distribution
#   2. per-subset degenerate baselines for GPT-2 and BERT
#
# 1.3703 has been carried as one number for the whole campaign. It came from one
# subset's valid split, so it says nothing about how predictable any other
# subset's valid split is, and the reported 0.2062 spread between subsets cannot
# be split into "data composition" and "this valid split is easier" without a
# baseline per subset.
#
# The GPU is not used; gpu is the only partition here.
#
# ROWS must match what the trainer actually evaluates on, not the size of the
# valid split. main.py takes the FIRST EVAL_SUBSET_ROWS rows (eval_subset_random
# is False), so a baseline over all 50,000 rows describes a range the run never
# scores against: on mammal_centered that is 1.3680 against the 1.3703 the run
# is judged by, a 0.0023 gap on a metric whose run-to-run noise is 0.0011.
#
# Required: SRC_ROOT, BERT_ROOT. Optional: SUBSETS, WINDOW, OUT_DIR, ROWS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC_ROOT:?set SRC_ROOT to the learning_source genome_sequence root}"
: "${BERT_ROOT:?set BERT_ROOT to the derived 1,026-token build root}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# window-loss.json is read back by the results table, so it is a record rather
# than scratch: it defaults beside the runs in the main checkout. RESULTS_ROOT
# overrides it; PYROOT already resolves the main checkout from a worktree.
RESULTS_ROOT="${RESULTS_ROOT:-${PYROOT}/learning_source_genome_runs/_results}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== outputs: ${OUT_DIR} ==="
rc=0

echo ""; echo "================ 1. window loss ================"
"${PY}" scripts/genome_window_loss.py \
    --src-root "${SRC_ROOT}" \
    --window "${WINDOW:-1024}" \
    ${SUBSETS:+--subsets "${SUBSETS}"} \
    --check-against-dataset \
    --out "${OUT_DIR}/window-loss.json" || rc=1

echo ""; echo "================ 2. degenerate baselines ================"
"${PY}" scripts/genome_degenerate_baselines.py \
    --src-root "${SRC_ROOT}" \
    --bert-root "${BERT_ROOT}" \
    ${SUBSETS:+--subsets "${SUBSETS}"} \
    --split valid \
    --rows "${ROWS:-10000}" \
    --out "${OUT_DIR}/degenerate-baselines.json" || rc=1

echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
