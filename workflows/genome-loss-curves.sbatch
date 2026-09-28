#!/bin/bash
#SBATCH --job-name=mc-gen-curves
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Build the genome loss-curve table, then draw every figure from it.
#
# The table comes first and the figures are drawn from nothing else, so a value
# on a chart can always be found in the TSV and the two cannot disagree.
#
# The five series are kept apart at every step. Their windows are 512, 1,024
# and 1,026 and their lengths are 3 and 9 epochs, so a curve from one says
# nothing about a curve from another; they share only the metric, which is
# pinned to eval_loss_mask for BERT rather than taken from whichever metric
# each run selected on.
#
# A baseline is drawn only where one was measured on that series' own build.
# The 1,026-window BERT runs and the GPT-2 runs have one from the final scorers
# (2026-09-16, valid all 50,000 rows, taken at the positions that paid loss).
# The 512-window series had none, so step 1 tallies it here -- a different
# measurement from the scorers', agreeing to about 0.002 where both exist, and
# the figure names which one it drew. The 1,024 probe still has none and its
# figure says so rather than borrowing the 1,026 build's number.
#
# No GPU work happens here. It runs on the gpu partition because that is where
# compute nodes are, and nothing in this repo runs on the login node.
#
# gpt2-lr-sweep is tabled but not drawn. Nine of its eleven runs predate the
# manifest and their directory names carry the arm, not the subset, so colour,
# baseline and the comparability note would all have to be guessed. The numbers
# are in the TSV; the figure would be the part that misleads.
#
# SRC_ROOT_1026 is the 1,026-token build, used only to count its train rows so
# the sweep's caption can state an epoch count. The sweep was launched without
# SUBSET_BERT_EPOCHS in the environment, so its manifest does not record one and
# the caption read "unknown" for a number the data answers.
#
# Required: RUNS_ROOT, OUT_DIR, SRC_ROOT_512. Optional: RESULTS_ROOT, SRC_ROOT_1026,
# SACCT_START, SACCT_END, SKIP_TALLY (set to 1 to reuse an earlier tally).
#
# The sacct window has to cover the whole campaign, not just one launch: a job
# number is only accepted when the window it ran in contains the time the run
# last wrote a checkpoint, which is what tells the 512-window series apart from
# the 1,026-window one when both produce a job of the same name for the same
# subset. SRC_ROOT_512 doubles as the row count that turns the untagged
# series' max_steps into an epoch count.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${OUT_DIR:?set OUT_DIR to the report directory the figures and TSV go to}"
: "${SRC_ROOT_512:?set SRC_ROOT_512 to the genome_sequence root of the 512 build}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${RUNS_ROOT}/_results}"
mkdir -p "${OUT_DIR}"
TSV="${OUT_DIR}/genome-loss-curves.tsv"
META="${OUT_DIR}/genome-loss-curves-meta.json"
TALLY="${RESULTS_ROOT}/analyses/degenerate-baselines-512-full.json"
TALLY_1024="${RESULTS_ROOT}/analyses/degenerate-baselines-w1024-ep9.json"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== runs:    ${RUNS_ROOT} ==="
echo "=== outputs: ${OUT_DIR} ==="

echo ""; echo "########## 1. 窓 512 の基準線（BERT のみ） ##########"
# gpt2 is skipped: its build was already measured in the same pass as its loss
# on 2026-09-16, and the tally reads every token of every row, so including it
# would double the wall time to reproduce a number that already exists.
if [ "${SKIP_TALLY:-0}" = "1" ] && [ -f "${TALLY}" ]; then
    echo "  SKIP_TALLY=1 なので既存の ${TALLY} を使う"
else
    "${PY}" scripts/genome_degenerate_baselines.py \
        --src-root "${SRC_ROOT_512}" \
        --bert-root "${SRC_ROOT_512}" \
        --models bert \
        --split valid \
        --out "${TALLY}" || { echo "  FAILED: tally"; exit 1; }
fi

echo ""; echo "########## 2. TSV ##########"
"${PY}" scripts/genome_loss_curves.py \
    --runs-root "${RUNS_ROOT}" \
    --log-dir "$PWD/workflows/slurm-logs" \
    --sacct-start "${SACCT_START:-2026-08-01}" \
    --sacct-end "${SACCT_END:-2026-09-30}" \
    --train-root "bert-base=${SRC_ROOT_512}" "bert-sat9=${SRC_ROOT_512}" \
        ${SRC_ROOT_1026:+"bert-lr-sweep=${SRC_ROOT_1026}"} \
    --out "${TSV}" \
    --meta-out "${META}" || { echo "  FAILED: TSV"; exit 1; }

echo ""; echo "########## 3. 図 ##########"
# series=directory. A series absent from this list gets no baseline line, which
# is the honest state for the builds that have not been measured.
"${PY}" scripts/plot_genome_loss_curves.py \
    --tsv "${TSV}" \
    --meta "${META}" \
    --baselines \
        "bert-w1026=${RESULTS_ROOT}/final-scores-bert-w1026" \
        "bert-lr-sweep=${RESULTS_ROOT}/final-scores-bert-w1026" \
        "gpt2=${RESULTS_ROOT}/final-scores-gpt2" \
    --baselines-tally \
        "bert-base=${TALLY}:bert" \
        "bert-sat9=${TALLY}:bert" \
        "bert-w1024-ep9=${TALLY_1024}:bert" \
    --skip-series ${SKIP_SERIES:-gpt2-lr-sweep} \
    --out-dir "${OUT_DIR}" || { echo "  FAILED: figures"; exit 1; }

echo ""; echo "---- rc=0 ----"
ls -la "${OUT_DIR}" | grep -E "genome-loss" | sed 's/^/  /'
echo "=== done @ $(date -Is) ==="
