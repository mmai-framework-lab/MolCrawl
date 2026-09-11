#!/bin/bash
#SBATCH --job-name=mc-gen-clinvar-probe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Linear probe on frozen representations for a genome BERT campaign, beside the
# model-free control, on the chromosome-rotation folds (test chr21 / chr22 /
# chrX, chrY always on the training side).
#
# Each run is probed at the checkpoint it adopted -- best_model_checkpoint from
# its own trainer_state, not the newest one; the two differ in all 21 runs.
# Regularisation is one C for every run and every fold, recorded with the result.
#
# Per-variant held-out scores are written for the control and every run. The
# fold AUROCs alone cannot be compared variant for variant afterwards, and the
# comparisons asked for -- probe against probe, probe against the control -- are
# all paired.
#
# Required: RUNS_ROOT, CLINVAR_CSV, TOKENIZER.
# Optional: RUN_GLOB, ARCH, RUN_TAG, C, SEED, BATCH, OUT_DIR.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${CLINVAR_CSV:?set CLINVAR_CSV to the 1,024-base ClinVar table}"
: "${TOKENIZER:?set TOKENIZER to the genome single-nucleotide tokenizer}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RUN_GLOB="${RUN_GLOB:-bert-small-*-w1026}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/_results/clinvar-probe-bert-w1026}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== runs:    ${RUNS_ROOT}/${RUN_GLOB} ==="
echo "=== outputs: ${OUT_DIR} ==="
echo "=== C ${C:-1.0}  seed ${SEED:-1026}  arch ${ARCH:-bert} ==="

"${PY}" scripts/clinvar_linear_probe.py \
    --clinvar-csv "${CLINVAR_CSV}" \
    --runs-root "${RUNS_ROOT}" \
    --run-glob "${RUN_GLOB}" \
    --arch "${ARCH:-bert}" \
    --run-tag "${RUN_TAG:-w1026}" \
    --tokenizer "${TOKENIZER}" \
    --C "${C:-1.0}" \
    --seed "${SEED:-1026}" \
    --batch-size "${BATCH:-8}" \
    --pred-dir "${OUT_DIR}" \
    --out "${OUT_DIR}/probe.json"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
