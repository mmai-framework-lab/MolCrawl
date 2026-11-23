#!/bin/bash
#SBATCH --job-name=mc-molnl-extend-report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the finished mol_nl extension runs and write what the 2026-11-16 order's items 3
# and 4 ask for: where each run stopped and why, and one curve file holding all seven.
#
# No GPU: it reads trainer_state.json and nanoGPT's logging CSVs. It still goes through
# the scheduler -- nothing computes on the login node.
#
#   sbatch --export=ALL,RUNS_ROOT=<.../molecule_nat_lang>,OUT_DIR=<report dir> \
#          workflows/molnl-extend-report.sbatch
#
# The BERT arms were continued twice from the same checkpoint-24000. The curve file the
# order asks for holds the _extm pass, which kept the optimizer's moments; the _ext pass
# is written beside it as a superseded file, because it is what was measured first.
set -eo pipefail

: "${RUNS_ROOT:?set RUNS_ROOT to the molecule_nat_lang directory holding bert-output-shuffled/ and gpt2-output-grid/}"
: "${OUT_DIR:?set OUT_DIR to the directory the tables are written to}"
STAMP="${STAMP:-$(date +%Y%m%d)}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

BERT_ROOT="${RUNS_ROOT}/bert-output-shuffled"
GPT2_ROOT="${RUNS_ROOT}/gpt2-output-grid"
for d in "${BERT_ROOT}" "${GPT2_ROOT}" "${OUT_DIR}"; do
  [ -d "${d}" ] || { echo "not a directory: ${d}" >&2; exit 1; }
done

if [ -z "${LOG_DIRS:-}" ]; then
  LOG_DIRS=""
  for d in "$(dirname "${PYROOT}")"/*/workflows/slurm-logs; do
    [ -d "${d}" ] && LOG_DIRS="${LOG_DIRS} ${d}"
  done
fi

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== extend-report on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    runs:    ${RUNS_ROOT}"
echo "    outputs: ${OUT_DIR}"
echo "    logs:   ${LOG_DIRS}"

rc=0
# shellcheck disable=SC2086  -- LOG_DIRS is a list of directories, not one word
echo "=== status, every run of both passes ==="
"${PYTHON}" scripts/molnl_extend_status.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" --markdown || rc=$?
"${PYTHON}" scripts/molnl_extend_status.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" \
  > "${OUT_DIR}/molnl-extend-status-${STAMP}.json" || rc=$?

echo "=== the seven as one curve each (BERT: the extm pass) ==="
"${PYTHON}" scripts/molnl_extended_curves_tsv.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" --cross-check \
  --log-dirs ${LOG_DIRS} \
  --out "${OUT_DIR}/rt1-molnl-extended-curves-${STAMP}.tsv" || rc=$?

echo "=== the superseded BERT pass, kept readable ==="
"${PYTHON}" scripts/molnl_extended_curves_tsv.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" --cross-check \
  --arch bert --bert-pass ext --allow-running --log-dirs ${LOG_DIRS} \
  --out "${OUT_DIR}/rt1-molnl-extended-curves-bert-momentless-${STAMP}.tsv" || rc=$?
# --allow-running is there because that pass ended at the 48,000 ceiling rather than at a
# stop, which is the thing the flag exists to make deliberate.

echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
