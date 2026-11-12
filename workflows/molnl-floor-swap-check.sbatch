#!/bin/bash
#SBATCH --job-name=mc-molnl-floor-swap-check
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# 2026-11-10 judgment 5.3: what moving the molnl floor from the train value to the valid
# one changes in the grid's own verdicts.
#
#   sbatch --export=ALL,CONFIG_BEFORE=<json>,CONFIG_AFTER=<json>,OUT_DIR=<dir> \
#          workflows/molnl-floor-swap-check.sbatch
#
# The floor decides two things in scripts/bert_grid_summary.py besides the line on a
# figure: whether an arm counts as collapsed (it ended back at the floor) and, when it
# did, whether that reads as the degenerate solution or as divergence. Moving the floor
# can therefore move a verdict, so the summary is built twice and the molnl rows compared
# instead of being argued about.
#
# Nothing is written over: both summaries go to OUT_DIR under their own names, and the
# shared report/bert-grid-summary-*.tsv is left alone.
set -eo pipefail

: "${CONFIG_BEFORE:?set CONFIG_BEFORE to the config carrying the old floor}"
: "${CONFIG_AFTER:?set CONFIG_AFTER to the config carrying the new floor}"
: "${OUT_DIR:?set OUT_DIR to the directory the two summaries are written into}"
[ -f "${CONFIG_BEFORE}" ] || { echo "CONFIG_BEFORE=${CONFIG_BEFORE} is not a file" >&2; exit 1; }
[ -f "${CONFIG_AFTER}" ] || { echo "CONFIG_AFTER=${CONFIG_AFTER} is not a file" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${OUT_DIR}"

BEFORE="${OUT_DIR}/bert-grid-summary-floor-before.tsv"
AFTER="${OUT_DIR}/bert-grid-summary-floor-after.tsv"

echo "=== molnl floor swap check on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
echo "--- with the old floor (${CONFIG_BEFORE}) ---"
"${PYTHON}" scripts/bert_grid_summary.py --config "${CONFIG_BEFORE}" --out "${BEFORE}" || rc=$?
echo "--- with the new floor (${CONFIG_AFTER}) ---"
"${PYTHON}" scripts/bert_grid_summary.py --config "${CONFIG_AFTER}" --out "${AFTER}" || rc=$?

echo "--- molnl rows, old then new ---"
grep -h "^molnl" "${BEFORE}" | sort > "${OUT_DIR}/.molnl-before"
grep -h "^molnl" "${AFTER}" | sort > "${OUT_DIR}/.molnl-after"
echo "    old rows: $(wc -l < "${OUT_DIR}/.molnl-before"), new rows: $(wc -l < "${OUT_DIR}/.molnl-after")"
if diff -u "${OUT_DIR}/.molnl-before" "${OUT_DIR}/.molnl-after"; then
  echo "    no molnl row changed"
fi
echo "--- rows of the other modalities, which must not move ---"
grep -hv "^molnl" "${BEFORE}" | sort > "${OUT_DIR}/.others-before"
grep -hv "^molnl" "${AFTER}" | sort > "${OUT_DIR}/.others-after"
if diff -q "${OUT_DIR}/.others-before" "${OUT_DIR}/.others-after" >/dev/null; then
  echo "    unchanged, as they should be"
else
  echo "    CHANGED -- the swap reached a modality it should not have:"
  diff -u "${OUT_DIR}/.others-before" "${OUT_DIR}/.others-after" || true
  rc=1
fi
rm -f "${OUT_DIR}/.molnl-before" "${OUT_DIR}/.molnl-after" \
      "${OUT_DIR}/.others-before" "${OUT_DIR}/.others-after"
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
