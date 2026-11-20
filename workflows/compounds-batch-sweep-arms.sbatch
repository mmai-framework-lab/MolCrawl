#!/bin/bash
#SBATCH --job-name=mc-bsweep-arms
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Derive the compounds batch-size grid's arms, write its 68 configs, and check both.
#
# No GPU and no training data: it is arithmetic over the rules the 2026-11-19c and
# 2026-11-20 directives state, plus the configs that follow from them. It runs here rather than on
# the login node because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS_OUT=<...>/rt1-...tsv[,ARMS_SET=order][,WRITE_CONFIGS=1] \
#          workflows/compounds-batch-sweep-arms.sbatch
#
# ARMS_OUT has no default: the report directory is outside every checkout and this
# repository is public. ARMS_SET is one GPU's column count -- 'micro-order' is the
# 2026-11-19 micro-batch directive, '19c-s8' the table it replaced. WRITE_CONFIGS=1
# writes the 68 config files; unset, the configs on disk are only checked.
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${ARMS_OUT:?set ARMS_OUT to the TSV to write}"
SET="${ARMS_SET:-micro-order}"
WRITE="${WRITE_CONFIGS:-0}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== compounds batch-sweep arms on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:      ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "micro set: ${SET}"
echo "configs:   $([ "${WRITE}" = "1" ] && echo write || echo "check only")"
echo "output:    ${ARMS_OUT}"

rc=0

echo
echo "########## 1. the arm inventory, against the directive's tables ##########"
"${PYTHON}" scripts/compounds_batch_sweep_arms.py \
    --micro-batch-set "${SET}" --out "${ARMS_OUT}"
step=$?; echo "-- arms rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 2. the 68 configs, against the inventory ##########"
if [ "${WRITE}" = "1" ]; then
  "${PYTHON}" scripts/compounds_batch_sweep_configs.py --arms "${ARMS_OUT}"
else
  "${PYTHON}" scripts/compounds_batch_sweep_configs.py --arms "${ARMS_OUT}" --check
fi
step=$?; echo "-- configs rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 3. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep.py \
    tests/unit/test_report_tsv.py tests/unit/test_config_seed_42.py \
    tests/unit/test_bert_eval_subset.py
step=$?; echo "-- pytest rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
