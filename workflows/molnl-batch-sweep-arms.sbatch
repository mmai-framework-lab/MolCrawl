#!/bin/bash
#SBATCH --job-name=mc-molnl-bsweep-arms
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Derive the molnl batch-size grid's arms, write its 68 configs, and check both.
#
# No GPU and no training: it is arithmetic over the rules the 2026-11-20 directive
# states, plus the configs that follow from them. It runs here rather than on the login
# node because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS_OUT=<...>/rt1-...tsv[,ARMS_SET=current][,WRITE_CONFIGS=1] \
#          workflows/molnl-batch-sweep-arms.sbatch
#
# ARMS_OUT has no default: the report directory is outside every checkout and this
# repository is public. ARMS_SET names one GPU's block count table (§4 leaves it to
# measurement; 'current' is what molnl's configs carry today). WRITE_CONFIGS=1 writes
# the 68 config files; unset, the configs on disk are only checked.
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${ARMS_OUT:?set ARMS_OUT to the TSV to write}"
SET="${ARMS_SET:-current}"
WRITE="${WRITE_CONFIGS:-0}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A worktree holds only tracked files, so the interpreter and the tokenizer are not
# inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# --check executes each config the way the launcher does, and molnl's base configs build
# their tokenizer and call the vocabulary guard at import time.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== molnl batch-sweep arms on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:      ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "micro set: ${SET}"
echo "configs:   $([ "${WRITE}" = "1" ] && echo write || echo "check only")"
echo "output:    ${ARMS_OUT}"

rc=0

echo
echo "########## 1. the arm inventory, against the directive's tables ##########"
"${PYTHON}" scripts/molnl_batch_sweep_arms.py \
    --micro-batch-set "${SET}" --out "${ARMS_OUT}"
step=$?; echo "-- arms rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 2. the 68 configs, against the inventory ##########"
if [ "${WRITE}" = "1" ]; then
  "${PYTHON}" scripts/molnl_batch_sweep_configs.py --arms "${ARMS_OUT}"
else
  "${PYTHON}" scripts/molnl_batch_sweep_configs.py --arms "${ARMS_OUT}" --check
fi
step=$?; echo "-- configs rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 3. unit tests ##########"
# The grid's own invariants, plus the two standing checks a new config can break:
# every config in the tree declares seed 42, and none of them declares an eval subset.
# The whole suite goes through workflows/run-unit-tests.sbatch before anything is
# submitted; these three are the ones this change can move.
"${PYTHON}" -m pytest -q tests/unit/test_molnl_batch_sweep.py \
    tests/unit/test_config_seed_42.py tests/unit/test_bert_eval_subset.py
step=$?; echo "-- pytest rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
