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
#          [,SHAPES_TABLE_OUT=<...>.md][,PROBE_LOG_GLOB=<...>/mc-molnl-bshape-*.out] \
#          workflows/molnl-batch-sweep-arms.sbatch
#
# ARMS_OUT has no default: the report directory is outside every checkout and this
# repository is public. ARMS_SET names one GPU's block count table (§4 leaves it to
# measurement; 'current' is what molnl's configs carry today). WRITE_CONFIGS=1 writes
# the 68 config files; unset, the configs on disk are only checked.
#
# CHOOSE_FROM_PROBES=1 with PROBE_LOG_GLOB first reads the §4 probe logs and writes the
# chosen block count per size into scripts/molnl_batch_shape_choice_20261120.json, which
# the derivation cross-checks its 'measured-1120' literal against. 'adopted-1124' is
# that measurement with the gpt2 xl column at the runner-up shape, which is what the
# grid runs; pass ARMS_SET to derive from either.
#
# SHAPES_TABLE_OUT writes §9's table -- all the arms with one GPU's block count and its
# accumulation count, plus the measured step times PROBE_LOG_GLOB points at. Both are
# optional: before the §4 probes have run there is nothing measured to put in it.
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there. A glob is passed unquoted on purpose and is
# expanded by the generator, not by this shell.
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
echo "s9 table:  ${SHAPES_TABLE_OUT:-(not asked for)}"

rc=0

if [ "${CHOOSE_FROM_PROBES:-0}" = "1" ]; then
  : "${PROBE_LOG_GLOB:?CHOOSE_FROM_PROBES=1 needs PROBE_LOG_GLOB}"
  echo
  echo "########## 0. one GPU's block count per size, out of the probes ##########"
  "${PYTHON}" scripts/molnl_batch_shape_choice.py \
      --probe-logs "${PROBE_LOG_GLOB}" \
      --out scripts/molnl_batch_shape_choice_20261120.json
  step=$?; echo "-- choice rc=${step}"
  # A half-finished measurement must not become a grid: stop here rather than fall
  # back to the current set under a name that says measured.
  [ "${step}" -eq 0 ] || exit "${step}"
fi

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

if [ -n "${SHAPES_TABLE_OUT:-}" ]; then
  echo
  echo "########## 4. the table §9 asks for, before anything is submitted ##########"
  "${PYTHON}" scripts/molnl_batch_sweep_shapes_table.py \
      --arms "${ARMS_OUT}" --set-name "${SET}" --out "${SHAPES_TABLE_OUT}" \
      ${PROBE_LOG_GLOB:+--probe-logs "${PROBE_LOG_GLOB}"}
  step=$?; echo "-- §9 table rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"
fi

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
