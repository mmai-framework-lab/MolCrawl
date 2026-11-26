#!/bin/bash
#SBATCH --job-name=mc-molnl-floor-provenance
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# What the four quoted molnl floors are, and which of them answer which metric.
#
# The 2026-11-26 data order's item 5 puts two pairs against each other: the reports
# carry 3.8562 and 4.6449, and the order carries 3.8638 and 4.6514. All four are in
# this project's own records, and they are not one quantity measured four times:
#
#   3.8562  unigram at [MASK] positions, TRAIN split, a fixed random 10,000-block
#           sample of 318,118, fit on 200,000 train rows, mask rate 0.2
#           (report/mlm-baselines-20261106-molnl-train/molnl-train.json)
#   3.8638  the same quantity on 8,000 blocks, job 22503 of 2026-08-12, which printed
#           no split (workflows/slurm-logs/mc-molnl-probeprep-22503.out)
#   4.6449  the same quantity on the VALID split, all 1,631 blocks
#           (report/mlm-baselines-20261102/full/molnl.json)
#   4.6514  a DIFFERENT quantity: the unigram over EVERY token of the valid split,
#           which is what an autoregressive loss is read against, vocab 50,257
#           (report/molnl-gpt2-unigram-floor.json)
#
# So the pair in the order mixes a train-split MLM floor with a valid-split
# all-token floor. This job measures all of them again under one pipeline, so the
# answer rests on a measurement rather than on four records of different ages:
#
#   1. train, 8,000 rows     -- does job 22503's number come back?
#   2. train, 10,000 rows    -- does 2026-11-06's number come back?
#   3. train, LARGE rows     -- the value the two of them are estimating
#   4. valid, whole split    -- 4.6449
#   5. valid, every token    -- 4.6514, on the tree the grid actually ran on
#
# No GPU: every number is a count table fitted on train rows and scored at the
# positions the collator masks, or over every token. It is still not login-node work.
#
#   sbatch --export=ALL,FLOOR_OUT_DIR=<dir outside every checkout>[,FLOOR_CONFIG=<cfg>] \
#          [,FLOOR_LARGE_ROWS=100000][,FLOOR_SKIP_CLM=1] \
#          workflows/molnl-floor-provenance.sbatch
set -uo pipefail

: "${FLOOR_OUT_DIR:?export FLOOR_OUT_DIR to where the JSONs go, outside every input tree}"
CONFIG="${FLOOR_CONFIG:-molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_small_b2048_lr8p94e4.py}"
LARGE="${FLOOR_LARGE_ROWS:-100000}"

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

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the molnl input root the grid ran on}"
[ -d "${LEARNING_SOURCE_DIR}" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} is not a directory" >&2; exit 1; }

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "${REPO}"; mkdir -p workflows/slurm-logs "${FLOOR_OUT_DIR}"

echo "=== molnl floor provenance on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config:  ${CONFIG}"
echo "input:   ${LEARNING_SOURCE_DIR}"
echo "outputs: ${FLOOR_OUT_DIR}"
echo "large:   ${LARGE} rows"

rc=0

# FLOOR_TABLE_ONLY=1 rebuilds the table from JSONs already in FLOOR_OUT_DIR and
# measures nothing. The measurement is the expensive half and does not change when
# the way it is read back does.
if [ "${FLOOR_TABLE_ONLY:-0}" = "1" ]; then
  echo
  echo "########## the table alone, from what is already measured ##########"
  "${PYTHON}" scripts/molnl_floor_provenance.py --results "${FLOOR_OUT_DIR}" \
      --out "${FLOOR_OUT_DIR}/molnl-floor-provenance.md"
  rc=$?
  echo
  echo "=== rc=${rc} @ $(date -Is) ==="
  exit "${rc}"
fi

run_mlm() {  # label split rows
  local label="$1" split="$2" rows="$3"
  echo
  echo "########## ${label}: ${split} split, ${rows:-the evaluation cap} rows ##########"
  "${PYTHON}" scripts/mlm_lookup_baselines.py \
      --config "${CONFIG}" --label "${label}" --split "${split}" \
      --unigram-only --fit-rows 200000 \
      ${rows:+--eval-rows "${rows}"} \
      --output "${FLOOR_OUT_DIR}/${label}.json"
  local step=$?
  echo "-- ${label} rc=${step}"
  [ "${step}" -eq 0 ] || rc="${step}"
}

run_mlm molnl-train-8000 train 8000
run_mlm molnl-train-10000 train 10000
run_mlm "molnl-train-${LARGE}" train "${LARGE}"
# The valid split is 1,631 blocks, which is under every cap, so it is scored whole
# whatever is asked for.
run_mlm molnl-valid-whole valid ""

if [ "${FLOOR_SKIP_CLM:-0}" != "1" ]; then
  echo
  echo "########## every token of valid: the floor an autoregressive loss is read against ##########"
  DS="${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset_shuffled"
  if [ -d "${DS}" ]; then
    "${PYTHON}" scripts/measure_clm_ngram_baselines.py \
        --dataset-dir "${DS}" --vocab 50257 \
        --output "${FLOOR_OUT_DIR}/molnl-valid-all-tokens.json"
    step=$?; echo "-- all-token rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"
  else
    echo "!! no packed dataset at ${DS}; the all-token floor is not measured here"
    rc=1
  fi
  # 4.6514 was measured on another tree than the one this grid ran on, so the tree is
  # part of the answer rather than a detail: the same definition on two builds says
  # whether the quoted number belongs to this grid's data at all.
  if [ -n "${FLOOR_CLM_RECORD_DIR:-}" ]; then
    echo
    echo "---------- the same all-token floor on the tree 4.6514 came from ----------"
    DS2="${FLOOR_CLM_RECORD_DIR}/molecule_nat_lang/training_ready_hf_dataset_shuffled"
    if [ -d "${DS2}" ]; then
      "${PYTHON}" scripts/measure_clm_ngram_baselines.py \
          --dataset-dir "${DS2}" --vocab 50257 \
          --output "${FLOOR_OUT_DIR}/molnl-valid-all-tokens-record-tree.json"
      step=$?; echo "-- all-token (record tree) rc=${step}"
      [ "${step}" -eq 0 ] || rc="${step}"
    else
      echo "!! no packed dataset at ${DS2}"
      rc=1
    fi
  fi
fi

echo
echo "########## what came back, against what is quoted ##########"
"${PYTHON}" scripts/molnl_floor_provenance.py --results "${FLOOR_OUT_DIR}" \
    --out "${FLOOR_OUT_DIR}/molnl-floor-provenance.md"
step=$?; echo "-- table rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
