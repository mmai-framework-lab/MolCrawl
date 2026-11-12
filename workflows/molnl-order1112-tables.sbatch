#!/bin/bash
#SBATCH --job-name=mc-molnl-order1112-tables
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The two tables the 2026-11-12 order asks for, and the parameter counts their provenance
# block quotes.
#
#   sbatch --export=ALL,BERT_ROOT=<dir>,GPT2_ROOT=<dir>,RETRIEVAL_DIR=<dir>,\
#          OUT_DIR=<dir>[,CROSS_CHECK=<all-curves tsv>,ADOPTED_JSON=<json>,\
#          PAIRS_FILE=<jsonl>] workflows/molnl-order1112-tables.sbatch
#
# The adopted runs are named here, one per size per architecture: the seven the retrieval
# scoring was run on. Override with BERT_RUNS / GPT2_RUNS as space separated <size>=<run>
# pairs -- space separated, not comma separated, because --export splits a value on
# commas and the list would be cut in half.
#
# Reading a nanoGPT xl checkpoint for its parameter count pulls its state dict through
# memory, which is what the 64G is for. No GPU is needed; the gpu partition is the default
# here and bills by GPU, so one is allocated anyway. It runs on a compute node because
# nothing in this project runs on a login node.
set -eo pipefail

: "${BERT_ROOT:?set BERT_ROOT to the bert-output-shuffled directory}"
: "${GPT2_ROOT:?set GPT2_ROOT to the gpt2-output-grid directory}"
: "${RETRIEVAL_DIR:?set RETRIEVAL_DIR to the directory holding retrieval-*.json}"
: "${OUT_DIR:?set OUT_DIR to the directory the tables are written into}"
[ -d "${BERT_ROOT}" ] || { echo "BERT_ROOT=${BERT_ROOT} is not a directory" >&2; exit 1; }
[ -d "${GPT2_ROOT}" ] || { echo "GPT2_ROOT=${GPT2_ROOT} is not a directory" >&2; exit 1; }
[ -d "${RETRIEVAL_DIR}" ] || { echo "RETRIEVAL_DIR=${RETRIEVAL_DIR} is not a directory" >&2; exit 1; }

BERT_RUNS="${BERT_RUNS:-small=bert_small_24k_lr1e3 medium=bert_medium_24k_lr3e4 large=bert_large_24k_lr1p7e4}"
GPT2_RUNS="${GPT2_RUNS:-small=gpt2_small_1500_lr1p2e3 medium=gpt2_medium_1500_lr6e4 large=gpt2_large_1500_lr3e4 xl=gpt2_xl_1500_lr3e4}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

# scripts/ is on the path because the downstream table imports the binomial tail from the
# sheet already handed over rather than carrying a second copy of it.
export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${OUT_DIR}"

SIZES="${OUT_DIR}/molnl-model-sizes-adopted-20261112.json"
CURVES="${OUT_DIR}/rt1-molnl-loss-curves-20261112.tsv"
DOWNSTREAM="${OUT_DIR}/rt1-molnl-downstream-smoltest-20261112.tsv"

echo "=== molnl order 1112 tables on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    bert runs ${BERT_RUNS}"
echo "    gpt2 runs ${GPT2_RUNS}"
echo "    outputs   ${CURVES}"
echo "              ${DOWNSTREAM}"

NAMES_BERT=(); for p in ${BERT_RUNS}; do NAMES_BERT+=("${p#*=}"); done
NAMES_GPT2=(); for p in ${GPT2_RUNS}; do NAMES_GPT2+=("${p#*=}"); done

rc=0
echo "--- 1.2 parameter counts and the batch shape ---"
"${PYTHON}" scripts/molnl_model_sizes.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" \
  --bert-runs "${NAMES_BERT[@]}" --gpt2-runs "${NAMES_GPT2[@]}" \
  --out "${SIZES}" || rc=$?

echo "--- 1. the loss curves ---"
# shellcheck disable=SC2086
[ "${rc}" -eq 0 ] && "${PYTHON}" scripts/molnl_loss_curves_tsv.py \
  --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" \
  --bert-runs ${BERT_RUNS} --gpt2-runs ${GPT2_RUNS} \
  --sizes "${SIZES}" \
  ${CROSS_CHECK:+--cross-check "${CROSS_CHECK}"} \
  ${SLURM_JOB_ID:+--job "${SLURM_JOB_ID}"} \
  --out "${CURVES}" || rc=$?

echo "--- 2. the downstream table ---"
[ "${rc}" -eq 0 ] && "${PYTHON}" scripts/molnl_downstream_tsv.py \
  --retrieval-dir "${RETRIEVAL_DIR}" \
  ${ADOPTED_JSON:+--adopted-json "${ADOPTED_JSON}"} \
  ${PAIRS_FILE:+--pairs-file "${PAIRS_FILE}"} \
  ${SLURM_JOB_ID:+--job "${SLURM_JOB_ID}"} \
  --out "${DOWNSTREAM}" || rc=$?

echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
