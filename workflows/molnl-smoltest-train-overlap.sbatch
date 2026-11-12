#!/bin/bash
#SBATCH --job-name=mc-molnl-smoltest-trainsplit
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# 2026-11-10 judgment 3.2.1 and 3.2.2: split the 14 test-split retrieval measurements by
# whether the pair's molecule occurs in train, and count which tasks the matched
# molecules occur in.
#
#   sbatch --export=ALL,ARROW_TRAIN=<arrow_splits/train.arrow>,PAIRS_FILE=<jsonl>,\
#          RANKS_DIR=<dir of retrieval-*-ranks.tsv>,OUT=<json>,TSV=<tsv> \
#          workflows/molnl-smoltest-train-overlap.sbatch
#
# Optional: PROCESSES, BATCH_SIZE, GPT2_TOKENIZER_DIR, PYTHON.
#
# The work needs no GPU -- decoding the response side and calling RDKit are the whole
# cost -- but --gpus=0 does not get an allocation without one here: the gpu partition is
# the default and bills by GPU. What it buys is a compute node rather than a login node,
# where a scan of 3.3M records may not run.
set -eo pipefail

: "${ARROW_TRAIN:?set ARROW_TRAIN to the arrow_splits/train.arrow directory}"
: "${PAIRS_FILE:?set PAIRS_FILE to the frozen test-pair jsonl that was scored}"
: "${RANKS_DIR:?set RANKS_DIR to the directory holding retrieval-*-ranks.tsv}"
: "${OUT:?set OUT to the json to write}"
: "${TSV:?set TSV to the tsv to write}"
[ -d "${ARROW_TRAIN}" ] || { echo "ARROW_TRAIN=${ARROW_TRAIN} is not a directory" >&2; exit 1; }
[ -f "${PAIRS_FILE}" ] || { echo "PAIRS_FILE=${PAIRS_FILE} is not a file" >&2; exit 1; }
[ -d "${RANKS_DIR}" ] || { echo "RANKS_DIR=${RANKS_DIR} is not a directory" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The response side is stored as ids and has to be decoded with the corpus's own BPE.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${OUT}")" "$(dirname "${TSV}")"

echo "=== molnl smoltest train overlap split on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    train   ${ARROW_TRAIN}"
echo "    pairs   ${PAIRS_FILE}"
echo "    ranks   ${RANKS_DIR}"
echo "    outputs ${OUT} / ${TSV}"
echo "    tokenizer ${GPT2_TOKENIZER_DIR}"

rc=0
"${PYTHON}" scripts/molnl_smoltest_train_overlap_split.py \
  --arrow-train "${ARROW_TRAIN}" \
  --pairs-file "${PAIRS_FILE}" \
  --ranks-dir "${RANKS_DIR}" \
  --processes "${PROCESSES:-16}" \
  --batch-size "${BATCH_SIZE:-2000}" \
  --out "${OUT}" --tsv "${TSV}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
