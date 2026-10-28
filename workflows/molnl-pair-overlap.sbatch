#!/bin/bash
#SBATCH --job-name=molnl-pairoverlap
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Does the pretraining corpus hold the evaluation pairs, and which ones?
#
# No GPU is requested -- it decodes the packed corpus and calls RDKit -- but a scan of
# 3.3M records does not belong on the login node. The whole scan takes about 20 minutes.
#
#   sbatch --export=ALL,DATASET=<packed corpus>,PAIRS_ROOT=<Mol-Instructions dir>,\
#          OUT=<json>,PAIR_INDEX_OUT=<json> workflows/molnl-pair-overlap.sbatch
#
# selfies is not in the training environment. Where it is installed as a plain
# directory, carry it in on PYTHONPATH, which is appended to rather than replaced:
#   --export=ALL,PYTHONPATH=<dir holding selfies>,DATASET=...
#
# Optional: SUBSET, SPLIT, SHOW_RECORDS (print this many records as read, which is how
# the record form a retrieval measurement imitates is checked), CANONICAL_CORPUS=1
# (canonicalise every corpus molecule as well -- adds about 40 minutes), PROCESSES,
# PYTHON.
set -eo pipefail

: "${DATASET:?set DATASET to the packed training_ready_hf_dataset the models read}"
: "${PAIRS_ROOT:?set PAIRS_ROOT to the Mol-Instructions dataset directory}"
[ -d "${DATASET}" ] || { echo "DATASET=${DATASET} is not a directory" >&2; exit 1; }
[ -d "${PAIRS_ROOT}" ] || { echo "PAIRS_ROOT=${PAIRS_ROOT} is not a directory" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The corpus is packed with GPT-2 BPE and has to be decoded to be read back.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== pair overlap on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "=== corpus: ${DATASET} ==="
echo "=== pairs:  ${PAIRS_ROOT} ==="
echo "=== out:    ${OUT:-<none>} / ${PAIR_INDEX_OUT:-<none>} ==="
rc=0
"${PYTHON}" scripts/molnl_downstream_pair_overlap.py \
  --dataset "${DATASET}" --pairs-root "${PAIRS_ROOT}" \
  --split "${SPLIT:-train}" --subset "${SUBSET:-molecular_description_generation}" \
  --processes "${PROCESSES:-8}" \
  ${SHOW_RECORDS:+--show-records "${SHOW_RECORDS}"} \
  ${CANONICAL_CORPUS:+--canonical-corpus} \
  ${PAIR_INDEX_OUT:+--pair-index-out "${PAIR_INDEX_OUT}"} \
  ${OUT:+--out "${OUT}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
