#!/bin/bash
#SBATCH --job-name=mc-cmp-clm-base
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The model-free floor for a causal model on this corpus: uniform, unigram and bigram,
# counted on train and scored on valid. No GPU and no model -- it is a count table.
#
# A GPT-2 loss says how many nats a token cost, not how many it could have cost. The
# BERT floor (2.4733) is not the one to draw: it scores masked positions with context
# on both sides, where a causal model sees only the left.
#
#   sbatch --export=ALL,CLM_DATASET=<packed gpt2 dataset root>,CLM_VOCAB=<n>,\
# CLM_OUT=<file.json> workflows/compounds-clm-baselines.sbatch
set -uo pipefail
: "${CLM_DATASET:?set CLM_DATASET to the packed dataset root holding train/ and valid/}"
: "${CLM_VOCAB:?set CLM_VOCAB to the vocabulary size}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds CLM baselines on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "dataset: ${CLM_DATASET}"
"${PYTHON}" scripts/measure_clm_ngram_baselines.py \
    --dataset-dir "${CLM_DATASET}" --vocab "${CLM_VOCAB}" \
    --smoothing "${CLM_SMOOTHING:-0.1}" ${CLM_OUT:+--output "${CLM_OUT}"}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
