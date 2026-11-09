#!/bin/bash
#SBATCH --job-name=mc-molnl-provenance
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Where the molnl retrieval scoring got its tokenizers, checkpoints and pairs.
#
#   sbatch --export=ALL,PAIRS_FILE=<jsonl>,PAIR_INDEX=<json>,BERT_CKPT=<dir>,\
#          GPT2_CKPT=<ckpt.pt>,OUT=<json> workflows/molnl-scoring-provenance.sbatch
#
# Optional: GPT2_CKPTS (space separated ckpt.pt list whose steps are read),
# INVENTORY_ROOT (walked for checkpoints that carry no tokenizer), MODALITY,
# GPT2_TOKENIZER_DIR, PYTHON.
#
# No GPU: --gpus=0 keeps it off the accelerators while still running on a compute node,
# because nothing here may run on a login node. The tokenizer tiers and the nanoGPT
# metadata read are the whole cost.
set -eo pipefail

: "${OUT:?set OUT to the json to write}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The same default the scan got on 2026-10-28 and the scoring on 2026-11-05. Left
# explicit so the run records which tokenizer directory answered.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${OUT}")"

STEP_ARGS=()
for c in ${GPT2_CKPTS:-}; do STEP_ARGS+=(--gpt2-ckpt "${c}"); done

echo "=== molnl scoring provenance on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    GPT2_TOKENIZER_DIR ${GPT2_TOKENIZER_DIR}"
echo "    out ${OUT}"

rc=0
"${PYTHON}" scripts/molnl_scoring_provenance.py \
  ${PAIRS_FILE:+--pairs-file "${PAIRS_FILE}"} \
  ${PAIR_INDEX:+--pair-index "${PAIR_INDEX}"} \
  --modality "${MODALITY:-molecule_nat_lang}" \
  "${STEP_ARGS[@]}" \
  ${INVENTORY_ROOT:+--inventory-root "${INVENTORY_ROOT}"} \
  --out "${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
