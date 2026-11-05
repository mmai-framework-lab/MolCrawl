#!/bin/bash
#SBATCH --job-name=molnl-test-pairs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Freeze the molecule/description pairs of the SMolInstruct test split.
#
# No GPU is requested -- it reads one split and decodes its responses -- but it does not
# belong on the login node. Everything downstream reads the file this writes, so the
# scoring and the overlap check cannot drift onto different pairs.
#
#   sbatch --export=ALL,ARROW_ROOT=<arrow_splits>,OUT=<jsonl>,COUNTS_OUT=<json> \
#          workflows/molnl-test-pairs.sbatch
#
# Optional: SPLIT (test), PAIRS (1000), SEED (42), GPT2_TOKENIZER_DIR, PYTHON.
set -eo pipefail

: "${ARROW_ROOT:?set ARROW_ROOT to the directory holding test.arrow}"
: "${OUT:?set OUT to the pair file to write}"
[ -d "${ARROW_ROOT}" ] || { echo "ARROW_ROOT=${ARROW_ROOT} is not a directory" >&2; exit 1; }

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
# The split keeps its responses as token ids only, so the corpus's own BPE decodes them.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== test pairs on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "=== records: ${ARROW_ROOT} ==="
echo "=== out:     ${OUT} ==="
rc=0
"${PYTHON}" scripts/molnl_smolinstruct_test_pairs.py \
  --arrow-root "${ARROW_ROOT}" --split "${SPLIT:-test}" \
  --pairs "${PAIRS:-1000}" --seed "${SEED:-42}" \
  --out "${OUT}" ${COUNTS_OUT:+--counts-out "${COUNTS_OUT}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
