#!/bin/bash
#SBATCH --job-name=mc-mlm-token-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Report the MLM special-token ids each modality's BERT config resolves, so the
# silent-None path in models/bert/main.py can be ruled in or out per modality
# before checkpoint selection moves to eval_loss_mask.
#
#   sbatch workflows/mlm-special-tokens-check.sbatch
#
# Nothing is trained. A GPU is requested only because it is the one partition here.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

# A git worktree holds only tracked files, so the interpreter is not inside it.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

# Results are a durable record, so they must not land in a disposable worktree.
OUT_ROOT="${VERIFY_OUT_ROOT:?set VERIFY_OUT_ROOT to a results root outside any worktree}"
OUT="${OUT_ROOT}/mlm-special-tokens.json"
mkdir -p "$(dirname "${OUT}")"

echo "=== mlm special-token check ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "outputs: ${OUT}"
"${PYTHON}" scripts/check_mlm_special_tokens.py --output "${OUT}" && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? could only ever be 0.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
