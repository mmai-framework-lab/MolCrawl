#!/bin/bash
#SBATCH --job-name=mc-smiles-profile
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# What a SMILES corpus loses at each filter the ZINC-22 order specifies, and how
# often the existing vocabulary meets a character it does not have. No GPU and no
# model: RDKit and the tokenizer over the corpus.
#
#   sbatch --export=ALL,PROFILE_ARGS='--parquet <file> --label organix13' \
#          workflows/profile-smiles-corpus.sbatch
set -uo pipefail
: "${PROFILE_ARGS:?export PROFILE_ARGS with the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
# assets/ lives in the main checkout when this runs from a worktree.
VOCAB="${PROFILE_VOCAB:-${PYROOT}/assets/molecules/vocab.txt}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== SMILES corpus profile on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "vocab: ${VOCAB}"
# shellcheck disable=SC2086 -- PROFILE_ARGS is an argument list on purpose
set -f
"${PYTHON}" scripts/profile_smiles_corpus.py --vocab "${VOCAB}" \
    --processes "${SLURM_CPUS_PER_TASK:-32}" ${PROFILE_ARGS}
rc=$?
set +f
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
