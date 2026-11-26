#!/bin/bash
#SBATCH --job-name=mc-molnl-chebi20-prep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# ChEBI-20 for the 2026-11-26 retrieval order: build the 33,010-pair table, then measure
# how much of its test split the molnl pretraining corpus holds (item 6).
#
# No GPU. It still goes through the scheduler -- nothing computes on the login node.
#
#   sbatch --export=ALL,TEXT2MOL=<dir>,MOLT5=<dir>,CORPUS=<packed dataset>,OUT_DIR=<dir> \
#          workflows/molnl-chebi20-prep.sbatch
#
# TEXT2MOL holds Text2Mol's training/val/test.txt and sdf/ with the unzipped SDFs;
# MOLT5 holds MolT5's train/validation/test.txt (workflows/data/eval-data-chebi20.sh).
# The table is written beside the Text2Mol files; the overlap files go to OUT_DIR.
set -eo pipefail

: "${TEXT2MOL:?set TEXT2MOL to the Text2Mol data directory}"
: "${MOLT5:?set MOLT5 to the MolT5 ChEBI-20 directory}"
: "${CORPUS:?set CORPUS to the packed molnl corpus the pretraining read}"
: "${OUT_DIR:?set OUT_DIR to the directory the overlap files are written to}"
STAMP="${STAMP:-$(date +%Y%m%d)}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== chebi20 prep on ${SLURMD_NODENAME} @ $(date -Is) commit $(git rev-parse --short HEAD) ==="
"${PYTHON}" scripts/molnl_chebi20_build.py --text2mol "${TEXT2MOL}" --molt5 "${MOLT5}" \
  --out "${TEXT2MOL}/chebi20.jsonl" --summary "${TEXT2MOL}/chebi20-counts.json"

"${PYTHON}" scripts/molnl_chebi20_overlap.py --dataset "${CORPUS}" \
  --chebi20 "${TEXT2MOL}/chebi20.jsonl" --which test --processes "${SLURM_CPUS_PER_TASK:-8}" \
  --out-tsv "${OUT_DIR}/molnl-chebi20-overlap-items-${STAMP}.tsv" \
  --out-json "${OUT_DIR}/molnl-chebi20-overlap-counts-${STAMP}.json"
echo "=== done @ $(date -Is) ==="
