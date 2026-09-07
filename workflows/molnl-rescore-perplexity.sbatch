#!/bin/bash
#SBATCH --job-name=mc-molnl-rescore
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Whole-split perplexity for the mol_nl GPT-2 ladder, on the checkpoints adopted.
#
# The generic workflows/eval-ladder-perplexity-gpt2.sbatch resolves checkpoints and
# dataset from the modality's defaults. mol_nl's adopted ladder is neither: it lives
# in gpt2-output-shuffled-<stage>/ and was trained on the shuffled rebuild, so both
# have to be named. Getting either wrong is not visible in the output -- the run
# succeeds and reports numbers for whatever it did score.
#
# That is how molnl_valid_full.json came to hold the 2026-08-20 source-order 373-iter
# figures long after the ladder had been rebuilt shuffled (#143) and re-run to 3,000
# iters, and three sizes were reported from it. The old file is kept beside it as
# molnl_valid_full_sourceorder_stage1.json.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,STAGE=s2 \
#     workflows/molnl-rescore-perplexity.sbatch
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
STAGE="${STAGE:-s2}"
SIZES="${SIZES:-small medium large xl}"
SPLIT="${SPLIT:-valid}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

RUNS_ROOT="${RUNS_ROOT:-${LEARNING_SOURCE_DIR}/molecule_nat_lang}"
CKPT_TEMPLATE="${CKPT_TEMPLATE:-${RUNS_ROOT}/gpt2-output-shuffled-${STAGE}/molecule_nat_lang-{size}/ckpt.pt}"
DATASET_DIR="${DATASET_DIR:-${RUNS_ROOT}/training_ready_hf_dataset_shuffled}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/experiment_data/eval/ladder-perplexity}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs

echo "=== mol_nl ladder re-score (stage ${STAGE}, shuffled) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "sizes: ${SIZES} | split=${SPLIT}"
echo "ckpts: ${CKPT_TEMPLATE}"
echo "data : ${DATASET_DIR}"
echo "outputs: ${OUT}"

rc=0
"${PYTHON}" scripts/eval_gpt2_perplexity.py \
  --modality molecule_nat_lang \
  --sizes ${SIZES} \
  --split "${SPLIT}" \
  --checkpoint-template "${CKPT_TEMPLATE}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUT}" \
  --output-name "${OUTPUT_NAME:-molnl_${SPLIT}_full.json}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
