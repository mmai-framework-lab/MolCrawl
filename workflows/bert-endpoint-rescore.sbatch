#!/bin/bash
#SBATCH --job-name=mc-bert-rescore
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Move the compounds MLM endpoints onto the random eval slice.
#
# Every one of them was measured on the first 10,000 rows of its eval split, which the
# source-ordered compounds sets make unrepresentative. Each arm is scored on both
# slices here, so the difference is readable rather than inferred.
#
# The W arms trained on the packed build that now lives under _nonshuffled - the
# canonical name holds the shuffled rebuild, which they never saw.
set -eo pipefail
# Code root is wherever this was submitted from (a worktree is fine); the datasets
# live with the main checkout, so they come in as env vars rather than being derived.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${PACKED_DIR:?export PACKED_DIR (…/learning_source_20260805_compounds_packed)}"
: "${B200_DIR:?export B200_DIR (…/learning_source_20260723_b200prep)}"
# The interpreter lives with the main checkout, not in a worktree.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON"; exit 1; }
PACKED="${PACKED_DIR}"
B200="${B200_DIR}"
export LEARNING_SOURCE_DIR="${PACKED}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/miniconda/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT="${PACKED}/experiment_data/eval/mlm-endpoints"
cd "${REPO}"; mkdir -p "${OUT}"

PACKED_BERT="${PACKED}/compounds/organix13/training_ready_hf_dataset_bert_nonshuffled"
ONEMOL="${B200}/compounds/organix13/training_ready_hf_dataset_bert"

echo "=== compounds MLM endpoint re-score ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"

# label | checkpoint | dataset | max_length | document_masking
ARMS=(
  "V1L|${PACKED}/runs/crossdoc-V1L-1mol128-long/checkpoint-15000|${ONEMOL}|128|"
  "W2|${PACKED}/runs/crossdoc-W2-docmask-long/checkpoint-15000|${PACKED_BERT}|1024|--document-masking"
  "W3|${PACKED}/runs/crossdoc-W3-docmask-lr3e-4/checkpoint-15000|${PACKED_BERT}|1024|--document-masking"
  "W4|${PACKED}/runs/crossdoc-W4-docmask-lr1e-3/checkpoint-15000|${PACKED_BERT}|1024|--document-masking"
)
rc=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r LABEL CKPT DATA MAXLEN DOCMASK <<< "${arm}"
  for slice in head random; do
    echo "--- ${LABEL} / ${slice} slice ---"
    "${PYTHON}" scripts/eval_mlm_checkpoint.py \
      --checkpoint "${CKPT}" --dataset-dir "${DATA}" --max-length "${MAXLEN}" \
      --eval-subset "${slice}" --label "${LABEL}" \
      --output "${OUT}/${LABEL}_${slice}.json" ${DOCMASK} || rc=$?
  done
done
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
