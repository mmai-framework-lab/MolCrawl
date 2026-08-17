#!/bin/bash
#SBATCH --job-name=mc-molnet-gpt2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# MoleculeNet property prediction for the compounds GPT-2 ladder.
#
# The ladder writes nanoGPT-style ckpt.pt, which is what GPT2Adapter loads, and the
# compounds tokenizer comes from assets/molecules/vocab.txt. Note the size suffix:
# get_gpt2_output_path normalises "xl" to "ex-large", so that is the directory name.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<packed dir> MOLECULENET_DIR=<eval dir> \
#     sbatch --export=ALL,SIZE=small,SUBTASKS="esol" \
#     workflows/eval-moleculenet-compounds-gpt2.sbatch
set -e
: "${SIZE:?SIZE must be set (small|medium|large|ex-large)}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR before submitting}"
: "${MOLECULENET_DIR:?export MOLECULENET_DIR (…/eval/moleculenet)}"
SUBTASKS="${SUBTASKS:-esol}"
SPLIT="${SPLIT:-scaffold}"
SEED="${SEED:-0}"
# --max-examples is deprecated upstream (it is re-read as n-examples anyway);
# leave N_EXAMPLES empty for a full-dataset run.
N_EXAMPLES="${N_EXAMPLES:-}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
MODEL_PATH="${MODEL_PATH:-${LEARNING_SOURCE_DIR}/compounds/gpt2-output/compounds-${SIZE}/ckpt.pt}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/experiment_data/eval/moleculenet-gpt2-${SIZE}}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT}" workflows/slurm-logs

echo "=== moleculenet / compounds gpt2-${SIZE} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "model : ${MODEL_PATH}"
echo "tasks : ${SUBTASKS} | split=${SPLIT} seed=${SEED} ${N_EXAMPLES:+n_examples=${N_EXAMPLES}}"
echo "out   : ${OUT}"
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
[ -f "${MODEL_PATH}" ] || { echo "checkpoint not found: ${MODEL_PATH}"; exit 1; }

for subtask in ${SUBTASKS}; do
  echo "--- ${subtask} ---"
  "${REPO}/miniconda/bin/python" -m molcrawl.tasks.evaluation.moleculenet \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path assets/molecules/vocab.txt \
    --arch gpt2 --modality compounds --size "${SIZE}" \
    --subtask "${subtask}" \
    --task-dir "${MOLECULENET_DIR}/${subtask}" \
    --output-dir "${OUT}/${subtask}" \
    --split "${SPLIT}" --seed "${SEED}" \
    ${N_EXAMPLES:+--n-examples ${N_EXAMPLES}}
done
echo "=== rc=$? @ $(date -Is) ==="
