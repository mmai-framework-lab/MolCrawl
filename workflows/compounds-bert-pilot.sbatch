#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# compounds bert-small pilot: is the 2.16 plateau a step-count problem or an LR problem?
# All conditions see the SAME token budget (~786M): global batch x max_steps x 1024 is
# held constant, so a win for TAG=b640 means steps, and a win for lr3e-4 / lr6e-4 means LR.
# Submit with: sbatch --job-name=mc-cmp-bpilot-<tag> --export=ALL,TAG=<tag>,LR=<lr>,BS=<n>,GA=<n>,STEPS=<n>,WARM=<n>,EVERY=<n> ...
set -e
: "${TAG:?}" "${LR:?}" "${BS:?}" "${GA:?}" "${STEPS:?}" "${WARM:?}" "${EVERY:?}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$PWD"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the packed compounds learning_source root (learning_source_20260805_compounds_packed in the main checkout)}"
export LEARNING_SOURCE_DIR
OUT="${LEARNING_SOURCE_DIR}/runs/pilot-bert-${TAG}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH}"
export HF_DATASETS_CACHE="${REPO}/tmp/hf-cache-compounds"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}"
echo "=== compounds bert-small PILOT [${TAG}] ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "LR=${LR} batch=${BS} grad_accum=${GA} -> global $((BS*GA*4)) seq | max_steps=${STEPS} warmup=${WARM} eval_every=${EVERY}"
echo "token budget = $((BS*GA*4))*${STEPS}*1024"
"${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py molcrawl/tasks/pretrain/configs/compounds/bert_small.py \
  --model_path="${OUT}" --learning_rate=${LR} --batch_size=${BS} \
  --gradient_accumulation_steps=${GA} --max_steps=${STEPS} --warmup_steps=${WARM} \
  --log_interval=${EVERY} --save_steps=${STEPS}
echo "=== rc=$? @ $(date -Is) ==="
