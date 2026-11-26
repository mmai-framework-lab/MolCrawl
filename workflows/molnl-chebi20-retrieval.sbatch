#!/bin/bash
#SBATCH --job-name=mc-molnl-chebi20-ret
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# One arm of the 2026-11-26 ChEBI-20 retrieval order: contrastive fine-tuning from a
# molnl pretraining checkpoint, then the rank of every test description's molecule
# among all 33,010. Everything that is held equal across arms is in the script's
# SETTINGS; this file only says which checkpoint to start from and where to write.
#
#   sbatch --export=ALL,ARCH=bert|gpt2,ARM=<arm name>,INIT=<checkpoint>,STEP=<step>,\
# DATA=<chebi20.jsonl>,OUT_ROOT=<dir>[,MAX_STEPS=<n>] workflows/molnl-chebi20-retrieval.sbatch
#
# INIT is the checkpoint directory for BERT and checkpoint-<step>/training_state.bin
# for GPT-2; STEP is the step it must hold, checked on load. MAX_STEPS is for smoke
# tests and is recorded in the run's manifest. The output is OUT_ROOT/<ARM>[-smoke].
set -eo pipefail

: "${ARCH:?}" "${ARM:?}" "${INIT:?}" "${STEP:?}" "${DATA:?}" "${OUT_ROOT:?}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

EXTRA=()
OUT="${OUT_ROOT%/}/${ARM}"
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA=(--max-steps "${MAX_STEPS}")
  OUT="${OUT}-smoke"
fi
[ -e "${OUT}/metrics.json" ] && { echo "REFUSING: ${OUT} already holds a finished run" >&2; exit 1; }
[ -e "${INIT}" ] || { echo "no checkpoint at ${INIT}" >&2; exit 1; }

echo "=== chebi20 retrieval ${ARM} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)" \
     "dirty=$(git status --porcelain --untracked-files=no | wc -l)"
echo "init   : ${INIT} (step ${STEP})"
echo "output : ${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0

"${TORCHRUN}" --standalone --nproc_per_node=4 scripts/molnl_chebi20_retrieval.py \
  --arch "${ARCH}" --arm "${ARM}" --init "${INIT}" --expect-step "${STEP}" \
  --data "${DATA}" --out-dir "${OUT}" "${EXTRA[@]}"
echo "=== done @ $(date -Is) ==="
