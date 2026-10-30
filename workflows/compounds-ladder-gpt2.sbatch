#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# compounds GPT-2 ladder on the v4 packed data (10 epochs, global batch 2,560).
# Submit with:  sbatch --job-name=mc-cmp-gpt2-<size> --time=<hh:mm:ss> \
#                      --export=ALL,SIZE=<size> workflows/compounds-ladder-gpt2.sbatch
# out_dir / init_from come from the config (scratch, LEARNING_SOURCE_DIR-derived).
# tensorboard is disabled: the package is not installed in this env.
set -e
: "${SIZE:?SIZE must be set (small|medium|large|xl)}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$PWD"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the packed compounds learning_source root (learning_source_20260805_compounds_packed in the main checkout)}"
export LEARNING_SOURCE_DIR
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"
echo "=== compounds gpt2-${SIZE} (v4 packed) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "data: ${LEARNING_SOURCE_DIR}"
"${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "molcrawl/tasks/pretrain/configs/compounds/gpt2_${SIZE}.py" \
  --compile=False --tensorboard=False
rc=$?
echo "=== training rc=${rc} @ $(date -Is) ==="
OUT="${LEARNING_SOURCE_DIR}/compounds/gpt2-output/compounds-${SIZE}"
if [ -f "${OUT}/ckpt.pt" ]; then
  echo "best-ckpt check:"; ls -la "${OUT}/ckpt.pt"
  "${PYROOT}/miniconda/bin/python" -c "import torch;c=torch.load('${OUT}/ckpt.pt',map_location='cpu',weights_only=False);print('  best_val_loss=',c.get('best_val_loss'),' iter=',c.get('iter_num'))" 2>&1 | tail -1
else
  echo "!! ckpt.pt MISSING at ${OUT}"
fi
exit $rc
