#!/bin/bash
# Does this size fit the micro-batch the grid asks for?
#
# The large and xl configs lower the micro-batch deliberately -- "Micro-batch is
# lowered at this size to stay memory-safe: 8 * 320 = 2560" and "4 * 640 = 2560".
# Raising both to 16 x 160 keeps the effective batch at 2,560 and changes only
# how much is resident at once, so it is an execution choice, not a research one
# -- but it is the kind of execution choice that turns into an OOM four days into
# a holiday. Three steps is enough to find out.
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=0:40:00
#SBATCH --job-name=rna-gpt2-shape-probe
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -uo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the RNA learning_source root}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${PROBE_OUT:?set PROBE_OUT to a scratch directory outside any input tree}"
MB="${MB:-16}"
GA="${GA:-160}"
STEPS="${STEPS:-3}"

if [ -x "./miniconda/bin/torchrun" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda" >&2; exit 1; }

export LEARNING_SOURCE_DIR PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HOME="${PROBE_OUT}/hf" XDG_CACHE_HOME="${PROBE_OUT}/xdg" TMPDIR="${PROBE_OUT}/tmp"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
OUT="${PROBE_OUT}/shape-${SIZE}-${MB}x${GA}"
mkdir -p "${OUT}" "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" workflows/slurm-logs
echo "=== shape probe ${SIZE} MB=${MB} GA=${GA} -> $((MB * GA)) @ $(date -Is) on ${SLURMD_NODENAME} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py \
  "molcrawl/tasks/pretrain/configs/rna/gpt2_${SIZE}.py" \
  --compile=False --early_stopping=False --init_from=scratch \
  --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
  --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
  --max_iters="${STEPS}" --lr_decay_iters="${STEPS}" --warmup_iters=1 \
  --eval_interval="$((STEPS + 10))" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
if [ "${rc}" -ne 0 ]; then
  echo "VERDICT ${SIZE} ${MB}x${GA}: FAILED"
else
  echo "VERDICT ${SIZE} ${MB}x${GA}: OK"
fi
rm -rf "${OUT}"
exit 0
