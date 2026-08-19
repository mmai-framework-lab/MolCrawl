#!/bin/bash
# RNA GPT-2 pretraining launcher (rikyu / ai.r-ccs).
#
#   SIZE=small sbatch workflows/rna-gpt2.sbatch                      # production
#   SIZE=small SMOKE_MAX_ITERS=200 sbatch workflows/rna-gpt2.sbatch  # smoke
#
# Env knobs:
#   SIZE            required: small | medium | large | xl
#   TAG             output-dir suffix for smoke runs (default: smoke)
#   SMOKE_MAX_ITERS if set, runs a short scratch job into runs/smoke-* instead
#                   of the production out_dir (see §5 of the instruction: the
#                   configs are init_from="resume", so a smoke sharing the
#                   production out_dir would be picked up by the real run)
#
# Production runs deliberately keep init_from="resume" from the config so that
# resubmitting the same job continues from the last checkpoint (resume chain for
# runs that exceed the 4-day walltime).
#SBATCH --account=rkp00024
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --job-name=rna-gpt2
#SBATCH --output=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/%x-%j.out
#SBATCH --error=/data1/rkp00024/matsubara/MolCrawl/workflows/slurm-logs/%x-%j.out
set -e

# Code and data are separate: REPO is the checkout to run (a worktree when
# testing a branch), MAIN holds the conda env, the learning source and the
# log directory. Pinning both to the same path silently ran whatever branch
# the shared checkout happened to be on.
MAIN="/data1/rkp00024/matsubara/MolCrawl"
REPO="${REPO:-$MAIN}"
SIZE="${SIZE:?SIZE is required (small|medium|large|xl)}"
TAG="${TAG:-smoke}"
CONFIG="molcrawl/tasks/pretrain/configs/rna/gpt2_${SIZE}.py"

# Training reads training_ready through a symlink in this tree; logs and
# checkpoints are the only things written here, so the b200prep original stays
# untouched.
export LEARNING_SOURCE_DIR="${MAIN}/learning_source_20260803_rna_train"

# Keep caches and temporaries off $HOME (quota) — they all go under /data1.
export HF_HOME="${MAIN}/tmp/hf-rna"
export HF_DATASETS_CACHE="${MAIN}/tmp/hf-rna/datasets"
export XDG_CACHE_HOME="${MAIN}/tmp/hf-rna/xdg"
export TMPDIR="${MAIN}/tmp/hf-rna/tmp"
export LD_LIBRARY_PATH="${MAIN}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1

cd "${REPO}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
mkdir -p "${MAIN}/workflows/slurm-logs" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

# --compile=False: no Triton on these nodes. Everything else comes from the
# config; the only other overrides are output paths and the smoke schedule.
OVERRIDES=(--compile=False --early_stopping=False)
if [ -n "${SMOKE_MAX_ITERS:-}" ]; then
  OUT="${LEARNING_SOURCE_DIR}/rna/runs/${TAG}-gpt2-${SIZE}"
  WARM=$(( SMOKE_MAX_ITERS / 10 )); [ "${WARM}" -lt 1 ] && WARM=1
  OVERRIDES+=(--init_from=scratch --out_dir="${OUT}" --tensorboard_dir="${OUT}"
              --max_iters="${SMOKE_MAX_ITERS}" --lr_decay_iters="${SMOKE_MAX_ITERS}"
              --warmup_iters="${WARM}" --eval_interval=50)
else
  OUT="$("${MAIN}"/miniconda/bin/python -c "
from molcrawl.core.paths import get_gpt2_output_path
print(get_gpt2_output_path('rna', '${SIZE}'))")"
fi
mkdir -p "${OUT}"

echo "=== rna gpt2-${SIZE} ${SMOKE_MAX_ITERS:+SMOKE max_iters=${SMOKE_MAX_ITERS}} @ $(date -Is) on ${SLURMD_NODENAME} ==="
echo "repo:   ${REPO}"
echo "branch: $(git -C "${REPO}" rev-parse --abbrev-ref HEAD)  head: $(git -C "${REPO}" rev-parse --short HEAD)"
"${MAIN}"/miniconda/bin/python -c "import molcrawl; print('molcrawl from:', molcrawl.__file__)"
echo "config: ${CONFIG}"
echo "data:   ${LEARNING_SOURCE_DIR}/rna/training_ready_hf_dataset"
echo "out:    ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Sanity anchors to grep for in this log (see §5 of the instruction):
#   "number of parameters: 104.48M"      gpt2-small at vocab 25,426 (131.54M = vocab fix missing)
#   "tokens per iteration will be: 2,621,440"   = 2560 x 1024, invariant to GPU count
rc=0
"${MAIN}"/miniconda/bin/torchrun --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CONFIG}" "${OVERRIDES[@]}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
ls -d "${OUT}"/checkpoint-* 2>/dev/null | tail -3
exit $rc
