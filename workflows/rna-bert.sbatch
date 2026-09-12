#!/bin/bash
# RNA BERT pretraining launcher (rikyu / ai.r-ccs).
#
#   SIZE=small sbatch workflows/rna-bert.sbatch                      # production
#   SIZE=small SMOKE_MAX_STEPS=200 sbatch workflows/rna-bert.sbatch  # smoke
#
# Env knobs:
#   SIZE            required: small | medium | large
#   TAG             output-dir suffix for smoke runs (default: smoke)
#   SMOKE_MAX_STEPS if set, runs a short job into runs/smoke-* instead of the
#                   production model_path
#
# --gpus=4 is NOT a free choice here. HF Trainer's effective batch is
# per_device (8) x grad_accum (80) x world_size, so the 2560 spec only holds at
# 4 GPUs; changing the GPU count silently changes the amount of data the model
# sees. GPT-2 has no such coupling (nanoGPT divides grad_accum by world_size).
#SBATCH --account=rkp00024
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --job-name=rna-bert
# Relative so nothing here names a machine. sbatch resolves it against the
# submitting directory, so submit production runs from the main checkout --
# from a worktree the log lands in a tree meant to be disposable. The run
# echoes MAIN and REPO below, so the log says which it was either way.
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e

# Code and data are separate: REPO is the checkout to run (a worktree when
# testing a branch), MAIN holds the conda env, the learning source and the
# log directory. Pinning both to the same path silently ran whatever branch
# the shared checkout happened to be on.
# MAIN holds the conda env, the learning source and the log directory; REPO is
# the checkout to run. A worktree has no miniconda, so MAIN comes from git's
# common dir rather than from a path written in here.
MAIN="${MAIN:-$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)}"
[ -x "${MAIN}/miniconda/bin/python" ] || {
    echo "no miniconda under MAIN=${MAIN} -- export MAIN to the main checkout" >&2; exit 1; }
REPO="${REPO:-$MAIN}"
SIZE="${SIZE:?SIZE is required (small|medium|large)}"
TAG="${TAG:-smoke}"
CONFIG="molcrawl/tasks/pretrain/configs/rna/bert_${SIZE}.py"

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

# Outputs go to a stated root, never under LEARNING_SOURCE_DIR. core.paths
# derives model_path from LEARNING_SOURCE_DIR, which puts checkpoints inside the
# tree the run reads; output_guard (core/output_guard.py, merged 2026-09-01)
# refuses that at startup. The root lives in the main checkout, not in a
# worktree -- learning_source* is gitignored, so a worktree keeps its own copy
# and a production run launched from one would leave its record somewhere built
# to be deleted.
RUNS_ROOT="${RUNS_ROOT:-${MAIN}/learning_source_rna_runs}"
OVERRIDES=()
if [ -n "${SMOKE_MAX_STEPS:-}" ]; then
  OUT="${RUNS_ROOT}/${TAG}-bert-${SIZE}"
  OVERRIDES+=(--max_steps="${SMOKE_MAX_STEPS}" --save_steps=100)
else
  OUT="${RUNS_ROOT}/bert-${SIZE}"
fi
OVERRIDES+=(--model_path="${OUT}")
mkdir -p "${OUT}"
echo "=== outputs: ${OUT} (RUNS_ROOT=${RUNS_ROOT}) ==="

echo "=== rna bert-${SIZE} ${SMOKE_MAX_STEPS:+SMOKE max_steps=${SMOKE_MAX_STEPS}} @ $(date -Is) on ${SLURMD_NODENAME} ==="
# A run can outlive the checkout it started from: production takes days, resume
# resubmits later, and REPO is a worktree someone may switch in between. Pass
# EXPECT_COMMIT to refuse to start on anything else rather than silently
# continuing a run on different code.
if [ -n "${EXPECT_COMMIT:-}" ]; then
  ACTUAL=$(git -C "${REPO}" rev-parse --short HEAD)
  if [ "${ACTUAL}" != "${EXPECT_COMMIT}" ]; then
    echo "FATAL: ${REPO} is at ${ACTUAL}, expected ${EXPECT_COMMIT}" >&2
    exit 1
  fi
  echo "commit guard: ${ACTUAL} == ${EXPECT_COMMIT}"
fi
echo "repo:   ${REPO}"
echo "branch: $(git -C "${REPO}" rev-parse --abbrev-ref HEAD)  head: $(git -C "${REPO}" rev-parse --short HEAD)"
"${MAIN}"/miniconda/bin/python -c "import molcrawl; print('molcrawl from:', molcrawl.__file__)"
echo "config: ${CONFIG}"
echo "data:   ${LEARNING_SOURCE_DIR}/rna/training_ready_hf_dataset"
echo "out:    ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Sanity anchors to grep for in this log (see §5 of the instruction):
#   vocab_size 25,432   (= (25,426 + [UNK] + [PAD]) rounded up to a multiple of 8)
#   eval runs on the valid split — test stays held out (PR #98)
#   "effective global batch 2560 matches the declared expected_global_batch"
#     absent => the run got a batch other than 2,560 and is not this ladder
rc=0
"${MAIN}"/miniconda/bin/torchrun --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CONFIG}" "${OVERRIDES[@]}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
ls -d "${OUT}"/checkpoint-* 2>/dev/null | tail -3
exit $rc
