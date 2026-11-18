#!/bin/bash
#SBATCH --job-name=mc-rna-extend-prepare
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Stage the rna 2026-11-18 extension, on a compute node.
#
#   check the configs and print the plan (writes nothing):
#     MODEL_OUTPUT_ROOT=... SCHEDULER=rewrite sbatch workflows/rna-extend-prepare.sbatch
#   do the copy:
#     MODEL_OUTPUT_ROOT=... SCHEDULER=rewrite STAGE=1 sbatch workflows/rna-extend-prepare.sbatch
#
# Env:
#   MODEL_OUTPUT_ROOT  required; every path comes from the configs, which read this
#   SCHEDULER          required; remove | rewrite -- how the BERT copies hand
#                      scheduler.pt over. scripts/probe_extend_resume_scheduler.py
#                      measures what each one restores; read it before choosing.
#   STAGE              1 copies. Unset means dry run.
#
# About 60 GB is copied: four BERT checkpoint-120960 trees and four GPT-2 ckpt.pt.
# No training is launched here, and nothing in the nine-epoch directories is written.
#
# One GPU because the partition requires one; this is file copying and a pytest run.

set -u
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the rna runs root}"
: "${SCHEDULER:?export SCHEDULER=remove or SCHEDULER=rewrite}"
[ -d "${MODEL_OUTPUT_ROOT}" ] || { echo "MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT} does not exist" >&2; exit 1; }

# The interpreter lives in the main checkout; a worktree has no miniconda.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

export MODEL_OUTPUT_ROOT
echo "=== checkout:          $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== MODEL_OUTPUT_ROOT: ${MODEL_OUTPUT_ROOT} ==="
echo "=== scheduler mode:    ${SCHEDULER} ==="
echo "=== stage:             ${STAGE:-0 (dry run)} ==="

echo
echo "=== the configs say what the judgment asked for ==="
PYTHONPATH="$PWD" "${PYTHON}" -m pytest tests/unit/test_rna_extension_configs.py \
    -q -p no:cacheprovider || exit 1

echo
echo "=== staging ==="
ARGS=(--scheduler "${SCHEDULER}")
if [ "${STAGE:-0}" = "1" ]; then ARGS+=(--verify); else ARGS+=(--dry-run); fi
PYTHONPATH="$PWD" "${PYTHON}" scripts/prepare_rna_extension.py "${ARGS[@]}"
rc=$?
echo "=== prepare rc=${rc} ==="
exit ${rc}
