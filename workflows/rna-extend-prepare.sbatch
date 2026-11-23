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
#     BERT_ROOT=... GPT2_ROOT=... SCHEDULER=rewrite sbatch workflows/rna-extend-prepare.sbatch
#   do the copy:
#     BERT_ROOT=... GPT2_ROOT=... SCHEDULER=rewrite STAGE=1 sbatch workflows/rna-extend-prepare.sbatch
#
# Env:
#   BERT_ROOT   required; the runs root the BERT configs resolve against.
#               get_bert_output_path appends rna/bert-output to it.
#   GPT2_ROOT   required; the runs root the GPT-2 configs resolve against. The
#               nine-epoch points were launched with it pointing at the gpt2-grid
#               subdirectory, so it is NOT the same value as BERT_ROOT.
#   SCHEDULER   required; remove | rewrite -- how the BERT copies hand scheduler.pt
#               over. scripts/probe_extend_resume_scheduler.py measures what each
#               one restores; read it before choosing. Measured 2026-11-18 (job
#               363879): remove loses the Adam moments, rewrite keeps them and
#               applies the config's rate.
#   STAGE       1 copies. Unset means dry run.
#
# The two roots are passed separately because the configs read MODEL_OUTPUT_ROOT at
# import time, so one process can only stage one framework.
#
# About 60 GB is copied: four BERT checkpoint-120960 trees and four GPT-2 ckpt.pt.
# No training is launched here, and nothing in the nine-epoch directories is written.
#
# One GPU because the partition requires one; this is file copying and a pytest run.

set -u
: "${BERT_ROOT:?export BERT_ROOT to the runs root the BERT configs resolve against}"
: "${GPT2_ROOT:?export GPT2_ROOT to the runs root the GPT-2 configs resolve against}"
: "${SCHEDULER:?export SCHEDULER=remove or SCHEDULER=rewrite}"
for d in "${BERT_ROOT}" "${GPT2_ROOT}"; do
  [ -d "${d}" ] || { echo "no such runs root: ${d}" >&2; exit 1; }
done

# The interpreter lives in the main checkout; a worktree has no miniconda.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

echo "=== checkout:       $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== BERT_ROOT:      ${BERT_ROOT} ==="
echo "=== GPT2_ROOT:      ${GPT2_ROOT} ==="
echo "=== scheduler mode: ${SCHEDULER} ==="
echo "=== stage:          ${STAGE:-0 (dry run)} ==="

echo
echo "=== the configs say what the judgment asked for ==="
MODEL_OUTPUT_ROOT="${BERT_ROOT}" PYTHONPATH="$PWD" "${PYTHON}" -m pytest \
    tests/unit/test_rna_extension_configs.py -q -p no:cacheprovider || exit 1

ARGS=(--scheduler "${SCHEDULER}")
if [ "${STAGE:-0}" = "1" ]; then ARGS+=(--verify); else ARGS+=(--dry-run); fi

rc=0
for side in bert gpt2; do
  case "${side}" in
    bert) ROOT="${BERT_ROOT}" ;;
    gpt2) ROOT="${GPT2_ROOT}" ;;
  esac
  echo
  echo "=== staging ${side} under ${ROOT} ==="
  MODEL_OUTPUT_ROOT="${ROOT}" PYTHONPATH="$PWD" "${PYTHON}" \
      scripts/prepare_rna_extension.py --framework "${side}" "${ARGS[@]}" || rc=$?
done
echo
echo "=== prepare rc=${rc} ==="
exit ${rc}
