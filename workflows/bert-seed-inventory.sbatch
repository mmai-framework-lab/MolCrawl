#!/bin/bash
#SBATCH --job-name=mc-bert-seed-inv
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive all-seed-and-metric-order-2026-11-04b §3: one row per BERT run, the seed
# the config named beside the seed run_manifest.json records.
#
# Reads run_manifest.json, and training_args.bin only for the runs that have no
# manifest. No weight file is opened and nothing is written inside a run directory.
# The allocation is here because this cluster has one partition and nothing heavier
# than a shell runs on the login node.
#
# SEED_ROOTS is a whitespace-separated list of MODALITY=DIR, repeated per tree a
# modality has. Required and unset-fatal, because the run trees are gitignored and
# differ per checkout.
#
#   sbatch --export=ALL,SEED_ROOTS="compounds=<dir> rna=<dir> ...",OUT=<path> \
#          workflows/bert-seed-inventory.sbatch
set -eo pipefail

: "${SEED_ROOTS:?set SEED_ROOTS to a whitespace-separated list of MODALITY=DIR}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

ROOT_ARGS=()
for pair in ${SEED_ROOTS}; do
  case "${pair}" in
    *=*) ;;
    *) echo "SEED_ROOTS wants MODALITY=DIR, got '${pair}'" >&2; exit 1 ;;
  esac
  dir="${pair#*=}"
  [ -d "${dir}" ] || { echo "not a directory: ${dir}" >&2; exit 1; }
  ROOT_ARGS+=(--root "${pair}")
done

OUT="${OUT:-${PYROOT}/tmp/rt1-bert-seed-inventory-${SLURM_JOB_ID}.tsv}"
mkdir -p "$(dirname "${OUT}")"

echo "=== bert seed inventory ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "roots : ${SEED_ROOTS}"
echo "=== outputs: ${OUT} ==="
echo

"${PYTHON}" scripts/bert_seed_inventory.py "${ROOT_ARGS[@]}" --out "${OUT}" \
  && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
