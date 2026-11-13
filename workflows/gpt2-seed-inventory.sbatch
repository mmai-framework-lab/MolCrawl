#!/bin/bash
#SBATCH --job-name=mc-gpt2-seed-inv
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Judgment all-seed-judgment-2026-11-10 §5: one row per GPT-2 run, the seed its
# config declared beside the value that reached torch.manual_seed.
#
# Reads run_manifest.json and the job logs. No GPU work, no weight file is opened
# and nothing is written inside a run directory. The allocation is here because
# this cluster has one partition and nothing heavier than a shell runs on the
# login node.
#
# SEED_ROOTS is a whitespace-separated list of MODALITY=DIR, repeated per tree a
# modality has. LOG_ROOTS is a whitespace-separated list of directories holding
# slurm logs; the configurator's echo of the config lives there and nowhere else,
# so a missing log root shows up as an empty config_seed rather than a wrong one.
# Both are required and unset-fatal, because the trees are gitignored and differ
# per checkout.
#
#   sbatch --export=ALL workflows/gpt2-seed-inventory.sbatch
#     with SEED_ROOTS, LOG_ROOTS and OUT exported beforehand
set -eo pipefail

: "${SEED_ROOTS:?set SEED_ROOTS to a whitespace-separated list of MODALITY=DIR}"
: "${LOG_ROOTS:?set LOG_ROOTS to a whitespace-separated list of slurm log directories}"

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

LOG_ARGS=()
for dir in ${LOG_ROOTS}; do
  LOG_ARGS+=(--log-root "${dir}")
done

OUT="${OUT:-${PYROOT}/tmp/rt1-gpt2-seed-inventory-${SLURM_JOB_ID}.tsv}"
mkdir -p "$(dirname "${OUT}")"

echo "=== gpt2 seed inventory ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "roots : ${SEED_ROOTS}"
echo "logs  : ${LOG_ROOTS}"
echo "=== outputs: ${OUT} ==="
echo

"${PYTHON}" scripts/gpt2_seed_inventory.py "${ROOT_ARGS[@]}" "${LOG_ARGS[@]}" --out "${OUT}" \
  && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
