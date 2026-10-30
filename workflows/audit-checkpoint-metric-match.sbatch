#!/bin/bash
#SBATCH --job-name=mc-ckpt-metric
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive all-checkpoint-metric-order-2026-10-28-v2 §3/§4/§6: for every BERT run,
# does best_model_checkpoint point at the step where eval_loss_mask bottomed out.
# The 2026-10-30 judgment §1 adds MISMATCHES: the production BERT runs that do not
# match, on their own, in the same columns.
#
# Reads trainer_state.json and run_manifest.json. No GPU work, no weight file is
# opened and nothing is written inside a run directory. The allocation is here
# because this cluster has one partition and nothing heavier than a shell runs on
# the login node.
#
# AUDIT_ROOTS is a whitespace-separated list of MODALITY=DIR, repeated per tree a
# modality has. Required and unset-fatal, because the run trees are gitignored and
# differ per checkout.
#
#   sbatch --export=ALL,AUDIT_ROOTS="compounds=<dir> genome_sequence=<dir> ..." \
#          workflows/audit-checkpoint-metric-match.sbatch
set -eo pipefail

: "${AUDIT_ROOTS:?set AUDIT_ROOTS to a whitespace-separated list of MODALITY=DIR}"

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
for pair in ${AUDIT_ROOTS}; do
  case "${pair}" in
    *=*) ;;
    *) echo "AUDIT_ROOTS wants MODALITY=DIR, got '${pair}'" >&2; exit 1 ;;
  esac
  dir="${pair#*=}"
  [ -d "${dir}" ] || { echo "not a directory: ${dir}" >&2; exit 1; }
  ROOT_ARGS+=(--root "${pair}")
done

OUT_DIR="${REPORT_DIR:-${PYROOT}/tmp}"
REPORT="${REPORT:-${OUT_DIR}/checkpoint-metric-match-${SLURM_JOB_ID}.json}"
TSV="${TSV:-${OUT_DIR}/checkpoint-metric-match-${SLURM_JOB_ID}.tsv}"
CONDITIONS="${CONDITIONS:-${OUT_DIR}/run-conditions-${SLURM_JOB_ID}.tsv}"
MISMATCHES="${MISMATCHES:-${OUT_DIR}/checkpoint-metric-mismatches-${SLURM_JOB_ID}.tsv}"
mkdir -p "${OUT_DIR}"

echo "=== checkpoint metric match ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "roots : ${AUDIT_ROOTS}"
echo "report: ${REPORT}"
echo "tsv   : ${TSV}"
echo "cond  : ${CONDITIONS}"
echo "mism  : ${MISMATCHES}"
echo

"${PYTHON}" scripts/audit_checkpoint_metric_match.py \
  "${ROOT_ARGS[@]}" --report "${REPORT}" --tsv "${TSV}" \
  --tsv-conditions "${CONDITIONS}" --tsv-mismatches "${MISMATCHES}" \
  && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
