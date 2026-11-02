#!/bin/bash
#SBATCH --job-name=mc-ladder-crosscheck
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive all-ladder-check-order-2026-11-02b §2-§5/§7: the ladder's rungs beside the
# checkpoint-selection error. Reads the audit JSON, each rung's run_manifest.json and,
# for GPT-2, each checkpoint's training_args.json. No weight file is opened and nothing
# is written inside a run directory. The allocation is here because this cluster has one
# partition and nothing heavier than a shell runs on the login node.
#
#   sbatch --export=ALL,AUDIT_JSON=<path>,MISMATCHES=<path>,OUT=<path> \
#          workflows/ladder-checkpoint-crosscheck.sbatch
set -eo pipefail

: "${AUDIT_JSON:?set AUDIT_JSON to the audit_checkpoint_metric_match.py JSON}"
[ -f "${AUDIT_JSON}" ] || { echo "no such file: ${AUDIT_JSON}" >&2; exit 1; }

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

LADDER="${LADDER:-scripts/ladder_runs.tsv}"
OUT="${OUT:-${PYROOT}/tmp/ladder-checkpoint-crosscheck-${SLURM_JOB_ID}.tsv}"
mkdir -p "$(dirname "${OUT}")"

echo "=== ladder crosscheck ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "audit : ${AUDIT_JSON}"
echo "ladder: ${LADDER}"
echo "=== outputs: ${OUT} ==="
echo

"${PYTHON}" scripts/ladder_checkpoint_crosscheck.py \
  --audit-json "${AUDIT_JSON}" --ladder "${LADDER}" --out "${OUT}" \
  ${MISMATCHES:+--mismatches "${MISMATCHES}"} \
  && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
