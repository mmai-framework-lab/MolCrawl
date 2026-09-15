#!/bin/bash
#SBATCH --job-name=mc-audit-legacy-ckpt
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive 2026-09-15 §6.3: which ckpt.pt files hold the best weights and which
# hold merely the last ones. See scripts/audit_legacy_ckpt.py for how the question
# is decided; this only points it at trees and records where the answer came from.
#
# No GPU work is done -- the script reads a few KB of pickle per checkpoint and
# never materialises a tensor. The allocation is here because this cluster has one
# partition and nothing heavier than a shell runs on the login node.
#
# Submit with the trees to walk, colon-separated:
#
#   sbatch --export=ALL,AUDIT_ROOTS="/path/to/learning_source_a:/path/to/learning_source_b" \
#          workflows/audit-legacy-ckpt.sbatch
#
# The result is the job log. Copy the table into the report rather than leaving it
# in slurm-logs, which is not a durable record.
set -eo pipefail

: "${AUDIT_ROOTS:?set AUDIT_ROOTS to the learning_source trees to walk, colon-separated}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# A worktree holds only tracked files, so the interpreter is not inside it. Resolve
# it through git's common dir rather than writing a path in.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# Fail on a mistyped root rather than reporting "0 checkpoints examined", which
# reads like a clean result.
IFS=':' read -r -a _roots <<< "${AUDIT_ROOTS}"
for _r in "${_roots[@]}"; do
  [ -d "${_r}" ] || { echo "AUDIT_ROOTS entry is not a directory: ${_r}" >&2; exit 1; }
done

echo "=== legacy ckpt.pt audit ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "roots: ${AUDIT_ROOTS}"
echo

"${PYTHON}" scripts/audit_legacy_ckpt.py "${_roots[@]}" && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0, and a failure
# would otherwise leave nothing at the end of the log.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
