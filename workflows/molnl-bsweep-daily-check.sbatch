#!/bin/bash
#SBATCH --job-name=mc-molnl-bsweep-check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# One day's unattended check on the 2026-11-20 molnl batch grid.
#
# It reports and does not act. Nothing here submits a training job: an unattended
# resubmission would start GPU work nobody asked for, and the thing that went wrong
# today -- a launcher missing one environment variable -- would have been resubmitted
# 112 times by a script that could not tell it was the same failure every time.
#
#   RECORD=<live submission record> OUT=<append-only log> \
#     sbatch --begin=now+1day --export=ALL,RECORD=...,OUT=... \
#            workflows/molnl-bsweep-daily-check.sbatch
#
# Several of these are submitted at once with different --begin times rather than each
# one submitting the next: a chain that re-submits itself keeps going when nobody wants
# it to and stops for good if one link fails to queue.
#
# No GPU is asked for. It reads Slurm and the logs, which is not computation, but it
# runs here rather than on a login node because that is the rule for this cluster.
set -uo pipefail
: "${RECORD:?set RECORD to the submission record to read the job ids from}"
: "${OUT:?set OUT to the file to append the snapshot to}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
cd "${REPO}"
mkdir -p "$(dirname "${OUT}")" workflows/slurm-logs

OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-}"
{
  echo
  echo "## $(date -Is)  (job ${SLURM_JOB_ID:-local})"
  echo
  echo "queue:"
  squeue -u "${USER}" -h -o '%j %T' 2>/dev/null \
    | awk '$1 ~ /^mc-molnl-(bert|gpt2)-b/ {c[$2]++} END{for(s in c) printf "  %-10s %d\n", s, c[s]; if (!length(c)) print "  (none left)"}'
  echo
  echo "arms with output on disk:"
  for arch in bert gpt2; do
    n=0
    if [ -n "${OUTPUT_ROOT}" ]; then
      n=$(find "${OUTPUT_ROOT}/molecule_nat_lang/${arch}-output" -maxdepth 1 -type d \
            -name "*-${arch}-b*-lr*" 2>/dev/null | wc -l)
    fi
    printf "  %-5s %s\n" "${arch}" "${n}"
  done
  echo
  echo "outcomes:"
  "${PYTHON}" -I scripts/check_job_outcomes.py --record "${RECORD}" \
      --logs workflows/slurm-logs 2>&1 | sed 's/^/  /'
} >> "${OUT}"

echo "=== appended one snapshot to ${OUT} @ $(date -Is) ==="
tail -40 "${OUT}"
