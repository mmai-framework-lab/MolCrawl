#!/bin/bash
# Submit a job, record its id, and chain a failure recorder onto it.
#
#   SUBMIT_RECORD=<file> workflows/sbatch-watched.sh [label] -- <sbatch args...>
#
# Prints the job id on stdout like `sbatch --parsable`, so it drops into a submitter
# loop in place of sbatch. Two things happen besides the submission:
#
#   the id and a label are appended to SUBMIT_RECORD, so a later session can ask what
#       was submitted without having to remember
#   a job with --dependency=afternotok is queued behind it, so Slurm records a bad
#       ending whether or not anyone is watching
#
# Both exist because four compounds BERT runs died on a Friday evening and were not
# noticed until Monday: 56 hours in which nothing of that workstream ran. The only
# watcher was a background shell inside the session that submitted them, and the only
# record of what had been submitted was in that session too.
#
# The failure recorder is itself submitted with --parsable and its id is recorded, so
# a recorder that fails to queue is visible rather than silently absent.
set -uo pipefail
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
LABEL="${1:-}"
if [ "${1:-}" = "--" ]; then LABEL=""; shift; else [ $# -gt 0 ] && shift; fi
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no sbatch arguments given" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$(dirname "${SUBMIT_RECORD}")"

JID=$(sbatch --parsable "$@") || { echo "submission failed" >&2; exit 1; }
printf '%s\t%s\t%s\n' "${JID}" "${LABEL:-unlabelled}" "$(date -Is)" >> "${SUBMIT_RECORD}"

WATCH_LOG="${WATCH_LOG:-$(dirname "${SUBMIT_RECORD}")/job-failures.tsv}"
WID=$(sbatch --parsable --dependency=afternotok:"${JID}" \
      --export=ALL,WATCH_JOB="${JID}",WATCH_LOG="${WATCH_LOG}",WATCH_NAME="${LABEL}" \
      "${HERE}/workflows/on-job-failure.sbatch" 2>/dev/null) || WID=""
if [ -n "${WID}" ]; then
  printf '%s\t%s\t%s\n' "${WID}" "watcher-for-${JID}" "$(date -Is)" >> "${SUBMIT_RECORD}"
else
  echo "WARNING: no failure recorder queued for ${JID}" >&2
fi
echo "${JID}"
