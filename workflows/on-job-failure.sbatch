#!/bin/bash
#SBATCH --job-name=mc-on-failure
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Record that a watched job ended badly, in a place that outlives the session.
#
#   sbatch --dependency=afternotok:<jobid> \
#          --export=ALL,WATCH_JOB=<jobid>,WATCH_LOG=<file>[,WATCH_NAME=<label>] \
#          workflows/on-job-failure.sbatch
#
# Slurm fires this, not a process of ours, so it survives a session ending, a
# background waiter being killed at its time limit, and nobody being logged in. Four
# compounds BERT runs died on a Friday evening and were not noticed until Monday
# morning -- 56 hours in which nothing of that workstream ran -- because the only
# watcher was a background shell inside one session.
#
# afternotok fires on TIMEOUT as well, which is the ordinary end of a segment in a
# chained run, so the state is read back and a TIMEOUT is recorded as expected rather
# than as a failure. What is worth waking up for is FAILED, NODE_FAIL, CANCELLED and
# OUT_OF_MEMORY.
set -uo pipefail
: "${WATCH_JOB:?export WATCH_JOB to the job id being watched}"
: "${WATCH_LOG:?export WATCH_LOG to the file to append to}"
mkdir -p "$(dirname "${WATCH_LOG}")"

state=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  state=$(sacct -X -n -j "${WATCH_JOB}" -o State%30 2>/dev/null | head -1 | awk '{print $1}')
  case "${state}" in
    ""|RUNNING|PENDING|COMPLETING|REQUEUED|RESIZING|SUSPENDED) sleep 20 ;;
    *) break ;;
  esac
done
state="${state:-UNKNOWN}"
name="${WATCH_NAME:-$(sacct -X -n -j "${WATCH_JOB}" -o JobName%60 2>/dev/null | head -1 | tr -d ' ')}"
elapsed=$(sacct -X -n -j "${WATCH_JOB}" -o Elapsed 2>/dev/null | head -1 | tr -d ' ')
stdout=$(scontrol show job "${WATCH_JOB}" 2>/dev/null | grep -oP 'StdOut=\K\S+' | head -1)

case "${state}" in
  TIMEOUT)  severity="expected" ;;
  COMPLETED) severity="ok" ;;
  *)        severity="ATTENTION" ;;
esac

# One tab-separated line per event, appended, so a later session greps it rather than
# trusting that someone remembered.
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -Is)" "${severity}" "${WATCH_JOB}" "${name}" "${state}" "${elapsed}" \
  >> "${WATCH_LOG}"

if [ "${severity}" = "ATTENTION" ] && [ -n "${stdout}" ] && [ -r "${stdout}" ]; then
  # The lines worth reading first, not the whole log.
  {
    printf '# %s  job %s  %s  %s\n' "$(date -Is)" "${WATCH_JOB}" "${name}" "${state}"
    grep -nE "Error|error|Traceback|abort|Watchdog|CUDA|out of memory|REFUSING" "${stdout}" 2>/dev/null \
      | grep -viE "WARNING|FutureWarning" | tail -12
    printf '\n'
  } >> "${WATCH_LOG%.tsv}-detail.txt"
fi
exit 0
