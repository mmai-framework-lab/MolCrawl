#!/bin/bash
#SBATCH --job-name=mc-rna-bsweep-daily
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# One day's unattended check on the rna workstream. It reports and never acts.
#
#   WATCH_FILE=... sbatch --begin=2026-11-24T09:00 workflows/rna-bsweep-daily-check.sbatch
#
# Queued several days ahead with --begin rather than resubmitting itself, so a day
# that fails to run does not stop the days after it. Slurm holds a pending job for
# nothing, and the answer lands in a file that outlives whoever queued it.
#
# Env:
#   WATCH_FILE   the file to append to (default: beside this job's log)
#   BERT_ROOT    runs root the extension's BERT configs resolve against
#   GPT2_ROOT    runs root the extension's GPT-2 configs resolve against
#   GRID_ROOT    runs root the batch-size grid writes under
#   RECORDS      space-separated submission record TSVs to check outcomes against
#   HANDLED      TSV of job ids already dealt with: `job_id<TAB>why`. They are
#                counted and named separately instead of being listed as needing
#                action every day. The molnl grid lost three days to this: its
#                daily check said "60 to deal with" each morning and the two
#                genuine failures were sitting at the top of that list.
#
# Nothing here cancels, resubmits or deletes. It reads finished files, squeue and
# sacct. If something needs doing, it says so and a person does it.

set -u
WATCH_FILE="${WATCH_FILE:-workflows/slurm-logs/rna-daily.log}"
mkdir -p "$(dirname "${WATCH_FILE}")"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"

{
  echo "================ $(date -Is)  job ${SLURM_JOB_ID:-none}  $(git rev-parse --short HEAD) ================"

  echo "-- queue, by group --"
  for pat in 'rna-bert-.*-b[0-9]+' 'mc-rna-tput2' 'ext-(S|C)[0-9]+'; do
    r=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -cE "${pat}")
    p=$(squeue -u "$USER" -h -t PD -o "%j" 2>/dev/null | grep -cE "${pat}")
    printf '   %-26s running %3d  pending %4d\n' "${pat}" "${r}" "${p}"
  done

  echo "-- outcomes, from the submission records --"
  args=()
  for rec in ${RECORDS:-}; do [ -f "${rec}" ] && args+=(--record "${rec}"); done
  if [ "${#args[@]}" -gt 0 ] && [ -x "${PYTHON}" ]; then
    # The handled list goes to the checker, not through a filter here: filtering the
    # rows out left the count above them saying nine when one needed an answer.
    [ -n "${HANDLED:-}" ] && [ -f "${HANDLED}" ] && args+=(--handled "${HANDLED}")
    PYTHONPATH="$PWD" "${PYTHON}" scripts/check_job_outcomes.py "${args[@]}" 2>&1 \
      | grep -vE "^\s*$" | head -60
  else
    echo "   no record to check (set RECORDS)"
  fi

  echo "-- the GPT-2 re-measurement --"
  done_n=$(sacct -n -X -P -j "$(seq -s, 384856 384900)" --format=State 2>/dev/null \
           | grep -c COMPLETED)
  echo "   ${done_n} of 45 arms complete"
  for j in 385134 385135; do
    s=$(sacct -n -X -P -j "${j}" --format=JobName,State 2>/dev/null | head -1)
    echo "   ${j} ${s:-unknown}"
  done

  echo "-- the grid's arms with output on disk --"
  if [ -d "${GRID_ROOT:-}/rna/bert-output" ]; then
    n=$(find "${GRID_ROOT}/rna/bert-output" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "   ${n} arm directories"
    find "${GRID_ROOT}/rna/bert-output" -maxdepth 2 -name "trainer_state.json" \
      -printf '%h\n' 2>/dev/null | sed "s#${GRID_ROOT}/rna/bert-output/##" | sort \
      | head -8 | sed 's/^/     /'
  else
    echo "   no grid root yet (set GRID_ROOT)"
  fi

  echo "-- the 18-epoch extension --"
  if [ -n "${BERT_ROOT:-}" ] && [ -n "${GPT2_ROOT:-}" ] && [ -x "${PYTHON}" ]; then
    PYTHONPATH="$PWD" BERT_ROOT="${BERT_ROOT}" GPT2_ROOT="${GPT2_ROOT}" \
      "${PYTHON}" scripts/rna_extension_progress.py 2>&1 \
      | grep -E "^==|最小|エポックの最小" | head -40
  else
    echo "   not read (set BERT_ROOT and GPT2_ROOT)"
  fi
  echo
} >> "${WATCH_FILE}" 2>&1

echo "appended to ${WATCH_FILE}"
tail -5 "${WATCH_FILE}"
