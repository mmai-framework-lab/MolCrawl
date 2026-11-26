#!/bin/bash
# Submit the eight runs of the 2026-11-26 ChEBI-20 retrieval order and the job that
# writes the attachment tables once they end.
#
#   SUBMIT_RECORD=<file> ARMS=<rt1-molnl-chebi20-arms-*.tsv> RUNS=<.../molecule_nat_lang> \
#   DATA=<chebi20.jsonl> OUT_ROOT=<dir> REPORT_DIR=<dir> OVERLAP_ITEMS=<tsv> STAMP=YYYYMMDD \
#       workflows/molnl-chebi20-submit.sh
#
# The checkpoint and its step come from the arms table, which scripts/molnl_chebi20_arms.py
# wrote from the runs themselves; nothing is typed here. The table job is queued with
# afterany so it runs whether or not anyone is awake; it refuses a run that did not
# finish, so a gap shows as a failed job rather than as a short table.
set -euo pipefail
: "${SUBMIT_RECORD:?}" "${ARMS:?}" "${RUNS:?}" "${DATA:?}" "${OUT_ROOT:?}" \
  "${REPORT_DIR:?}" "${OVERLAP_ITEMS:?}" "${STAMP:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
mkdir -p "${OUT_ROOT}"

jids=()
while IFS=$'\t' read -r arm arch size _batch _lr _seed step _rest; do
  case "${arm}" in arm|'#'*|'') continue;; esac
  ckpt="$(awk -F'\t' -v a="${arm}" '$1==a{print $14}' "${ARMS}")"
  init="${RUNS%/}/${ckpt}"
  [ -e "${init}" ] || { echo "no checkpoint at ${init}" >&2; exit 1; }
  jid=$(workflows/sbatch-watched.sh "chebi20-${arch}-${size}" -- \
        --job-name="mc-chebi20-${arch}-${size}" \
        --export=ALL,ARCH="${arch}",ARM="${arm}",INIT="${init}",STEP="${step}",DATA="${DATA}",OUT_ROOT="${OUT_ROOT}" \
        workflows/molnl-chebi20-retrieval.sbatch)
  echo "${arm}: ${jid}" >&2
  jids+=("${jid}")
done < <(grep -v '^#' "${ARMS}")
[ "${#jids[@]}" -eq 8 ] || { echo "submitted ${#jids[@]} runs, expected 8" >&2; exit 1; }

dep="afterany:$(IFS=:; echo "${jids[*]}")"
PY="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/miniconda/bin/python"
workflows/sbatch-watched.sh "chebi20-report" -- --dependency="${dep}" \
  -p gpu -c 2 --mem=16G -t 00:30:00 -J mc-chebi20-report -o workflows/slurm-logs/%x-%j.out \
  --wrap "cd ${HERE} && ${PY} scripts/molnl_chebi20_report.py --arms ${ARMS} --runs ${OUT_ROOT} --overlap-items ${OVERLAP_ITEMS} --out-dir ${REPORT_DIR} --stamp ${STAMP}"
