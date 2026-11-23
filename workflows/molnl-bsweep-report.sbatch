#!/bin/bash
#SBATCH --job-name=mc-molnl-bsweep-report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Write what the 2026-11-20 directive's items 10, 11 and 12 ask for, off the finished runs.
#
# No GPU: it reads trainer_state.json, nanoGPT's logging CSVs and run_manifest.json. It
# still goes through the scheduler -- nothing computes on the login node.
#
#   sbatch [--dependency=afterany:<arm jobs>] \
#          --export=ALL,ARMS=<arm inventory tsv>,RUNS_ROOT=<MODEL_OUTPUT_ROOT>,\
# OUT_DIR=<report dir>[,SUBMIT_RECORD_IN=<file>][,STAMP=YYYYMMDD] \
#          workflows/molnl-bsweep-report.sbatch
#
# RUNS_ROOT is the output root the grid was submitted with -- the PARENT of
# molecule_nat_lang/, not that directory. Both readers append the modality themselves,
# and pointing one level too deep makes every arm read as "nothing to read".
#
# Meant to be queued with --dependency=afterany on the last arms, so the tables are
# written whether or not a session is awake when the grid ends. afterany rather than
# afterok because an arm that ends badly still has to be reported, and because the
# completeness check below is what decides whether the tables are final: it exits
# non-zero when any arm is short of its planned step, so the failure recorder chained
# behind this job fires and the gap is not mistaken for a finished grid.
set -eo pipefail

: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${RUNS_ROOT:?set RUNS_ROOT to the grid output root, the parent of molecule_nat_lang/}"
: "${OUT_DIR:?set OUT_DIR to the directory the tables are written to}"
STAMP="${STAMP:-$(date +%Y%m%d)}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
[ -f "${ARMS}" ] || { echo "not a file: ${ARMS}" >&2; exit 1; }
[ -d "${RUNS_ROOT}/molecule_nat_lang" ] || {
  echo "RUNS_ROOT=${RUNS_ROOT} has no molecule_nat_lang/ under it -- it is one level too deep" >&2
  exit 1; }
[ -d "${OUT_DIR}" ] || { echo "not a directory: ${OUT_DIR}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

CURVES="${OUT_DIR}/rt1-molnl-bsweep-curves-${STAMP}.tsv"
SUMMARY="${OUT_DIR}/rt1-molnl-bsweep-arm-summary-${STAMP}.tsv"
MANIFESTS="${OUT_DIR}/rt1-molnl-bsweep-manifests-${STAMP}.tsv"

echo "=== bsweep report on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    arms:    ${ARMS}"
echo "    runs:    ${RUNS_ROOT}"
echo "    outputs: ${CURVES}"
echo "             ${SUMMARY}"
echo "             ${MANIFESTS}"

rc=0
# Items 10 and 12 come out of one pass: the per-arm summary and the curve table are read
# off the same series, so they cannot disagree about where an arm's minimum is.
# No --band and no --floor: neither has been measured for this grid, and the readers
# refuse to claim a minimum or to call an arm collapsed against a number with no source.
echo "=== items 10 and 12: the curves and the per-arm summary ==="
"${PYTHON}" scripts/molnl_batch_sweep_curves.py \
  --arms "${ARMS}" --root "${RUNS_ROOT}" \
  --curves-out "${CURVES}" --summary-out "${SUMMARY}" || rc=$?

echo "=== item 11: every arm's manifest read back against the plan ==="
"${PYTHON}" scripts/molnl_batch_sweep_manifests.py \
  --arms "${ARMS}" --runs-root "${RUNS_ROOT}" \
  ${SUBMIT_RECORD_IN:+--submit-record "${SUBMIT_RECORD_IN}"} \
  --out "${MANIFESTS}" || rc=$?

# Is the grid actually finished? Read it from the summary the pass just wrote rather
# than from the queue: a spare segment refused after a finished run is recorded FAILED,
# so the queue cannot answer this.
echo "=== completeness: every arm at its planned step ==="
if [ -f "${SUMMARY}" ]; then
  "${PYTHON}" - "${SUMMARY}" <<'PYEOF' || rc=$?
import csv, sys
rows = [r for r in csv.DictReader(
    (l for l in open(sys.argv[1], encoding="utf-8") if not l.startswith("#")),
    delimiter="\t")]
short = [r for r in rows
         if not r.get("steps_reached") or not r.get("steps_planned")
         or int(r["steps_reached"]) < int(r["steps_planned"])]
print(f"    arms in the summary: {len(rows)}")
for r in short:
    print(f"    SHORT {r['arch']} {r['size']} batch {r['batch']} lr {r['lr']}: "
          f"{r.get('steps_reached') or '-'} of {r.get('steps_planned') or '-'}")
if short:
    raise SystemExit(f"{len(short)} arm(s) short of the planned step: these tables are "
                     f"not the final ones")
print("    all arms at or past their planned step")
PYEOF
else
  echo "    no summary was written, so completeness cannot be judged" >&2; rc=1
fi

echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
