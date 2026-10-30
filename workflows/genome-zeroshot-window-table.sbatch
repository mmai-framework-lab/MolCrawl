#!/bin/bash
#SBATCH --job-name=mc-gen-zs-wintable
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The one table the zero-shot values are reported in: per chromosome and a
# count-weighted mean, with no pooled-overall column.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: WINDOW_DIR, OUT, BERT_WINDOW_ONLY.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

WINDOW_DIR="${WINDOW_DIR:-${RESULTS_ROOT}/analyses/ep9}"
OUT="${OUT:-${RESULTS_ROOT}/analyses/ep9/zeroshot-window-21.tsv}"

# The series the comparison is built from are stored as per-variant scores, not
# as window-*.json, and they are scored over their own 1,024-token window with
# no [CLS]/[SEP]. They are in the table by default: a table that carries only
# the BERT side cannot be checked against the comparison it feeds.
#
# gpt2 (three epochs) is read from clinvar-zeroshot-gpt2-w1024, not from
# clinvar-zeroshot-gpt2. That directory scored at context_length 512 with flank
# 64, while the nine-epoch series and -w1024 both scored at 1024 with flank 512,
# so a row from it would differ by the scoring window and not by the epochs.
#
# BERT_WINDOW_ONLY=1 drops them and leaves the whole-window BERT rows alone.
CMP_ARGS=()
if [ -z "${BERT_WINDOW_ONLY:-}" ]; then
    CMP_ARGS+=(--predictions-dir "gpt2-ep9:bare:window:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-ep9")
    CMP_ARGS+=(--predictions-dir "gpt2:bare:window:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-w1024")
    CMP_ARGS+=(--predictions-dir "bert-w1026-variant:bare:variant:${RESULTS_ROOT}/clinvar-zeroshot-bert-w1026")
fi

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_zeroshot_window_table.py \
    --window-dir "bert-w1026:${WINDOW_DIR}" \
    ${CMP_ARGS[@]+"${CMP_ARGS[@]}"} \
    --out-tsv "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
