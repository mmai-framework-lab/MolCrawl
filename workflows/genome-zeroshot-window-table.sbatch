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
# The table carries one note line on what input_form means per series: the
# column reads as "not matched" while each architecture is scored in its own
# training-time form. The part the rows cannot say is passed in with --note.
#
# Required: RESULTS_ROOT. Optional: WINDOW_DIR, OUT, BERT_WINDOW_ONLY, WITH_EP3.

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
# The three-epoch GPT-2 series is not among them. The 2026-10-30 judgment puts
# that series away: nine epochs is the base, and the three-epoch runs go into
# neither a figure nor the text. Its outputs stay on disk, and WITH_EP3=1 reads
# them back -- from clinvar-zeroshot-gpt2-w1024, not from clinvar-zeroshot-gpt2,
# which scored at context_length 512 with flank 64, so a row from that directory
# would differ by the scoring window and not by the epochs.
#
# BERT_WINDOW_ONLY=1 drops the comparison series and leaves the whole-window
# BERT rows alone.
CMP_ARGS=()
if [ -z "${BERT_WINDOW_ONLY:-}" ]; then
    CMP_ARGS+=(--predictions-dir "gpt2-ep9:bare:window:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-ep9")
    CMP_ARGS+=(--predictions-dir "bert-w1026-variant:bare:variant:${RESULTS_ROOT}/clinvar-zeroshot-bert-w1026")
    # The one series whose form is not its training-time one. The 2026-09-09
    # evaluator passed add_special_tokens=True, and this tokeniser has [CLS]
    # and [SEP] in its vocabulary but no post-processor template, so nothing
    # was prepended: BERT read 1,024 bare bases. Checked by encoding a row
    # (job 207150, scripts/check_clinvar_scored_position.py).
    CMP_ARGS+=(--note "bert-w1026-variant は 2026-09-09 の評価経路で、BERT に 1,024 トークンをそのまま入れている。これは BERT の学習時の形ではない（このトークナイザは [CLS]/[SEP] を語彙に持つが後処理のテンプレートを持たず、add_special_tokens=True でも何も付かない）。")
    if [ -n "${WITH_EP3:-}" ]; then
        CMP_ARGS+=(--predictions-dir "gpt2:bare:window:${RESULTS_ROOT}/clinvar-zeroshot-gpt2-w1024")
    fi
fi

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_zeroshot_window_table.py \
    --window-dir "bert-w1026:${WINDOW_DIR}" \
    ${CMP_ARGS[@]+"${CMP_ARGS[@]}"} \
    --out-tsv "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
