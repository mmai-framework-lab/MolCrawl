#!/bin/bash
#SBATCH --job-name=mc-gen-clinvar-zs
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Zero-shot ClinVar scoring for a genome subset campaign, one adopted
# checkpoint per subset.
#
# The two architectures are scored differently, by the 2026-09-08 correction.
#
#   BERT   the variant position alone is masked while the rest of the window
#          stays visible, and the score is the log probability of the reference
#          base minus that of the variant base at that one position.
#   GPT-2  the difference of whole-window log likelihood. A single conditional
#          probability drops everything downstream of the variant, and putting
#          the variant into the context is exactly what changes those
#          predictions, so that has to be kept in the score.
#
# FLANK is where the variant sits inside the window, and it must match the data:
# the 1,024-base build puts it at index 512, the older 129-base one at 64. Get it
# wrong and BERT masks a position that is not the variant, with no error.
#
# CONTEXT_LENGTH must cover the window. The evaluator truncates to it silently
# (ids = ids[:context_length]), so a 1,024-base window under the 512 default
# loses the variant itself and still returns numbers.
#
# CHROMS restricts scoring to the chromosomes every pretraining subset held out.
# The evaluator has no chromosome filter of its own, so the table is subset here
# once and every run scores the same rows -- which is also what makes the
# per-variant comparison across subsets a paired one.
#
# Required: RUNS_ROOT, CLINVAR_CSV, TOKENIZER, ARCH, RUN_GLOB.
# Optional: ADOPTED_CSV (subset,adopted_ckpt_step), RUN_TAG, N_PER_CLASS,
#           SCORE_WINDOW_HALF, OUT_DIR, CHROMS, FLANK, CONTEXT_LENGTH.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${CLINVAR_CSV:?set CLINVAR_CSV to clinvar_sequences.csv}"
: "${TOKENIZER:?set TOKENIZER to the genome single-nucleotide tokenizer}"
: "${ARCH:?set ARCH to bert or gpt2}"
: "${RUN_GLOB:?set RUN_GLOB to the run directory pattern, e.g. gpt2-small-*}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RESULTS_ROOT="${RESULTS_ROOT:-${PYROOT}/learning_source_genome_runs/_results}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/clinvar-zeroshot-${ARCH}}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== runs:    ${RUNS_ROOT}/${RUN_GLOB} ==="
echo "=== outputs: ${OUT_DIR} ==="
# Defaults follow the architecture and the 1,024-base build; both stay overridable.
FLANK="${FLANK:-512}"
if [ "${ARCH}" = "gpt2" ]; then
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"        # block_size
    SCORE_ARGS=""                                    # whole-window log likelihood
    SCORE_DESC="whole-window log likelihood"
else
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-1026}"        # [CLS] + 1024 + [SEP]
    SCORE_ARGS="--score-window-half ${SCORE_WINDOW_HALF:-0}"
    SCORE_DESC="masked at the variant position (score-window-half ${SCORE_WINDOW_HALF:-0})"
fi
echo "=== scoring: ${SCORE_DESC} ==="
echo "=== flank ${FLANK}, context-length ${CONTEXT_LENGTH} ==="

# Restrict the table to the held-out chromosomes, once, into a file every run
# reads. Scoring all 221,873 variants would spend most of the time on sequence
# all 21 subsets trained on.
CHROMS="${CHROMS:-21,22,X,Y}"
SCORED_CSV="${OUT_DIR}/clinvar_sequences_chr${CHROMS//,/_}.csv"
if [ ! -s "${SCORED_CSV}" ]; then
    "${PY}" - "${CLINVAR_CSV}" "${SCORED_CSV}" "${CHROMS}" <<'PYEOF'
import csv, sys
src, dst, chroms = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
kept = 0
with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=r.fieldnames)
    w.writeheader()
    for row in r:
        if (row.get("chrom") or "").replace("chr", "").strip() in chroms:
            w.writerow(row); kept += 1
print(f"  filtered to {kept:,} variants on chr{'/'.join(sorted(chroms))}")
PYEOF
    [ -s "${SCORED_CSV}" ] || { echo "REFUSING: chromosome filter produced nothing" >&2; exit 1; }
else
    echo "=== reusing ${SCORED_CSV} ==="
fi
echo "=== scored table: ${SCORED_CSV} ($(( $(wc -l < "${SCORED_CSV}") - 1 )) variants) ==="

rc=0
for d in "${RUNS_ROOT}"/${RUN_GLOB}/; do
    [ -d "${d}" ] || continue
    run=$(basename "${d}")
    subset=$(echo "${run}" | sed "s/^${ARCH}-small-//;s/^bert-small-//;s/-${RUN_TAG:-__none__}\$//")

    # Which checkpoint this subset adopted. A CSV of (subset, step) is the
    # authority when one is given -- the campaign's own record -- and the newest
    # checkpoint is only a fallback for a run that has none.
    step=""
    if [ -n "${ADOPTED_CSV:-}" ] && [ -f "${ADOPTED_CSV}" ]; then
        step=$(awk -F, -v s="${subset}" 'NR>1 && $1==s {print $11}' "${ADOPTED_CSV}")
    fi
    if [ -z "${step}" ]; then
        step=$(ls -d "${d}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    fi
    # The two architectures store a checkpoint differently and their adapters
    # take different things: the MLM adapter loads an HF directory, the nanoGPT
    # adapter torch.loads a single .pt file. ckpt.pt at the run root is the
    # best-val model, which is the checkpoint the campaign adopted, and its
    # iter_num is checked against the adoption record below.
    if [ "${ARCH}" = "gpt2" ]; then
        ckpt="${d}/ckpt.pt"
    else
        ckpt="${d}/checkpoint-${step}"
    fi
    if [ ! -e "${ckpt}" ]; then
        echo "  SKIP ${subset}: ${ckpt} is not there"; rc=1; continue
    fi

    echo ""; echo "########## ${subset}  (adopted step ${step}, scoring ${ckpt##*/}) ##########"
    "${PY}" -m molcrawl.tasks.evaluation.clinvar \
        --model-path "${ckpt}" \
        --tokenizer-path "${TOKENIZER}" \
        --clinvar-data "${SCORED_CSV}" \
        --arch "${ARCH}" \
        --modality genome_sequence \
        ${SCORE_ARGS} \
        --flank "${FLANK}" \
        --context-length "${CONTEXT_LENGTH}" \
        ${N_PER_CLASS:+--n-per-class "${N_PER_CLASS}"} \
        --output-dir "${OUT_DIR}/${subset}" || { echo "  FAILED: ${subset}"; rc=1; }
done
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
