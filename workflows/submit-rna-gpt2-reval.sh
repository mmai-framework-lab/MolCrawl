#!/bin/bash
# Queue the fixed-validation re-measurement of finished rna GPT-2 runs.
#
# One job per (run, checkpoint). They are separate jobs rather than one job over
# the whole grid because the cost scales with the model -- 4h53m per checkpoint at
# small, about 71h at xl -- and a single job holding all of them would lose every
# point it had not yet written when it hit the wall.
#
# RUNS names run directories under GPT2_RUNS_ROOT. For each one, both the best
# checkpoint (ckpt.pt) and the final one are measured, and each writes its own TSV
# named after the run, so a job that dies costs only its own point.
#
# The run names carry suffixes: -seed1 / -seed17 for the seed-variance points and
# -9ep for the nine-epoch reruns. rna_revalidate_gpt2.py matches them explicitly;
# eight jobs were lost in 2026-10 to a pattern that could not, so the submitter
# prints the names it is about to pass and the script lists what it matched before
# measuring anything.
#
# Usage:
#   LEARNING_SOURCE_DIR=... MODEL_OUTPUT_ROOT=... GPT2_RUNS_ROOT=... \
#     REVAL_OUT_DIR=... SLURM_ACCOUNT=... \
#     RUNS="rna-gpt2-small-lr6e4-seed1 ..." ./workflows/submit-rna-gpt2-reval.sh
#
#   DRY_RUN=1 ...        prints the plan without submitting.
#   FINAL_CKPT=...       the final checkpoint's path inside a run directory.
#   WHICH="ckpt final"   restricts to one of the two.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to the root the configs join run names onto}"
: "${GPT2_RUNS_ROOT:?set GPT2_RUNS_ROOT to the directory holding the rna-gpt2-* run directories}"
: "${REVAL_OUT_DIR:?set REVAL_OUT_DIR to a directory for the TSVs, outside every input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${RUNS:?set RUNS to the run directory names to measure}"

REVAL_CONFIG="${REVAL_CONFIG:-molcrawl/tasks/pretrain/configs/rna/gpt2_small_lr6e4.py}"
FINAL_CKPT="${FINAL_CKPT:-checkpoint-40320/training_state.bin}"
WHICH="${WHICH:-ckpt final}"
[ -f "${REVAL_CONFIG}" ] || { echo "missing config: ${REVAL_CONFIG}" >&2; exit 1; }

echo "=== rna GPT-2 fixed-validation re-measurement ==="
echo "  runs root: ${GPT2_RUNS_ROOT}"
echo "  out dir:   ${REVAL_OUT_DIR}"
echo "  config:    ${REVAL_CONFIG}"
n=0
for run in ${RUNS}; do
    dir="${GPT2_RUNS_ROOT}/${run}"
    [ -d "${dir}" ] || { echo "missing run directory: ${dir}" >&2; exit 1; }
    for which in ${WHICH}; do
        case "${which}" in
            ckpt)  ck="ckpt.pt" ;;
            final) ck="${FINAL_CKPT}" ;;
            *) echo "unknown WHICH entry: ${which}" >&2; exit 1 ;;
        esac
        # A missing checkpoint would make the script skip the only run it was given
        # and then exit on the "no run to measure" guard, after loading the whole
        # validation split. Cheaper to say so here.
        [ -f "${dir}/${ck}" ] || { echo "missing checkpoint: ${dir}/${ck}" >&2; exit 1; }

        # REVAL_RUN is an exact match on the directory name, not ONLY, which is a
        # substring test: the seed-variance runs end in -seed1 and -seed17, so
        # ONLY=...-seed1 would pull -seed17 into the same job and measure it twice.
        label="${run#rna-gpt2-}"
        out="${REVAL_OUT_DIR}/${label}-${which}.tsv"
        env=(ALL
             "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
             "MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
             "GPT2_RUNS_ROOT=${GPT2_RUNS_ROOT}"
             "REVAL_CONFIG=${REVAL_CONFIG}"
             "REVAL_OUT=${out}"
             "REVAL_CHECKPOINTS=${ck}"
             "REVAL_RUN=${run}")
        args=(--job-name="rna-reval-${label}-${which}"
              --account="${SLURM_ACCOUNT}"
              --export="$(IFS=,; echo "${env[*]}")")
        if [ -n "${DRY_RUN:-}" ]; then
            echo "  [dry-run] ${run} ${which} -> ${out}"
        elif [ -n "${SUBMIT_RECORD:-}" ]; then
            jid=$(workflows/sbatch-watched.sh "reval-${label}-${which}" -- \
                  "${args[@]}" workflows/rna-revalidate-gpt2.sbatch)
            echo "  ${run} ${which}: job ${jid}"
        else
            jid=$(sbatch --parsable "${args[@]}" workflows/rna-revalidate-gpt2.sbatch)
            echo "  ${run} ${which}: job ${jid}"
        fi
        n=$((n + 1))
    done
done
echo "=== ${n} jobs ==="
