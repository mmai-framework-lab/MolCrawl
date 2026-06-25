#!/bin/bash
# SLURM wrapper for the ClinVar AUROC evaluation across all subset pretrain ckpts.
#
# Calls scripts/run_clinvar_eval_subsets.py, which loops over
# (subset, model_type, seed) and invokes
# ``python -m molcrawl.tasks.evaluation.clinvar`` per run.
#
# Defaults are tuned for the "lightweight first pass" (n_per_class=200) —
# completes 54 runs in a few hours on a single H200, results land in
# ${LEARNING_SOURCE_DIR}/genome_sequence/analysis/clinvar_evaluation/.
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   conda activate molcrawl
#   bash workflows/eval-clinvar-subset-ckpts.sh                  # interactive / nohup
#   sbatch --gres=gpu:1 workflows/eval-clinvar-subset-ckpts.sh   # SLURM
#
# Env knobs (passed through to the Python driver):
#   N_PER_CLASS        — default 200 (lightweight pass; raise to 1000-5000 for final)
#   SEEDS              — default "1,2,3"
#   MODELS             — default "bert,gpt2"
#   SUBSETS            — default "" (auto-detect from existing ckpts)
#   DEVICE             — default "cuda"
#   CONTEXT_LEN        — default 512
#   CLINVAR_DATA       — default unset (uses runner default; lets you point
#                        at a filtered CSV, e.g. clinvar_sequences_2star.csv)
#   SCORE_WINDOW_HALF  — default unset (full-sequence average). Set to e.g. 32
#                        to restrict the PLL average to ±32 tokens around the
#                        variant centre.
#   FLANK              — default 64 (only used when SCORE_WINDOW_HALF is set)
#   OUT_TAG            — default unset → writes to .../clinvar_evaluation/.
#                        Set to e.g. "n1000" / "2star" / "window32" to write
#                        to .../clinvar_evaluation_<tag>/ and keep parallel
#                        sweeps from overwriting each other.
#
# ⚠️ SLURM comma-in-value pitfall:
#   sbatch's `--export=ALL,KEY=value,...` parses commas as the separator
#   *between* variables, so `--export=ALL,SEEDS=1,2,3` reaches the script
#   as SEEDS=1 plus the bogus variables `2` and `3`.
#
#   Correct invocation patterns when overriding comma-bearing knobs:
#     # (A) pre-export, then pass --export=ALL:
#     SEEDS="1,2,3" MODELS="bert,gpt2" \
#         sbatch --export=ALL --gres=gpu:1 workflows/eval-clinvar-subset-ckpts.sh
#
#     # (B) quote the entire --export argument:
#     sbatch --export="ALL,SEEDS=1,2,3" --gres=gpu:1 \
#         workflows/eval-clinvar-subset-ckpts.sh
#
#   The script's own `${SEEDS:-1,2,3}` default works fine because no
#   sbatch --export parsing is involved when relying on the default.
#
#   Re-using SKIP_EXISTING (added to the runner 2026-06-16): a second
#   invocation with seeds = a superset of an earlier run will reuse the
#   cached metrics.json files and only execute the new seed values.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/workflows/common_functions.sh" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/workflows"
fi
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

N_PER_CLASS=${N_PER_CLASS:-200}
SEEDS=${SEEDS:-1,2,3}
MODELS=${MODELS:-bert,gpt2}
SUBSETS=${SUBSETS:-}
DEVICE=${DEVICE:-cuda}
CONTEXT_LEN=${CONTEXT_LEN:-512}
CLINVAR_DATA=${CLINVAR_DATA:-}
SCORE_WINDOW_HALF=${SCORE_WINDOW_HALF:-}
FLANK=${FLANK:-64}
OUT_TAG=${OUT_TAG:-}

LOG_DIR="${LEARNING_SOURCE_DIR}/genome_sequence/logs"
mkdir -p "$LOG_DIR"
LOG_TAG_SUFFIX=""
[ -n "$OUT_TAG" ] && LOG_TAG_SUFFIX="-${OUT_TAG}"
LOG_FILE="${LOG_DIR}/clinvar-eval-subset${LOG_TAG_SUFFIX}-$(date +%Y-%m-%d_%H-%M-%S).log"

EXTRA_ARGS=()
if [ -n "$SUBSETS" ]; then
    EXTRA_ARGS+=(--subsets "$SUBSETS")
fi
if [ -n "$CLINVAR_DATA" ]; then
    EXTRA_ARGS+=(--clinvar-data "$CLINVAR_DATA")
fi
if [ -n "$SCORE_WINDOW_HALF" ]; then
    EXTRA_ARGS+=(--score-window-half "$SCORE_WINDOW_HALF" --flank "$FLANK")
fi
if [ -n "$OUT_TAG" ]; then
    EXTRA_ARGS+=(--out-tag "$OUT_TAG")
fi

echo "========================================"
echo "ClinVar AUROC eval — subset pretrain ckpts"
echo "========================================"
echo "GPU:               ${CUDA_VISIBLE_DEVICES}"
echo "n_per_class:       ${N_PER_CLASS}"
echo "seeds:             ${SEEDS}"
echo "models:            ${MODELS}"
echo "subsets:           ${SUBSETS:-auto-detect}"
echo "context_len:       ${CONTEXT_LEN}"
echo "device:            ${DEVICE}"
echo "clinvar_data:      ${CLINVAR_DATA:-<runner default>}"
echo "score_window_half: ${SCORE_WINDOW_HALF:-<full-seq mean>}"
echo "flank:             ${FLANK}"
echo "out_tag:           ${OUT_TAG:-<none → clinvar_evaluation/>}"
echo "log:               ${LOG_FILE}"
echo

run_training_background "$LOG_FILE" \
    scripts/run_clinvar_eval_subsets.py \
    --n-per-class "${N_PER_CLASS}" \
    --seeds "${SEEDS}" \
    --models "${MODELS}" \
    --device "${DEVICE}" \
    --context-length "${CONTEXT_LEN}" \
    "${EXTRA_ARGS[@]}"
