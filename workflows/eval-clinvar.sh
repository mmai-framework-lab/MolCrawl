#!/usr/bin/env bash
# Phase 3 - ClinVar pathogenicity evaluation.
#
# Required environment:
#   MODEL_PATH      - trained checkpoint (ckpt.pt for gpt2; HF dir for bert)
#   CLINVAR_DATA    - ClinVar CSV/TSV/JSON
#
# Optional:
#   ARCH            - adapter to build (default: gpt2)
#   TOKENIZER_PATH  - required for arch=gpt2 genome; defaulted/ignored
#                     per modality in adapter routing otherwise
#   OUTPUT_DIR      - default ${LEARNING_SOURCE_DIR}/experiment_data/eval/<model-slug>/<RUNTAG>
#                     where <model-slug> is derived from MODEL_PATH and
#                     <RUNTAG> defaults to a unique name (see RUNTAG below).
#   RUNTAG          - leaf directory name under the model slug
#                     (default: clinvar_default). Set this per run to keep
#                     historical results separated, e.g. RUNTAG=clinvar_nper1000.
#   N_PER_CLASS     - class-balanced sample size per class (pathogenic
#                     and benign). Omit to evaluate on the full dataset.
#   STRATIFY_CHROM  - "0" to disable per-chromosome stratified sampling
#                     (default: stratification is on)
#   SEED            - random seed for reproducible sampling (default 42)
#   MAX_EXAMPLES    - legacy cap; when set alone it is re-interpreted as
#                     N_PER_CLASS = MAX_EXAMPLES // 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CLINVAR_DATA:?CLINVAR_DATA must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-clinvar_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir genome_sequence "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.clinvar
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality genome_sequence
     --device "$DEVICE"
     --clinvar-data "$CLINVAR_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${N_PER_CLASS:-}" ]]; then
    cmd+=(--n-per-class "$N_PER_CLASS")
fi
if [[ "${STRATIFY_CHROM:-1}" = "0" ]]; then
    cmd+=(--no-stratify-chrom)
fi
if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    cmd+=(--max-examples "$MAX_EXAMPLES")
fi
"${cmd[@]}"
