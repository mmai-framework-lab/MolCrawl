#!/usr/bin/env bash
# External HuggingFace baseline sweep.
#
# Runs our evaluator suite against public HF models so the dashboard
# can compare in-house pretrain / fine-tune results against the
# canonical released baselines.
#
# Targets (one representative per arch family):
#   compounds × chemberta2  → seyonec/ChemBERTa-zinc-base-v1
#   protein_sequence × esm2 → facebook/esm2_t12_35M_UR50D
#   genome_sequence × dnabert2 → zhihan1996/DNABERT-2-117M
#
# Output layout: ``<eval>/external-<arch>-<short>/<task>_baseline_gapfill/``
# — a dedicated parent slug so external baselines sort separately from
# in-house model-slug entries in the dashboard. ``extras.finetune_corpus``
# stays unset (they're foreign pretrains).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LSD="${LEARNING_SOURCE_DIR:-${REPO_ROOT}/learning_source_20260316}"
export LEARNING_SOURCE_DIR="${LSD}"

DEVICE="${DEVICE:-cuda}"
MAX="${MAX_EXAMPLES:-200}"
BOOTSTRAP="${BOOTSTRAP:-50}"
DRY="${DRY_RUN:-0}"

# Dispatcher mirroring eval-gapfill.sh's gap_run idiom.
baseline_run() {
    local slug="$1" task="$2" arch="$3" modality="$4"
    shift 4

    local runtag="${task}_baseline_gapfill"
    local outdir="${LSD%/}/experiment_data/eval/external-${slug}/${runtag}"
    if find "$outdir" -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[baseline] SKIP done: external-${slug}/${task}"
        return 0
    fi
    if [[ "$DRY" = "1" ]]; then
        echo "[baseline] DRY:  external-${slug}/${task} -> ${outdir}"
        return 0
    fi
    echo "[baseline] RUN:  external-${slug}/${task} -> ${outdir}"
    mkdir -p "$outdir"
    OUTPUT_DIR="$outdir" DEVICE="$DEVICE" BOOTSTRAP="$BOOTSTRAP" MAX_EXAMPLES="$MAX" \
        "$@" 2>&1 | tail -3 || {
            echo "[baseline] FAIL: external-${slug}/${task}" >&2
            return 1
        }
}

# -----------------------------------------------------------------------------
# ChemBERTa-2 (seyonec/ChemBERTa-zinc-base-v1) — compounds × moleculenet
# -----------------------------------------------------------------------------
CHEMBERTA_ID="seyonec/ChemBERTa-zinc-base-v1"
MOLECULENET_DIR="${LSD}/eval/moleculenet"
if [[ -d "$MOLECULENET_DIR" ]]; then
    baseline_run "chemberta2-zinc-base" moleculenet chemberta2 compounds \
        bash -c "MODEL_PATH='${CHEMBERTA_ID}' \
                 MOLECULENET_DIR='${MOLECULENET_DIR}' \
                 ARCH=chemberta2 MODALITY=compounds \
                 SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                 bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
fi

# -----------------------------------------------------------------------------
# ESM-2 (facebook/esm2_t12_35M_UR50D) — protein_sequence × {proteingym, deeploc, tape}
# -----------------------------------------------------------------------------
ESM2_ID="facebook/esm2_t12_35M_UR50D"
OPSD_CSV="${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions/OPSD_HUMAN_Wan_2019.csv"
DEEPLOC_DATA="${LSD}/eval/deeploc/deeploc.csv"
TAPE_DIR="${LSD}/eval/tape"

if [[ -f "$OPSD_CSV" ]]; then
    baseline_run "esm2-t12-35M" proteingym esm2 protein_sequence \
        bash -c "MODEL_PATH='${ESM2_ID}' \
                 PROTEINGYM_DATA='${OPSD_CSV}' \
                 ARCH=esm2 MODALITY=protein_sequence \
                 bash '${REPO_ROOT}/workflows/eval-proteingym.sh'"
fi
if [[ -f "$DEEPLOC_DATA" ]]; then
    baseline_run "esm2-t12-35M" deeploc esm2 protein_sequence \
        bash -c "MODEL_PATH='${ESM2_ID}' \
                 DEEPLOC_DATA='${DEEPLOC_DATA}' \
                 ARCH=esm2 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
fi
if [[ -d "$TAPE_DIR" ]]; then
    baseline_run "esm2-t12-35M" tape esm2 protein_sequence \
        bash -c "MODEL_PATH='${ESM2_ID}' \
                 TAPE_DIR='${TAPE_DIR}' \
                 TASKS='fluorescence stability remote_homology secondary_structure_3 secondary_structure_8' \
                 ARCH=esm2 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-tape.sh'"
fi

# -----------------------------------------------------------------------------
# DNABERT-2 (zhihan1996/DNABERT-2-117M) — genome_sequence × {clinvar, gnomad, gue, cosmic, omim}
# -----------------------------------------------------------------------------
DNABERT2_ID="zhihan1996/DNABERT-2-117M"
CLINVAR_DATA="${LSD}/genome_sequence/clinvar/clinvar_sequences.csv"
GNOMAD_DATA="${LSD}/eval/gnomad_af_correlation/gnomad_chr22.csv"
COSMIC_DATA="${LSD}/eval/cosmic/cosmic_eval.csv"
OMIM_DATA="${LSD}/eval/omim/omim_eval.csv"
GUE_DIR="${LSD}/eval/gue"

if [[ -f "$CLINVAR_DATA" ]]; then
    baseline_run "dnabert2-117M" clinvar dnabert2 genome_sequence \
        bash -c "MODEL_PATH='${DNABERT2_ID}' \
                 CLINVAR_DATA='${CLINVAR_DATA}' \
                 ARCH=dnabert2 N_PER_CLASS=100 \
                 bash '${REPO_ROOT}/workflows/eval-clinvar.sh'"
fi
if [[ -f "$GNOMAD_DATA" ]]; then
    baseline_run "dnabert2-117M" gnomad_af_correlation dnabert2 genome_sequence \
        bash -c "MODEL_PATH='${DNABERT2_ID}' \
                 GNOMAD_DATA='${GNOMAD_DATA}' \
                 ARCH=dnabert2 N_PER_BIN=100 \
                 bash '${REPO_ROOT}/workflows/eval-gnomad.sh'"
fi
if [[ -d "$GUE_DIR" ]]; then
    baseline_run "dnabert2-117M" gue dnabert2 genome_sequence \
        bash -c "MODEL_PATH='${DNABERT2_ID}' \
                 GUE_DIR='${GUE_DIR}' \
                 TASKS='prom_300_all H3 tf_0' \
                 ARCH=dnabert2 \
                 bash '${REPO_ROOT}/workflows/eval-gue.sh'"
fi
if [[ -f "$COSMIC_DATA" ]]; then
    baseline_run "dnabert2-117M" cosmic dnabert2 genome_sequence \
        bash -c "MODEL_PATH='${DNABERT2_ID}' \
                 COSMIC_DATA='${COSMIC_DATA}' \
                 ARCH=dnabert2 N_PER_CLASS=50 BOOTSTRAP_SAMPLES=$BOOTSTRAP \
                 bash '${REPO_ROOT}/workflows/eval-cosmic.sh'"
fi
if [[ -f "$OMIM_DATA" ]]; then
    baseline_run "dnabert2-117M" omim dnabert2 genome_sequence \
        bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                 --model-path '${DNABERT2_ID}' \
                 --arch dnabert2 --modality genome_sequence --device '$DEVICE' \
                 --omim-data '${OMIM_DATA}' --output-dir \"\$OUTPUT_DIR\""
fi

echo
echo "[baseline] all baselines processed."
echo "[baseline] regenerate dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
