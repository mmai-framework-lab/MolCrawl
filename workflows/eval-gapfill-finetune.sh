#!/usr/bin/env bash
# Gap-fill driver for fine-tuned models.
#
# Walks the task-specific training output directories
# (compounds_chembl, compounds_guacamol, genome_sequence_clinvar,
#  molecule_nat_lang_mol_instructions, protein_sequence_proteingym,
#  rna_celltype) and dispatches each (corpus, arch, size) checkpoint
# through the evaluators whose data domain matches the fine-tune
# corpus.
#
# Outputs land in the canonical model-first layout::
#   ${LEARNING_SOURCE_DIR}/experiment_data/eval/<modality>-<arch>-<size>/finetune_<corpus_short>_<task>_gapfill/
#
# The leaf name carries ``finetune_<corpus_short>_<task>`` so fine-tune
# results don't collide with pretrain leaves under the same model-slug
# parent, and so the dashboard can group them visually.
#
# Optional environment:
#   GAPFILL_MAX        - shared MAX_EXAMPLES (default 200 on GPU)
#   GAPFILL_DEVICE     - cuda/cpu (default cpu)
#   GAPFILL_BOOTSTRAP  - bootstrap CI resamples (default 30)
#   GAPFILL_FILTER     - regex restricting which "<corpus>__<arch>__<size>__<task>"
#                        run ids are dispatched
#   GAPFILL_DRY_RUN    - "1" to print what would run and exit
#   GAPFILL_ONLY_ARCH  - "gpt2" / "bert" (default gpt2 — priority 高)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LSD="${LEARNING_SOURCE_DIR:-${REPO_ROOT}/../learning_source_20260316}"
export LEARNING_SOURCE_DIR="${LSD}"

MAX="${GAPFILL_MAX:-200}"
DEVICE="${GAPFILL_DEVICE:-cpu}"
BOOTSTRAP="${GAPFILL_BOOTSTRAP:-30}"
FILTER="${GAPFILL_FILTER:-}"
DRY="${GAPFILL_DRY_RUN:-0}"
ONLY_ARCH="${GAPFILL_ONLY_ARCH:-gpt2}"

# corpus → (base_modality, corpus_short_name)
declare -A CORPUS_MODALITY=(
    [compounds_chembl]="compounds"
    [compounds_guacamol]="compounds"
    [genome_sequence_clinvar]="genome_sequence"
    [molecule_nat_lang_mol_instructions]="molecule_nat_lang"
    [protein_sequence_proteingym]="protein_sequence"
    [rna_celltype]="rna"
)
declare -A CORPUS_SHORT=(
    [compounds_chembl]="chembl"
    [compounds_guacamol]="guacamol"
    [genome_sequence_clinvar]="clinvar"
    [molecule_nat_lang_mol_instructions]="mol_instructions"
    [protein_sequence_proteingym]="proteingym"
    [rna_celltype]="celltype"
)

# Resolve <corpus>/<arch>-output/<corpus>-<size>/ — the fine-tune
# checkpoint container. Returns 0 with stdout=ckpt-path on success.
resolve_finetune_ckpt() {
    local corpus="$1" arch="$2" size="$3"
    local container="${LSD}/${corpus}/${arch}-output/${corpus}-${size}"
    [[ -d "$container" ]] || return 1
    local ckpt="${container}/ckpt.pt"
    if [[ -f "$ckpt" ]]; then
        echo "$ckpt"; return 0
    fi
    ckpt=$(ls -1d "${container}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    if [[ -d "$ckpt" ]]; then
        echo "$ckpt"; return 0
    fi
    return 1
}

# Run a single fine-tune evaluation. Idempotent: if the leaf already has
# REPORT.md, skip. Args: corpus task arch size cmd...
ft_run() {
    local corpus="$1" task="$2" arch="$3" size="$4"
    shift 4

    local run_id="${corpus}__${arch}__${size}__${task}"
    if [[ -n "$FILTER" ]] && [[ ! "$run_id" =~ ${FILTER} ]]; then
        return 0
    fi
    if [[ -n "$ONLY_ARCH" ]] && [[ "$arch" != "$ONLY_ARCH" ]]; then
        return 0
    fi

    local modality="${CORPUS_MODALITY[$corpus]}"
    local short="${CORPUS_SHORT[$corpus]}"
    local slug="${modality}-${arch}-${size}"
    local runtag="finetune_${short}_${task}_gapfill"
    local outdir="${LSD%/}/experiment_data/eval/${slug}/${runtag}"

    if find "$outdir" -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[ft-gapfill] SKIP done: $run_id"
        return 0
    fi

    if [[ "$DRY" = "1" ]]; then
        echo "[ft-gapfill] DRY:  $run_id -> $outdir"
        return 0
    fi

    echo "[ft-gapfill] RUN:  $run_id -> $outdir"
    mkdir -p "$outdir"
    OUTPUT_DIR="$outdir" RUNTAG="$runtag" DEVICE="$DEVICE" \
        BOOTSTRAP="$BOOTSTRAP" MAX_EXAMPLES="$MAX" \
        "$@" 2>&1 | tail -3 || {
            echo "[ft-gapfill] FAIL: $run_id" >&2
            return 1
        }
}

# ---------------------------------------------------------------------------
# compounds_chembl × {moses, moleculenet}
# (chembl_scaffold_heldout × gpt2 already done in matrix_v1)
# ---------------------------------------------------------------------------
MOSES_DIR="${LSD}/eval/moses"
MOLECULENET_DIR="${LSD}/eval/moleculenet"
VOCAB="${REPO_ROOT}/assets/molecules/vocab.txt"
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt compounds_chembl gpt2 "$size") || continue
    if [[ -d "$MOSES_DIR" ]]; then
        ft_run compounds_chembl moses gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$VOCAB' \
                     MOSES_DIR='$MOSES_DIR' \
                     NUM_SAMPLES=200 MAX_NEW_TOKENS=128 \
                     bash '${REPO_ROOT}/workflows/eval-moses.sh'"
    fi
    if [[ -d "$MOLECULENET_DIR" ]]; then
        ft_run compounds_chembl moleculenet gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$VOCAB' \
                     MOLECULENET_DIR='$MOLECULENET_DIR' \
                     ARCH=gpt2 SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
    fi
done

# ---------------------------------------------------------------------------
# compounds_guacamol × {moses, moleculenet}
# ---------------------------------------------------------------------------
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt compounds_guacamol gpt2 "$size") || continue
    if [[ -d "$MOSES_DIR" ]]; then
        ft_run compounds_guacamol moses gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$VOCAB' \
                     MOSES_DIR='$MOSES_DIR' \
                     NUM_SAMPLES=200 MAX_NEW_TOKENS=128 \
                     bash '${REPO_ROOT}/workflows/eval-moses.sh'"
    fi
    if [[ -d "$MOLECULENET_DIR" ]]; then
        ft_run compounds_guacamol moleculenet gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$VOCAB' \
                     MOLECULENET_DIR='$MOLECULENET_DIR' \
                     ARCH=gpt2 SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
    fi
done

# ---------------------------------------------------------------------------
# genome_sequence_clinvar × {clinvar, gnomad, cosmic, omim}
# ---------------------------------------------------------------------------
TOKENIZER_GENOME="${LSD}/genome_sequence/spm_tokenizer.model"
CLINVAR_DATA="${LSD}/genome_sequence/clinvar/clinvar_sequences.csv"
GNOMAD_DATA="${LSD}/eval/gnomad_af_correlation/gnomad_chr22.csv"
COSMIC_DATA="${LSD}/eval/cosmic/cosmic_eval.csv"
OMIM_DATA="${LSD}/eval/omim/omim_eval.csv"
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt genome_sequence_clinvar gpt2 "$size") || continue
    if [[ -f "$CLINVAR_DATA" ]]; then
        ft_run genome_sequence_clinvar clinvar gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$TOKENIZER_GENOME' \
                     CLINVAR_DATA='$CLINVAR_DATA' ARCH=gpt2 N_PER_CLASS=100 \
                     bash '${REPO_ROOT}/workflows/eval-clinvar.sh'"
    fi
    if [[ -f "$GNOMAD_DATA" ]]; then
        ft_run genome_sequence_clinvar gnomad_af_correlation gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$TOKENIZER_GENOME' \
                     GNOMAD_DATA='$GNOMAD_DATA' ARCH=gpt2 N_PER_BIN=100 \
                     bash '${REPO_ROOT}/workflows/eval-gnomad.sh'"
    fi
    if [[ -f "$COSMIC_DATA" ]]; then
        ft_run genome_sequence_clinvar cosmic gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TOKENIZER_PATH='$TOKENIZER_GENOME' \
                     COSMIC_DATA='$COSMIC_DATA' ARCH=gpt2 \
                     N_PER_CLASS=50 BOOTSTRAP_SAMPLES=$BOOTSTRAP \
                     bash '${REPO_ROOT}/workflows/eval-cosmic.sh'"
    fi
    if [[ -f "$OMIM_DATA" ]]; then
        ft_run genome_sequence_clinvar omim gpt2 "$size" \
            bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                     --model-path '$ckpt' --tokenizer-path '$TOKENIZER_GENOME' \
                     --arch gpt2 --modality genome_sequence --device '$DEVICE' \
                     --omim-data '$OMIM_DATA' --output-dir \"\$OUTPUT_DIR\""
    fi
done

# ---------------------------------------------------------------------------
# molecule_nat_lang_mol_instructions × {chebi20, chemllmbench, molecule_nat_lang}
# ---------------------------------------------------------------------------
CHEBI20_DIR="${LSD}/eval/chebi20"
CHEMLLM_DIR="${LSD}/eval/chemllmbench"
PAIRS_CSV="${LSD}/eval/molecule_nat_lang/pairs.csv"
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt molecule_nat_lang_mol_instructions gpt2 "$size") || continue
    if [[ -d "$CHEBI20_DIR" ]]; then
        ft_run molecule_nat_lang_mol_instructions chebi20 gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' CHEBI20_DIR='$CHEBI20_DIR' ARCH=gpt2 \
                     bash '${REPO_ROOT}/workflows/eval-chebi20.sh'"
    fi
    if [[ -d "$CHEMLLM_DIR" ]]; then
        ft_run molecule_nat_lang_mol_instructions chemllmbench gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' CHEMLLMBENCH_DIR='$CHEMLLM_DIR' \
                     ARCH=gpt2 SUBTASKS='name_conversion property_prediction' \
                     MAX_EXAMPLES=30 MAX_NEW_TOKENS=24 \
                     bash '${REPO_ROOT}/workflows/eval-chemllmbench.sh'"
    fi
    if [[ -f "$PAIRS_CSV" ]]; then
        ft_run molecule_nat_lang_mol_instructions molecule_nat_lang gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' PAIRS_CSV='$PAIRS_CSV' \
                     ARCH=gpt2 MAX_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-molecule-nat-lang.sh'"
    fi
done

# ---------------------------------------------------------------------------
# protein_sequence_proteingym × proteingym
# ---------------------------------------------------------------------------
OPSD_CSV="${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions/OPSD_HUMAN_Wan_2019.csv"
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt protein_sequence_proteingym gpt2 "$size") || continue
    if [[ -f "$OPSD_CSV" ]]; then
        ft_run protein_sequence_proteingym proteingym gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' PROTEINGYM_DATA='$OPSD_CSV' \
                     ARCH=gpt2 MODALITY=protein_sequence \
                     bash '${REPO_ROOT}/workflows/eval-proteingym.sh'"
    fi
done

# ---------------------------------------------------------------------------
# rna_celltype × {tabula_sapiens, rna_benchmark, replogle_perturb_seq}
# ---------------------------------------------------------------------------
RNA_JSONL="${LSD}/eval/rna_benchmark/cells.jsonl"
TABULA_JSONL="${LSD}/eval/tabula_sapiens/cells.jsonl"
REPLOGLE_DATA="${LSD}/eval/replogle_perturb_seq/replogle.csv"
for size in small medium large ex-large; do
    ckpt=$(resolve_finetune_ckpt rna_celltype gpt2 "$size") || continue
    if [[ -f "$RNA_JSONL" ]]; then
        ft_run rna_celltype rna_benchmark gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' RNA_JSONL='$RNA_JSONL' \
                     ARCH=gpt2 CELLS_PER_GROUP=4 \
                     bash '${REPO_ROOT}/workflows/eval-rna-benchmark.sh'"
    fi
    if [[ -f "$TABULA_JSONL" ]]; then
        ft_run rna_celltype tabula_sapiens gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' TABULA_JSONL='$TABULA_JSONL' \
                     ARCH=gpt2 MAX_CELLS=100 \
                     bash '${REPO_ROOT}/workflows/eval-tabula-sapiens.sh'"
    fi
    if [[ -f "$REPLOGLE_DATA" ]]; then
        ft_run rna_celltype replogle_perturb_seq gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' REPLOGLE_DATA='$REPLOGLE_DATA' \
                     ARCH=gpt2 \
                     bash '${REPO_ROOT}/workflows/eval-replogle-perturb-seq.sh'"
    fi
done

# ===========================================================================
# BERT fine-tune blocks (priority 中)
#
# bert ckpts are HF-format (no ckpt.pt) — that's fine for the HfMlmAdapter
# which loads pytorch_model.bin directly. Tasks restricted to those the
# encoder supports: encoder-probe (moleculenet, deeploc, tape, gue) and
# masked-likelihood (clinvar, gnomad, cosmic, omim, proteingym).
# Decoder-only tasks (moses, chebi20, generation-style chemllmbench) are
# skipped — they need ``adapter.generate``.
# ===========================================================================

# compounds_chembl × bert × {small, medium, large} × moleculenet
for size in small medium large; do
    ckpt=$(resolve_finetune_ckpt compounds_chembl bert "$size") || continue
    if [[ -d "$MOLECULENET_DIR" ]]; then
        ft_run compounds_chembl moleculenet bert "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     MOLECULENET_DIR='$MOLECULENET_DIR' \
                     ARCH=bert SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
    fi
done

# compounds_guacamol × bert × {medium, large} × moleculenet
for size in medium large; do
    ckpt=$(resolve_finetune_ckpt compounds_guacamol bert "$size") || continue
    if [[ -d "$MOLECULENET_DIR" ]]; then
        ft_run compounds_guacamol moleculenet bert "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     MOLECULENET_DIR='$MOLECULENET_DIR' \
                     ARCH=bert SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
    fi
done

# genome_sequence_clinvar × bert × {medium, large} ×
# {clinvar, gnomad_af_correlation, cosmic, omim, gue}
GUE_DIR="${LSD}/eval/gue"
for size in medium large; do
    ckpt=$(resolve_finetune_ckpt genome_sequence_clinvar bert "$size") || continue
    if [[ -f "$CLINVAR_DATA" ]]; then
        ft_run genome_sequence_clinvar clinvar bert "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     CLINVAR_DATA='$CLINVAR_DATA' ARCH=bert N_PER_CLASS=100 \
                     bash '${REPO_ROOT}/workflows/eval-clinvar.sh'"
    fi
    if [[ -f "$GNOMAD_DATA" ]]; then
        ft_run genome_sequence_clinvar gnomad_af_correlation bert "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     GNOMAD_DATA='$GNOMAD_DATA' ARCH=bert N_PER_BIN=100 \
                     bash '${REPO_ROOT}/workflows/eval-gnomad.sh'"
    fi
    if [[ -f "$COSMIC_DATA" ]]; then
        ft_run genome_sequence_clinvar cosmic bert "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     COSMIC_DATA='$COSMIC_DATA' ARCH=bert \
                     N_PER_CLASS=50 BOOTSTRAP_SAMPLES=$BOOTSTRAP \
                     bash '${REPO_ROOT}/workflows/eval-cosmic.sh'"
    fi
    if [[ -f "$OMIM_DATA" ]]; then
        ft_run genome_sequence_clinvar omim bert "$size" \
            bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                     --model-path '$ckpt' \
                     --arch bert --modality genome_sequence --device '$DEVICE' \
                     --omim-data '$OMIM_DATA' --output-dir \"\$OUTPUT_DIR\""
    fi
    if [[ -d "$GUE_DIR" ]]; then
        ft_run genome_sequence_clinvar gue bert "$size" \
            bash -c "MODEL_PATH='$ckpt' GUE_DIR='$GUE_DIR' \
                     TASKS='prom_300_all H3 tf_0' ARCH=bert \
                     bash '${REPO_ROOT}/workflows/eval-gue.sh'"
    fi
done

# protein_sequence_proteingym × bert × {medium, large} ×
# {proteingym, deeploc, tape}
DEEPLOC_DATA="${LSD}/eval/deeploc/deeploc.csv"
TAPE_DIR="${LSD}/eval/tape"
for size in medium large; do
    ckpt=$(resolve_finetune_ckpt protein_sequence_proteingym bert "$size") || continue
    if [[ -f "$OPSD_CSV" ]]; then
        ft_run protein_sequence_proteingym proteingym bert "$size" \
            bash -c "MODEL_PATH='$ckpt' PROTEINGYM_DATA='$OPSD_CSV' \
                     ARCH=bert MODALITY=protein_sequence \
                     bash '${REPO_ROOT}/workflows/eval-proteingym.sh'"
    fi
    if [[ -f "$DEEPLOC_DATA" ]]; then
        ft_run protein_sequence_proteingym deeploc bert "$size" \
            bash -c "MODEL_PATH='$ckpt' DEEPLOC_DATA='$DEEPLOC_DATA' \
                     ARCH=bert PREDICTIONS_PREVIEW_COUNT=4 \
                     bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
    fi
    if [[ -d "$TAPE_DIR" ]]; then
        ft_run protein_sequence_proteingym tape bert "$size" \
            bash -c "MODEL_PATH='$ckpt' TAPE_DIR='$TAPE_DIR' \
                     TASKS='fluorescence stability remote_homology secondary_structure_3 secondary_structure_8' \
                     ARCH=bert MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                     bash '${REPO_ROOT}/workflows/eval-tape.sh'"
    fi
done

# rna_celltype × bert × {medium, large} ×
# {rna_benchmark, tabula_sapiens, replogle_perturb_seq}
for size in medium large; do
    ckpt=$(resolve_finetune_ckpt rna_celltype bert "$size") || continue
    if [[ -f "$RNA_JSONL" ]]; then
        ft_run rna_celltype rna_benchmark bert "$size" \
            bash -c "MODEL_PATH='$ckpt' RNA_JSONL='$RNA_JSONL' \
                     ARCH=bert CELLS_PER_GROUP=4 \
                     bash '${REPO_ROOT}/workflows/eval-rna-benchmark.sh'"
    fi
    if [[ -f "$TABULA_JSONL" ]]; then
        ft_run rna_celltype tabula_sapiens bert "$size" \
            bash -c "MODEL_PATH='$ckpt' TABULA_JSONL='$TABULA_JSONL' \
                     ARCH=bert MAX_CELLS=100 \
                     bash '${REPO_ROOT}/workflows/eval-tabula-sapiens.sh'"
    fi
    if [[ -f "$REPLOGLE_DATA" ]]; then
        ft_run rna_celltype replogle_perturb_seq bert "$size" \
            bash -c "MODEL_PATH='$ckpt' REPLOGLE_DATA='$REPLOGLE_DATA' \
                     ARCH=bert \
                     bash '${REPO_ROOT}/workflows/eval-replogle-perturb-seq.sh'"
    fi
done

echo
echo "[ft-gapfill] all combos processed."
echo "[ft-gapfill] regenerate dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
