#!/usr/bin/env bash
# Gap-fill driver: run every (modality × arch × size × task) combo that has
# a pretrained checkpoint on disk but no metrics.json yet. Writes outputs
# to the canonical model-first layout, idempotent across re-runs (skips
# any combo whose REPORT.md already exists).
#
# Optional environment:
#   GAPFILL_MAX        - shared MAX_EXAMPLES (default 100; light load on CPU)
#   GAPFILL_DEVICE     - cuda/cpu (default cpu)
#   GAPFILL_BOOTSTRAP  - bootstrap CI resamples (default 30)
#   GAPFILL_FILTER     - regex restricting which "<mod>__<arch>__<size>__<task>"
#                        run ids are dispatched
#   GAPFILL_DRY_RUN    - "1" to print what would run and exit

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LSD="${LEARNING_SOURCE_DIR:-${REPO_ROOT}/learning_source_20260316}"
export LEARNING_SOURCE_DIR="${LSD}"

MAX="${GAPFILL_MAX:-100}"
DEVICE="${GAPFILL_DEVICE:-cpu}"
BOOTSTRAP="${GAPFILL_BOOTSTRAP:-30}"
FILTER="${GAPFILL_FILTER:-}"
DRY="${GAPFILL_DRY_RUN:-0}"

# Helper that calls one workflow with the gap-fill knobs and a unique RUNTAG.
# If the destination already has a REPORT.md (any depth), skip.
gap_run() {
    local modality="$1" task="$2" arch="$3" size="$4"
    local runtag="${task}_gapfill"
    shift 4

    local run_id="${modality}__${arch}__${size}__${task}"
    if [[ -n "$FILTER" ]] && [[ ! "$run_id" =~ ${FILTER} ]]; then
        return 0
    fi

    local slug="${modality}-${arch}-${size}"
    local outdir="${LSD%/}/experiment_data/eval/${slug}/${runtag}"
    # Skip if THIS gapfill leaf already has a REPORT.md (idempotent re-run).
    if find "$outdir" -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[gapfill] SKIP done: $run_id"
        return 0
    fi
    # Also skip if ANY sibling leaf under the model-slug parent already
    # holds an evaluation for this task (so historical runs like
    # ``proteingym_bert_opsd/`` count as coverage and we don't re-run
    # the same model × task combo).
    local parent="${LSD%/}/experiment_data/eval/${slug}"
    if [[ -d "$parent" ]]; then
        # Use POSIX character class because GNU grep ERE doesn't understand
        # \s. Without this the existing-coverage check silently fails to
        # match and the combo is needlessly re-run.
        if find "$parent" -mindepth 2 -name metrics.json 2>/dev/null \
                | xargs -I{} grep -lE "\"task\":[[:space:]]*\"${task}\"" {} 2>/dev/null \
                | grep -q .; then
            echo "[gapfill] SKIP existing-coverage: $run_id"
            return 0
        fi
    fi

    if [[ "$DRY" = "1" ]]; then
        echo "[gapfill] DRY:  $run_id -> $outdir"
        return 0
    fi

    echo "[gapfill] RUN:  $run_id -> $outdir"
    mkdir -p "$outdir"
    OUTPUT_DIR="$outdir" RUNTAG="$runtag" DEVICE="$DEVICE" \
        BOOTSTRAP="$BOOTSTRAP" MAX_EXAMPLES="$MAX" \
        "$@" 2>&1 | tail -3 || {
            echo "[gapfill] FAIL: $run_id" >&2
            return 1
        }
}

# Resolve a checkpoint path for a given (modality, arch, size).
# Returns 0 with stdout=path on success, 1 if no checkpoint found.
resolve_ckpt() {
    local modality="$1" arch="$2" size="$3"
    local out_dir="${LSD}/${modality}/${arch}-output/${modality}-${size}"
    # Some archs use the arch name (chemberta2 / dnabert2 / esm2 / rnaformer)
    if [[ ! -d "$out_dir" ]]; then
        out_dir="${LSD}/${modality}/${arch}-output/${arch}-${size}"
    fi
    [[ -d "$out_dir" ]] || return 1

    local ckpt="${out_dir}/ckpt.pt"
    if [[ -f "$ckpt" ]]; then
        echo "$ckpt"; return 0
    fi
    ckpt=$(ls -1d "${out_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    if [[ -d "$ckpt" ]]; then
        echo "$ckpt"; return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Genome × cosmic — bert / dnabert2 (each size).
# ---------------------------------------------------------------------------
COSMIC_CSV="${LSD}/eval/cosmic/cosmic_eval.csv"
if [[ -f "$COSMIC_CSV" ]]; then
    for arch in bert dnabert2; do
        for size in small medium large; do
            ckpt=$(resolve_ckpt genome_sequence "$arch" "$size") || continue
            gap_run genome_sequence cosmic "$arch" "$size" \
                bash -c "MODEL_PATH='$ckpt' \
                         COSMIC_DATA='$COSMIC_CSV' \
                         ARCH='$arch' MODALITY=genome_sequence N_PER_CLASS=50 PREDICTIONS_PREVIEW_COUNT=4 \
                         BOOTSTRAP_SAMPLES=$BOOTSTRAP \
                         bash '${REPO_ROOT}/workflows/eval-cosmic.sh'"
        done
    done
fi

# ---------------------------------------------------------------------------
# Genome × omim — bert / dnabert2 / gpt2 ex-large.
# ---------------------------------------------------------------------------
OMIM_CSV="${LSD}/eval/omim/omim_eval.csv"
if [[ -f "$OMIM_CSV" ]]; then
    for arch in bert dnabert2; do
        for size in small medium large; do
            ckpt=$(resolve_ckpt genome_sequence "$arch" "$size") || continue
            tokenizer="${LSD}/genome_sequence/spm_tokenizer.model"
            gap_run genome_sequence omim "$arch" "$size" \
                bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                         --model-path '$ckpt' \
                         ${arch:+--tokenizer-path '$tokenizer'} \
                         --arch '$arch' --modality genome_sequence --device '$DEVICE' \
                         --omim-data '$OMIM_CSV' \
                         --output-dir \"\$OUTPUT_DIR\""
        done
    done
    # gpt2 ex-large
    ckpt=$(resolve_ckpt genome_sequence gpt2 ex-large) && \
        gap_run genome_sequence omim gpt2 ex-large \
            bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                     --model-path '$ckpt' \
                     --tokenizer-path '${LSD}/genome_sequence/spm_tokenizer.model' \
                     --arch gpt2 --modality genome_sequence --device '$DEVICE' \
                     --omim-data '$OMIM_CSV' \
                     --output-dir \"\$OUTPUT_DIR\""
fi

# ---------------------------------------------------------------------------
# Genome ex-large gaps (clinvar / gnomad / gue).
# ---------------------------------------------------------------------------
ckpt=$(resolve_ckpt genome_sequence gpt2 ex-large) || ckpt=""
if [[ -n "$ckpt" ]]; then
    tokenizer="${LSD}/genome_sequence/spm_tokenizer.model"
    CLINVAR_DATA="${LSD}/eval/clinvar/clinvar.csv"
    [[ -f "$CLINVAR_DATA" ]] || CLINVAR_DATA="${LSD}/eval/clinvar/clinvar_evaluation.csv"
    [[ -f "$CLINVAR_DATA" ]] && \
        gap_run genome_sequence clinvar gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TOKENIZER_PATH='$tokenizer' \
                     CLINVAR_DATA='$CLINVAR_DATA' \
                     ARCH=gpt2 N_PER_CLASS=50 \
                     bash '${REPO_ROOT}/workflows/eval-clinvar.sh'"

    GNOMAD_DATA="${LSD}/eval/gnomad_af_correlation/gnomad_eval.csv"
    [[ -f "$GNOMAD_DATA" ]] && \
        gap_run genome_sequence gnomad_af_correlation gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TOKENIZER_PATH='$tokenizer' \
                     GNOMAD_DATA='$GNOMAD_DATA' \
                     ARCH=gpt2 N_PER_BIN=50 \
                     bash '${REPO_ROOT}/workflows/eval-gnomad.sh'"

    GUE_DIR="${LSD}/eval/gue"
    [[ -d "$GUE_DIR" ]] && \
        gap_run genome_sequence gue gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TOKENIZER_PATH='$tokenizer' \
                     GUE_DIR='$GUE_DIR' \
                     TASKS='prom_300_all H3 tf_0' \
                     ARCH=gpt2 \
                     bash '${REPO_ROOT}/workflows/eval-gue.sh'"
fi

# ---------------------------------------------------------------------------
# ProteinGym — every (arch, size) gap. Uses the OPSD assay which is the
# established matrix-bench testing assay.
# ---------------------------------------------------------------------------
OPSD_CSV="${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions/OPSD_HUMAN_Wan_2019.csv"
if [[ -f "$OPSD_CSV" ]]; then
    for arch in gpt2 bert esm2; do
        for size in small medium large ex-large; do
            ckpt=$(resolve_ckpt protein_sequence "$arch" "$size") || continue
            gap_run protein_sequence proteingym "$arch" "$size" \
                bash -c "MODEL_PATH='$ckpt' \
                         PROTEINGYM_DATA='$OPSD_CSV' \
                         ARCH='$arch' MODALITY=protein_sequence \
                         bash '${REPO_ROOT}/workflows/eval-proteingym.sh'"
        done
    done
fi

# ---------------------------------------------------------------------------
# Protein × deeploc / tape — gpt2 small + ex-large.
# ---------------------------------------------------------------------------
DEEPLOC_DATA="${LSD}/eval/deeploc/deeploc.csv"
TAPE_DIR="${LSD}/eval/tape"
for size in small ex-large; do
    ckpt=$(resolve_ckpt protein_sequence gpt2 "$size") || continue
    if [[ -f "$DEEPLOC_DATA" ]]; then
        gap_run protein_sequence deeploc gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     DEEPLOC_DATA='$DEEPLOC_DATA' \
                     ARCH=gpt2 PREDICTIONS_PREVIEW_COUNT=4 \
                     bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
    fi
    if [[ -d "$TAPE_DIR" ]]; then
        gap_run protein_sequence tape gpt2 "$size" \
            bash -c "MODEL_PATH='$ckpt' \
                     TAPE_DIR='$TAPE_DIR' \
                     TASKS='fluorescence stability remote_homology' \
                     ARCH=gpt2 PREDICTIONS_PREVIEW_COUNT=4 \
                     bash '${REPO_ROOT}/workflows/eval-tape.sh'"
    fi
done

# ---------------------------------------------------------------------------
# Protein × protein_foldability — gpt2 small.
# ---------------------------------------------------------------------------
PF_REF="${LSD}/eval/protein_foldability/pdb_seqres.txt"
ckpt=$(resolve_ckpt protein_sequence gpt2 small) || ckpt=""
if [[ -n "$ckpt" && -f "$PF_REF" ]]; then
    gap_run protein_sequence protein_foldability gpt2 small \
        bash -c "MODEL_PATH='$ckpt' \
                 REFERENCE_FASTA='$PF_REF' \
                 ARCH=gpt2 NUM_SAMPLES=50 MAX_NEW_TOKENS=128 \
                 bash '${REPO_ROOT}/workflows/eval-protein-foldability.sh'"
fi

# ---------------------------------------------------------------------------
# RNA × {benchmark, tabula_sapiens, replogle_perturb_seq} × gpt2 ex-large.
# ---------------------------------------------------------------------------
ckpt=$(resolve_ckpt rna gpt2 ex-large) || ckpt=""
if [[ -n "$ckpt" ]]; then
    if [[ -f "${LSD}/eval/rna_benchmark/cells.jsonl" ]]; then
        gap_run rna rna_benchmark gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     RNA_JSONL='${LSD}/eval/rna_benchmark/cells.jsonl' \
                     ARCH=gpt2 CELLS_PER_GROUP=4 \
                     bash '${REPO_ROOT}/workflows/eval-rna-benchmark.sh'"
    fi
    if [[ -f "${LSD}/eval/tabula_sapiens/cells.jsonl" ]]; then
        gap_run rna tabula_sapiens gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TABULA_JSONL='${LSD}/eval/tabula_sapiens/cells.jsonl' \
                     ARCH=gpt2 MAX_CELLS=100 \
                     bash '${REPO_ROOT}/workflows/eval-tabula-sapiens.sh'"
    fi
    if [[ -f "${LSD}/eval/replogle_perturb_seq/replogle.csv" ]]; then
        gap_run rna replogle_perturb_seq gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     REPLOGLE_DATA='${LSD}/eval/replogle_perturb_seq/replogle.csv' \
                     ARCH=gpt2 \
                     bash '${REPO_ROOT}/workflows/eval-replogle-perturb-seq.sh'"
    fi
fi

# ---------------------------------------------------------------------------
# Compounds × {moses, moleculenet} × gpt2 ex-large.
# ---------------------------------------------------------------------------
ckpt=$(resolve_ckpt compounds gpt2 ex-large) || ckpt=""
if [[ -n "$ckpt" ]]; then
    if [[ -d "${LSD}/eval/moses" ]]; then
        gap_run compounds moses gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TOKENIZER_PATH='${REPO_ROOT}/assets/molecules/vocab.txt' \
                     MOSES_DIR='${LSD}/eval/moses' \
                     NUM_SAMPLES=200 MAX_NEW_TOKENS=128 \
                     bash '${REPO_ROOT}/workflows/eval-moses.sh'"
    fi
    if [[ -d "${LSD}/eval/moleculenet" ]]; then
        gap_run compounds moleculenet gpt2 ex-large \
            bash -c "MODEL_PATH='$ckpt' \
                     TOKENIZER_PATH='${REPO_ROOT}/assets/molecules/vocab.txt' \
                     MOLECULENET_DIR='${LSD}/eval/moleculenet' \
                     ARCH=gpt2 SUBTASKS='bbbp esol' N_EXAMPLES=200 \
                     bash '${REPO_ROOT}/workflows/eval-moleculenet.sh'"
    fi
fi

# ---------------------------------------------------------------------------
# molecule_nat_lang × chebi20 × gpt2 ex-large.
# ---------------------------------------------------------------------------
ckpt=$(resolve_ckpt molecule_nat_lang gpt2 ex-large) || ckpt=""
if [[ -n "$ckpt" && -d "${LSD}/eval/chebi20" ]]; then
    gap_run molecule_nat_lang chebi20 gpt2 ex-large \
        bash -c "MODEL_PATH='$ckpt' \
                 CHEBI20_DIR='${LSD}/eval/chebi20' \
                 ARCH=gpt2 \
                 bash '${REPO_ROOT}/workflows/eval-chebi20.sh'"
fi

echo
echo "[gapfill] all combos processed."
echo "[gapfill] regenerate dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
