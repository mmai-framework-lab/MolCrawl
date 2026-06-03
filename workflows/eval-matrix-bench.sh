#!/usr/bin/env bash
# Drive a representative (evaluator × arch × size) matrix to populate the
# docs-src dashboard. Idempotent — skips combos whose REPORT.md already
# exists. Designed to run on CPU with modest MAX_EXAMPLES so the full
# matrix completes in roughly half an hour.
#
# Optional environment:
#   MATRIX_TAG       - tag prefixed onto each leaf (default v1). With the
#                      flattened layout, runs land at
#                      ``<eval>/<model-slug>/matrix_<TAG>_<task>_<arch>_<size>/``
#   MATRIX_MAX       - common MAX_EXAMPLES override (default per-task tuned)
#   MATRIX_DEVICE    - cuda / cpu (default cpu)
#   MATRIX_BOOTSTRAP - bootstrap CI resamples (default 30)
#   MATRIX_DRY_RUN   - set to 1 to print the combos and exit
#   MATRIX_FILTER    - regex applied to "<task>__<arch>__<size>" run id
#                      (e.g. ``MATRIX_FILTER=tape`` runs only tape rows)
#
# After completion, regenerate the dashboard:
#   python -m molcrawl.tasks.evaluation._dashboard \
#     --input-dir <repo>/experiment_data/eval \
#     --output docs-src/assets/data/evaluations.json \
#     --repo-root .

set -uo pipefail
# Note: no `-e`. We want one failing combo not to abort the rest of the matrix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LSD="${LEARNING_SOURCE_DIR:-${REPO_ROOT}/../learning_source_20260316}"
TAG="${MATRIX_TAG:-v1}"
DEVICE="${MATRIX_DEVICE:-cpu}"
BOOTSTRAP="${MATRIX_BOOTSTRAP:-30}"
DRY="${MATRIX_DRY_RUN:-0}"
FILTER="${MATRIX_FILTER:-}"

# Matrix runs live under LEARNING_SOURCE_DIR (canonical eval root),
# directly under ``<modality>-<arch>-<size>/`` like every other run, with
# a ``matrix_<TAG>_`` prefix on the leaf so the batch is still
# identifiable inline. This keeps the model-first principle consistent
# (no separate ``matrix_<TAG>/`` wrapper directory).
EVAL_BASE="${LSD}/experiment_data/eval"
LEAF_PREFIX="matrix_${TAG}_"
mkdir -p "${EVAL_BASE}"

run_one() {
    # Args: modality task arch size leaf_basename cmd...
    local modality="$1" task="$2" arch="$3" size="$4" leaf="$5"
    shift 5

    local run_id="${task}__${arch}__${size}"
    if [[ -n "${FILTER}" ]] && [[ ! "${run_id}" =~ ${FILTER} ]]; then
        return 0
    fi

    local slug="${modality}-${arch}-${size}"
    local outdir="${EVAL_BASE}/${slug}/${LEAF_PREFIX}${leaf}"
    # gue / tape wrappers nest one level deeper (per-sub-task subdir),
    # so we look for *any* REPORT.md at depth ≤ 2 rather than only at
    # the top of ${outdir}.
    if [[ -d "${outdir}" ]] && find "${outdir}" -maxdepth 2 -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[matrix] SKIP (already done): ${run_id}"
        return 0
    fi

    if [[ "${DRY}" = "1" ]]; then
        echo "[matrix] DRY: ${run_id}"
        return 0
    fi

    echo "[matrix] RUN: ${run_id} -> ${outdir}"
    mkdir -p "${outdir}"
    OUTPUT_DIR="${outdir}" DEVICE="${DEVICE}" BOOTSTRAP="${BOOTSTRAP}" \
        "$@" 2>&1 | tail -3 || {
        echo "[matrix] FAILED: ${run_id}" >&2
        return 1
    }
}

export LEARNING_SOURCE_DIR="${LSD}"

#######################################
# Compounds: chembl_scaffold_heldout
#######################################
for size in small medium large ex-large; do
    ckpt="${LSD}/compounds_chembl/gpt2-output/compounds_chembl-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    run_one compounds chembl_scaffold_heldout gpt2 "${size}" "chembl_scaffold_heldout_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${REPO_ROOT}/assets/molecules/vocab.txt' \
                 HELDOUT_CSV='${LSD}/eval/chembl_scaffold_heldout/heldout.csv' \
                 MAX_EXAMPLES=200 SEED=42 PREDICTIONS_PREVIEW_COUNT=10 \
                 bash '${REPO_ROOT}/workflows/eval-chembl-heldout.sh'"
done

#######################################
# RNA: rna_benchmark
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/rna/bert-output/rna-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    # CELLS_PER_GROUP=4 (5 tissues × 4 = 20 cells, each 256 tokens)
    # keeps PLL tractable on CPU for bert-medium.
    run_one rna rna_benchmark bert "${size}" "rna_benchmark_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${LSD}/rna/custom_tokenizer_bert' \
                 RNA_JSONL='${LSD}/eval/rna_benchmark/cells.jsonl' \
                 ARCH=bert CELLS_PER_GROUP=4 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-rna-benchmark.sh'"
done

#######################################
# RNA: tabula_sapiens
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/rna/bert-output/rna-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one rna tabula_sapiens bert "${size}" "tabula_sapiens_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${LSD}/rna/custom_tokenizer_bert' \
                 TABULA_JSONL='${LSD}/eval/tabula_sapiens/cells.jsonl' \
                 ARCH=bert MAX_CELLS=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-tabula-sapiens.sh'"
done

#######################################
# RNA: replogle_perturb_seq
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/rna/bert-output/rna-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one rna replogle_perturb_seq bert "${size}" "replogle_perturb_seq_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${LSD}/rna/custom_tokenizer_bert' \
                 REPLOGLE_DATA='${LSD}/eval/replogle_perturb_seq/replogle.csv' \
                 ARCH=bert MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-replogle-perturb-seq.sh'"
done

#######################################
# RNA: gpt2 cross-arch — rna_benchmark (likelihood) + tabula_sapiens
# (encoder-probe via decoder hidden states) + replogle_perturb_seq
#
# The gpt2 adapter exposes both score_likelihood (for rna_benchmark)
# and embed (for the encoder probes). The rna gpt2 ckpts share the
# gene-id vocabulary so the same JSONL inputs the bert variants use
# work unchanged.
#######################################
for size in small medium large; do
    ckpt="${LSD}/rna/gpt2-output/rna-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    run_one rna rna_benchmark gpt2 "${size}" "rna_benchmark_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 RNA_JSONL='${LSD}/eval/rna_benchmark/cells.jsonl' \
                 ARCH=gpt2 CELLS_PER_GROUP=4 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-rna-benchmark.sh'"
    run_one rna tabula_sapiens gpt2 "${size}" "tabula_sapiens_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TABULA_JSONL='${LSD}/eval/tabula_sapiens/cells.jsonl' \
                 ARCH=gpt2 MAX_CELLS=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-tabula-sapiens.sh'"
    run_one rna replogle_perturb_seq gpt2 "${size}" "replogle_perturb_seq_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 REPLOGLE_DATA='${LSD}/eval/replogle_perturb_seq/replogle.csv' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-replogle-perturb-seq.sh'"
done

#######################################
# Protein: protein_foldability — gpt2 small was the only size missing
# from the existing matrix (protein_foldability_phase1b/medium/large/
# ex-large already exist outside matrix_v1). Backfill it here so the
# small/medium/large/ex-large quartet shows up in the dashboard's
# scaling view.
#######################################
pf_ref="${LSD}/eval/protein_foldability/pdb_seqres.txt"
if [[ -f "$pf_ref" ]]; then
    for size in small; do
        # Only run when a top-level ckpt.pt exists — the wrapper's
        # generate-from-checkpoint code expects a file, not the
        # ``checkpoint-XXXX/`` HF-format directory.
        ckpt="${LSD}/protein_sequence/gpt2-output/protein_sequence-${size}/ckpt.pt"
        [[ -f "$ckpt" ]] || continue
        run_one protein_sequence protein_foldability gpt2 "${size}" "protein_foldability_gpt2_${size}" \
            bash -c "MODEL_PATH='${ckpt}' \
                     REFERENCE_FASTA='${pf_ref}' \
                     ARCH=gpt2 NUM_SAMPLES=200 MAX_NEW_TOKENS=128 SEED=42 \
                     PREDICTIONS_PREVIEW_COUNT=10 \
                     bash '${REPO_ROOT}/workflows/eval-protein-foldability.sh'"
    done
fi

#######################################
# Protein: deeploc
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/protein_sequence/esm2-output/esm2-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one protein_sequence deeploc esm2 "${size}" "deeploc_esm2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 DEEPLOC_DATA='${LSD}/eval/deeploc/deeploc.csv' \
                 ARCH=esm2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
done

# DeepLoc × bert / gpt2 protein cross-arch (encoder probe).
for size in small medium large; do
    ckpt_dir="${LSD}/protein_sequence/bert-output/protein_sequence-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one protein_sequence deeploc bert "${size}" "deeploc_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 DEEPLOC_DATA='${LSD}/eval/deeploc/deeploc.csv' \
                 ARCH=bert MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
done
for size in small medium large; do
    ckpt="${LSD}/protein_sequence/gpt2-output/protein_sequence-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    run_one protein_sequence deeploc gpt2 "${size}" "deeploc_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 DEEPLOC_DATA='${LSD}/eval/deeploc/deeploc.csv' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-deeploc.sh'"
done

#######################################
# Genome: GUE (3 representative subtasks)
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/genome_sequence/dnabert2-output/dnabert2-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one genome_sequence gue dnabert2 "${size}" "gue_dnabert2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 GUE_DIR='${LSD}/eval/gue' \
                 TASKS='prom_300_all H3 tf_0' \
                 ARCH=dnabert2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-gue.sh'"
done

# GUE × bert genome cross-arch (encoder probe — bert has the same
# embedding capability as dnabert2)
for size in small medium large; do
    ckpt_dir="${LSD}/genome_sequence/bert-output/genome_sequence-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one genome_sequence gue bert "${size}" "gue_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 GUE_DIR='${LSD}/eval/gue' \
                 TASKS='prom_300_all H3 tf_0' \
                 ARCH=bert MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-gue.sh'"
done

# GUE × gpt2 genome cross-arch (decoder used as a feature extractor;
# probes the post-ln_f hidden states like the existing TAPE / DeepLoc
# evaluators do).
for size in small medium large; do
    ckpt="${LSD}/genome_sequence/gpt2-output/genome_sequence-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    tokenizer="${LSD}/genome_sequence/spm_tokenizer.model"
    run_one genome_sequence gue gpt2 "${size}" "gue_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${tokenizer}' \
                 GUE_DIR='${LSD}/eval/gue' \
                 TASKS='prom_300_all H3 tf_0' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-gue.sh'"
done

#######################################
# Protein: TAPE (regression + classification + sequence-labeling +
#               contact_prediction now wired via proteinglm mirror)
#######################################
for size in small medium large; do
    ckpt_dir="${LSD}/protein_sequence/esm2-output/esm2-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one protein_sequence tape esm2 "${size}" "tape_esm2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TAPE_DIR='${LSD}/eval/tape' \
                 TASKS='fluorescence stability remote_homology secondary_structure_3 secondary_structure_8 contact_prediction' \
                 ARCH=esm2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 CONTACT_PAIRS_PER_PROTEIN=30 CONTACT_MIN_SEPARATION=24 \
                 bash '${REPO_ROOT}/workflows/eval-tape.sh'"
done

# TAPE × bert protein cross-arch — bert encoder has both ``embed`` and
# ``embed_per_residue``, so all six TAPE sub-tasks are valid.
for size in small medium large; do
    ckpt_dir="${LSD}/protein_sequence/bert-output/protein_sequence-${size}"
    ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
    [[ -d "$ckpt" ]] || continue
    run_one protein_sequence tape bert "${size}" "tape_bert_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TAPE_DIR='${LSD}/eval/tape' \
                 TASKS='fluorescence stability remote_homology secondary_structure_3 secondary_structure_8 contact_prediction' \
                 ARCH=bert MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 CONTACT_PAIRS_PER_PROTEIN=30 CONTACT_MIN_SEPARATION=24 \
                 bash '${REPO_ROOT}/workflows/eval-tape.sh'"
done

# TAPE × gpt2 protein cross-arch — decoder lacks ``embed_per_residue``,
# so the SS3 / SS8 / contact_prediction sub-tasks would error out. Cap
# the TASKS list to the three encoder-probe sub-tasks.
for size in small medium large; do
    ckpt="${LSD}/protein_sequence/gpt2-output/protein_sequence-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    run_one protein_sequence tape gpt2 "${size}" "tape_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TAPE_DIR='${LSD}/eval/tape' \
                 TASKS='fluorescence stability remote_homology' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=4 \
                 bash '${REPO_ROOT}/workflows/eval-tape.sh'"
done

#######################################
# Protein: TAPE contact_prediction — separate run_id so the matrix
# runner's depth-2 REPORT.md skip doesn't lump it in with the existing
# tape_esm2_<size> directory, which already has 5 sub-task REPORT.md.
#######################################
cp_dir="${LSD}/eval/tape/contact_prediction"
if [[ -f "${cp_dir}/contact_prediction_train.json" ]]; then
    for size in small medium large; do
        ckpt_dir="${LSD}/protein_sequence/esm2-output/esm2-${size}"
        ckpt=$(ls -1d "${ckpt_dir}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
        [[ -d "$ckpt" ]] || continue
        run_one protein_sequence tape_contact_prediction esm2 "${size}" "tape_contact_prediction_esm2_${size}" \
            bash -c "MODEL_PATH='${ckpt}' \
                     TAPE_DIR='${LSD}/eval/tape' \
                     TASKS='contact_prediction' \
                     ARCH=esm2 MAX_EXAMPLES=20 PREDICTIONS_PREVIEW_COUNT=4 \
                     CONTACT_PAIRS_PER_PROTEIN=30 CONTACT_MIN_SEPARATION=24 \
                     bash '${REPO_ROOT}/workflows/eval-tape.sh'"
    done
fi

#######################################
# Genome: COSMIC (newly unblocked via Sanger SPA reverse-engineering;
#                 足固め with bootstrap CI + tier-stratified sampling)
#######################################
cosmic_csv="${LSD}/eval/cosmic/cosmic_eval.csv"
if [[ -f "${cosmic_csv}" ]]; then
    for size in small medium large ex-large; do
        ckpt="${LSD}/genome_sequence/gpt2-output/genome_sequence-${size}/ckpt.pt"
        [[ -f "$ckpt" ]] || continue
        tokenizer="${LSD}/genome_sequence/spm_tokenizer.model"
        run_one genome_sequence cosmic gpt2 "${size}" "cosmic_gpt2_${size}" \
            bash -c "MODEL_PATH='${ckpt}' \
                     TOKENIZER_PATH='${tokenizer}' \
                     COSMIC_DATA='${cosmic_csv}' \
                     ARCH=gpt2 N_PER_CLASS=50 PREDICTIONS_PREVIEW_COUNT=8 \
                     BOOTSTRAP_SAMPLES=100 \
                     bash '${REPO_ROOT}/workflows/eval-cosmic.sh'"
    done
else
    echo "[matrix] cosmic: ${cosmic_csv} missing — run prepare_csv first" >&2
fi

#######################################
# ChemLLMBench (7/9 sub-tasks live): scored against the molecule_nat_lang
# decoder. We pin generation budgets per sub-task so the long reaction
# SMILES of yield_prediction don't dominate wall time.
#######################################
chemllm_dir="${LSD}/eval/chemllmbench"
if [[ -d "${chemllm_dir}" ]] && ls "${chemllm_dir}"/*.jsonl >/dev/null 2>&1; then
    # large/ex-large take 60-90 min per size on CPU (medium already
    # burns ~10 min per generation-heavy sub-task) and ex-large goes
    # OOM at 1.3 B params on a 64-GB CPU node. Run them through the
    # GPU sbatch (workflows/eval-gpu-gapfill.sbatch) instead. The
    # matrix runner skips combos whose REPORT.md is already on disk,
    # so the GPU-produced runs land safely in matrix_v1.
    for size in small medium large ex-large; do
        ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-${size}/ckpt.pt"
        [[ -f "$ckpt" ]] || continue
        run_one molecule_nat_lang chemllmbench gpt2 "${size}" "chemllmbench_gpt2_${size}" \
            bash -c "MODEL_PATH='${ckpt}' \
                     CHEMLLMBENCH_DIR='${chemllm_dir}' \
                     SUBTASKS='molecule_captioning molecule_design reaction_prediction name_conversion retrosynthesis yield_prediction property_prediction' \
                     ARCH=gpt2 MAX_EXAMPLES=30 \
                     MAX_NEW_TOKENS=24 PREDICTIONS_PREVIEW_COUNT=8 \
                     bash '${REPO_ROOT}/workflows/eval-chemllmbench.sh'"
    done
else
    echo "[matrix] chemllmbench: ${chemllm_dir} empty — run eval-data-chemllmbench.sh first" >&2
fi

#######################################
# Molecule_nat_lang (likelihood mode only — generation modes are noisy)
#######################################
for size in small medium large ex-large; do
    ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-${size}/ckpt.pt"
    [[ -f "$ckpt" ]] || continue
    pairs="${LSD}/eval/molecule_nat_lang/pairs.csv"
    [[ -f "$pairs" ]] || continue
    run_one molecule_nat_lang molecule_nat_lang gpt2 "${size}" "molecule_nat_lang_gpt2_${size}" \
        bash -c "MODEL_PATH='${ckpt}' \
                 PAIRS_CSV='${pairs}' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash '${REPO_ROOT}/workflows/eval-molecule-nat-lang.sh'"
done

#######################################
# chebi20 + omim — fast smokes, want at least small+medium for shape
#######################################
chebi_dir="${LSD}/eval/chebi20"
if [[ -d "${chebi_dir}" ]] && [[ -f "${chebi_dir}/test.txt" ]]; then
    for size in small medium large; do
        ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-${size}/ckpt.pt"
        [[ -f "$ckpt" ]] || continue
        run_one molecule_nat_lang chebi20 gpt2 "${size}" "chebi20_gpt2_${size}" \
            bash -c "MODEL_PATH='${ckpt}' \
                     CHEBI20_DIR='${chebi_dir}' \
                     ARCH=gpt2 MAX_EXAMPLES=10 \
                     bash '${REPO_ROOT}/workflows/eval-chebi20.sh'"
    done
fi

omim_csv="${LSD}/eval/omim/omim_eval.csv"
if [[ -f "${omim_csv}" ]]; then
    for size in small medium large; do
        ckpt="${LSD}/genome_sequence/gpt2-output/genome_sequence-${size}/ckpt.pt"
        [[ -f "$ckpt" ]] || continue
        tokenizer="${LSD}/genome_sequence/spm_tokenizer.model"
        # No eval-omim.sh wrapper exists yet (the productionised
        # prepare_csv lives behind the OMIM API key approval), so we
        # invoke the python module directly. ``run_one`` exports
        # OUTPUT_DIR; we double-quote the inner bash -c body so it
        # expands inside the child shell rather than at composition time.
        run_one genome_sequence omim gpt2 "${size}" "omim_gpt2_${size}" \
            bash -c "\"$PYTHON\" -m molcrawl.tasks.evaluation.omim \
                     --model-path '${ckpt}' \
                     --tokenizer-path '${tokenizer}' \
                     --arch gpt2 --modality genome_sequence --device cpu \
                     --omim-data '${omim_csv}' \
                     --output-dir \"\$OUTPUT_DIR\""
    done
fi

echo
echo "[matrix] all combos processed under ${EVAL_BASE}/<model-slug>/${LEAF_PREFIX}<task>_<arch>_<size>/"
echo "[matrix] now rebuild the dashboard:"
echo "  bash ${REPO_ROOT}/workflows/eval-build-dashboard.sh"
