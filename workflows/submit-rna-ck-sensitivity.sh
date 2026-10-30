#!/bin/bash
# The 3-point checkpoint-sensitivity measurement on the two rna BERT runs whose
# downstream probe read a checkpoint that is not the eval_loss_mask minimum.
# all-checkpoint-metric-judgment-2026-10-30-v3 §9.
#
# Three points per run: the checkpoint the 2026-10-22 probe read (115,300), the step
# the judged metric actually bottomed at, and the terminal step. The first and third
# exist; the middle one has to be produced, because the original run evaluated that
# step without saving it.
#
#   rna-small-lr3e4   115,300 (2.695726) · 119,200 (2.692825) · 120,960
#   rna-medium-lr3e4  115,300 (2.638232) · 116,100 (2.636773) · 120,960
#
# Why it is worth measuring once: the 42 production BERT runs that do not match are
# spread over mask gaps of 0.000009 to 0.003977, and the only statement available
# about what that does downstream is an extrapolation -- small and medium differ by
# 0.0575 in mask and 0.0111 in probe accuracy, which scales 0.0029 to 0.0006, a sixth
# of the 95% bootstrap half-width (±0.0036). Measured once, the extrapolation can be
# retired.
#
# DRY RUN BY DEFAULT. Prints what it would submit and exits; DRY_RUN=no submits.
# Nothing here is launched by the branch that adds it.
#
#   DRY_RUN=no bash workflows/submit-rna-ck-sensitivity.sh
set -eo pipefail

DRY_RUN="${DRY_RUN:-yes}"

# Roots. Unset-fatal, and the output root is the MAIN CHECKOUT's run tree: these are
# production artifacts and a worktree is disposable (the genome BERT sweep put 102G of
# checkpoints in one). Pass it explicitly rather than deriving it from $PWD.
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna input root (the tree holding rna/training_ready_hf_dataset)}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to learning_source_rna_runs in the main checkout}"
: "${EVAL_JSONL:?set EVAL_JSONL to the Tabula Sapiens evaluation JSONL}"
: "${TOKENIZER_DIR:?set TOKENIZER_DIR to the rna BERT tokenizer directory}"
: "${PROBE_ROOT:?set PROBE_ROOT to the directory the probe results should go under}"

BERT_OUT="${MODEL_OUTPUT_ROOT}/rna/bert-output"
CFG=molcrawl/tasks/pretrain/configs/rna

echo "=== outputs: ${BERT_OUT} (training) / ${PROBE_ROOT} (probes) ==="
echo "dry run: ${DRY_RUN}"

run() {
    echo "+ $*"
    if [ "${DRY_RUN}" = "no" ]; then "$@"; fi
}

# 1. Seed the two re-run directories with the checkpoint each resume starts from.
for pair in "small:119200" "medium:116100"; do
    size="${pair%%:*}"; step="${pair##*:}"
    run sbatch --parsable \
        --export=ALL,SRC_RUN="${BERT_OUT}/rna-${size}-lr3e4",DEST_RUN="${BERT_OUT}/rna-${size}-lr3e4-ck${step}",STEP=115300 \
        workflows/rna-ck-sensitivity-prepare.sbatch
done

# 2. Resume each one and stop at the judged minimum. SEGMENT=2 because the directory
# already holds a checkpoint -- that is the whole point of step 1, and SEGMENT=1 would
# refuse. The config carries max_steps 120,960 and stop_at_step; only the operational
# values are passed here.
for pair in "small:119200" "medium:116100"; do
    size="${pair%%:*}"; step="${pair##*:}"
    run sbatch --parsable \
        --export=ALL,MODALITY=rna,LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}",CONFIG="${CFG}/bert_${size}_lr3e4_ck${step}.py",SEGMENT=2 \
        workflows/bert-grid.sbatch
done

# 3. The six probes. The 115,300 pair reproduces the 2026-10-22 numbers (0.8361 and
# 0.8472 accuracy), which is the control: a probe re-run on the same weights has to
# land inside its own bootstrap interval before a difference between checkpoints means
# anything.
for pair in "small:119200" "medium:116100"; do
    size="${pair%%:*}"; step="${pair##*:}"
    for ck in "${BERT_OUT}/rna-${size}-lr3e4/checkpoint-115300" \
              "${BERT_OUT}/rna-${size}-lr3e4-ck${step}/checkpoint-${step}" \
              "${BERT_OUT}/rna-${size}-lr3e4/checkpoint-120960"; do
        name="$(basename "$(dirname "${ck}")")-$(basename "${ck}")"
        run sbatch --parsable \
            --export=ALL,MODEL_PATH="${ck}",PROBE_ARCH=bert,EVAL_JSONL="${EVAL_JSONL}",TOKENIZER_DIR="${TOKENIZER_DIR}",PROBE_OUT_DIR="${PROBE_ROOT}/${name}" \
            workflows/rna-celltype-probe.sbatch
    done
done

echo
echo "The probes in step 3 read weights step 2 has not written yet. Submit 1 and 2,"
echo "wait for them, then submit 3 -- or add --dependency to the probe submissions."
