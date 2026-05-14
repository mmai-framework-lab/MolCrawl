#!/usr/bin/env bash
# Download the in-repo ChemLLMBench source files and convert them to the
# {prompt, answer, metadata} JSONL the evaluator expects.
#
# Coverage (7 of 9 sub-tasks — all backed by public GitHub artefacts):
#   molecule_captioning, molecule_design, reaction_prediction
#   name_conversion, retrosynthesis, yield_prediction, property_prediction
#
# Not yet wired (no obvious upstream artefact identified):
#   text_guided_generation, smiles_understanding
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/chemllmbench/<task>.jsonl
#   $LEARNING_SOURCE_DIR/eval/chemllmbench/<task>_source.{csv,npz}
#   $LEARNING_SOURCE_DIR/eval/chemllmbench/property_prediction_source/
#       BBBP_test.csv, BACE_test.csv, ClinTox_test.csv, HIV_test.csv, Tox_test.csv
#   manifest.json
#
# License: see https://github.com/ChemFoundationModels/ChemLLMBench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init chemllmbench

BASE="${CHEMLLMBENCH_BASE:-https://raw.githubusercontent.com/ChemFoundationModels/ChemLLMBench/main/data}"

# ---------------------------------------------------------------------
# 1) Single-file CSV tasks: download → convert
# ---------------------------------------------------------------------
SINGLE_FILE_TASKS=(
    "molecule_captioning|molecule_captioning/molecule_captioning_test.csv|csv"
    "molecule_design|molecule_design/molecule_design_test.csv|csv"
    "reaction_prediction|reaction_prediction/uspto_test.csv|csv"
    "name_conversion|name_prediction/llm_test.csv|csv"
    "retrosynthesis|retro/uspto50k_retro_test.csv|csv"
)

for entry in "${SINGLE_FILE_TASKS[@]}"; do
    task="${entry%%|*}"
    rest="${entry#*|}"
    rel_path="${rest%%|*}"
    ext="${rest##*|}"
    src_name="${task}_source.${ext}"
    ed_download "${BASE}/${rel_path}" "${src_name}"
    "$PYTHON" -m molcrawl.tasks.evaluation.chemllmbench.prepare_jsonl \
        --task "$task" \
        --source-csv "$(ed_dest)/${src_name}" \
        --output-jsonl "$(ed_dest)/${task}.jsonl"
done

# ---------------------------------------------------------------------
# 2) yield_prediction: NPZ binary (BH = Buchwald-Hartwig sample-100)
# ---------------------------------------------------------------------
ed_download "${BASE}/yield_prediction/BH_sample_100_test.npz" "yield_prediction_source.npz"
"$PYTHON" -m molcrawl.tasks.evaluation.chemllmbench.prepare_jsonl \
    --task yield_prediction \
    --source-csv "$(ed_dest)/yield_prediction_source.npz" \
    --output-jsonl "$(ed_dest)/yield_prediction.jsonl"

# ---------------------------------------------------------------------
# 3) property_prediction: union of 5 small per-dataset CSVs
# ---------------------------------------------------------------------
PROP_DIR_REL="property_prediction_source"
mkdir -p "$(ed_dest)/${PROP_DIR_REL}"
for f in BBBP_test.csv BACE_test.csv ClinTox_test.csv HIV_test.csv Tox_test.csv; do
    ed_download "${BASE}/property_prediction/${f}" "${PROP_DIR_REL}/${f}"
done
"$PYTHON" -m molcrawl.tasks.evaluation.chemllmbench.prepare_jsonl \
    --task property_prediction \
    --source-csv "$(ed_dest)/${PROP_DIR_REL}" \
    --output-jsonl "$(ed_dest)/property_prediction.jsonl"

ed_finalize_manifest \
    "ChemLLMBench" \
    "https://github.com/ChemFoundationModels/ChemLLMBench" \
    "see ChemLLMBench license" \
    "$(date -u +%Y%m%d)"
