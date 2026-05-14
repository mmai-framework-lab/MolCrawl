#!/usr/bin/env bash
# Materialise TAPE sub-task JSONL files from public HuggingFace mirrors.
#
# The original songlabdata S3 bucket has returned 403 anonymously since
# 2025, so this workflow uses public mirrors instead:
#
#   - AI4Protein/TAPE_Fluorescence              (regression)
#   - AI4Protein/TAPE_Stability                 (regression)
#   - proteinea/remote_homology                 (classification, 1195 fold classes)
#
# All three are gated=False on HuggingFace; no token required.
#
# Sub-tasks NOT covered here (need extra wiring beyond a public mirror):
#   - secondary_structure_3 / secondary_structure_8 — per-residue
#     labels; the evaluator's current probe head is sequence-level only.
#   - contact_prediction — needs residue-residue logits + PDB masking.
#
# Override TAPE_TASKS to limit which sub-tasks get materialised.
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/tape/<task>/<task>_<split>.json  (jsonl)
#   $LEARNING_SOURCE_DIR/eval/tape/manifest.json
#
# License: see https://github.com/songlab-cal/tape (BSD-3-Clause).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init tape
dest_dir="$(ed_dest)"

TAPE_TASKS="${TAPE_TASKS:-fluorescence stability remote_homology}"

"$PYTHON" -m molcrawl.tasks.evaluation.tape.prepare_csv \
    --output-dir "${dest_dir}" \
    --tasks ${TAPE_TASKS}

ed_finalize_manifest \
    "TAPE (public-mirror release)" \
    "https://github.com/songlab-cal/tape" \
    "BSD-3-Clause" \
    "$(date -u +%Y%m%d)"

cat <<EOF

Next step:
  bash workflows/eval-tape.sh \\
       MODEL_PATH=<esm2 ckpt> TAPE_DIR=${dest_dir} \\
       TASKS="fluorescence" MAX_EXAMPLES=300 BOOTSTRAP=30
EOF
