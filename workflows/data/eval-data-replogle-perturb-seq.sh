#!/usr/bin/env bash
# Materialise the Replogle Perturb-seq evaluator CSV from a public mirror.
#
# The original CellxGene H5AD URL has been returning 403 anonymously,
# and even when reachable the H5AD is single-cell (10+ GB, requires
# pseudobulk aggregation) which makes smoke runs impractical.
#
# The TruthSeq figshare release (10.6084/m9.figshare.31840141) ships
# the same Replogle 2022 K562 atlas as a 154 MB long-format parquet
# (knocked_down_gene, affected_gene, z_score, cell_line). prepare_csv
# pivots that into the (perturbation, baseline, perturbed) CSV the
# evaluator's load_replogle expects, with optional perturbation/gene
# subsampling for tractable runs.
#
# Optional environment:
#   REPLOGLE_PARQUET_URL   - figshare parquet URL
#                            (default: 154 MB TruthSeq release)
#   REPLOGLE_MAX_PERTURBATIONS - cap on # perturbations (default: unset)
#   REPLOGLE_MAX_GENES         - cap on # genes (default: unset)
#   REPLOGLE_CELL_LINE     - "K562" (the only one TruthSeq covers)
#   REPLOGLE_SOURCE        - bypass figshare; use a pre-downloaded parquet
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/replogle_perturb_seq/replogle_knockdown_effects.parquet
#   $LEARNING_SOURCE_DIR/eval/replogle_perturb_seq/replogle.csv
#   $LEARNING_SOURCE_DIR/eval/replogle_perturb_seq/manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init replogle_perturb_seq
dest_dir="$(ed_dest)"

PARQUET_URL="${REPLOGLE_PARQUET_URL:-https://ndownloader.figshare.com/files/63037363}"
PARQUET_LOCAL="${REPLOGLE_SOURCE:-${dest_dir}/replogle_knockdown_effects.parquet}"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.replogle_perturb_seq.prepare_csv
     --parquet-url "$PARQUET_URL"
     --parquet-path "$PARQUET_LOCAL"
     --output-csv "${dest_dir}/replogle.csv"
     --symbol-to-ensg-cache "${dest_dir}/symbol_to_ensg.csv")
if [[ -n "${REPLOGLE_CELL_LINE:-}" ]]; then
    cmd+=(--cell-line "$REPLOGLE_CELL_LINE")
fi
if [[ -n "${REPLOGLE_MAX_PERTURBATIONS:-}" ]]; then
    cmd+=(--max-perturbations "$REPLOGLE_MAX_PERTURBATIONS")
fi
if [[ -n "${REPLOGLE_MAX_GENES:-}" ]]; then
    cmd+=(--max-genes "$REPLOGLE_MAX_GENES")
fi
"${cmd[@]}"

ed_finalize_manifest \
    "Replogle Perturb-seq (TruthSeq mirror)" \
    "https://figshare.com/articles/dataset/_/31840141" \
    "see TruthSeq license" \
    "$(date -u +%Y%m%d)"

cat <<EOF

Next step:
  bash workflows/eval-replogle-perturb-seq.sh \\
       MODEL_PATH=<bert ckpt> \\
       REPLOGLE_DATA=${dest_dir}/replogle.csv \\
       MAX_EXAMPLES=200 BOOTSTRAP=30
EOF
