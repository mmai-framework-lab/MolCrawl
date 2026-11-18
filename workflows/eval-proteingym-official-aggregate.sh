#!/usr/bin/env bash
# Run the official ProteinGym DMS zero-shot aggregation on MolCrawl scores (Issue #388, M2).
#
# Given per-assay prediction CSVs already emitted under SCORES/<model>/<DMS_id>.csv (by
# scripts/proteingym_emit_official.py) and a config that registers those models (by
# scripts/proteingym_prepare_config.py), this calls the official merge.py then
# performance_DMS_benchmarks.py -- the official metrics and hierarchical averaging, not a
# re-implementation. Output goes under RESULTS.
#
#   LEARNING_SOURCE_DIR=<...> \
#   MUTATION_TYPE=substitutions \
#   SCORES=<base>/scores/zero_shot/substitutions \
#   bash workflows/eval-proteingym-official-aggregate.sh
#
# MUTATION_TYPE=indels switches to --indel_mode and the indel reference/data.
set -euo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR}"
MUTATION_TYPE="${MUTATION_TYPE:-substitutions}"
BASE="${PG_BASE_DIR:-${LEARNING_SOURCE_DIR%/}/eval/proteingym/official}"
PG_REPO="${BASE}/vendor/ProteinGym"
CONFIG="${CONFIG:-${BASE}/configs/molcrawl.json}"
SCORES="${SCORES:-${BASE}/scores/zero_shot/${MUTATION_TYPE}}"
RESULTS="${RESULTS:-${BASE}/results/zero_shot/${MUTATION_TYPE}}"

# Resolve a Python with the scientific stack (no bare `python` on this cluster).
if [ -z "${PYTHON:-}" ]; then
    _root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${_root}" ] && [ -x "${_root}/miniconda/bin/python" ]; then
        PYTHON="${_root}/miniconda/bin/python"
    else
        PYTHON="python3"
    fi
fi

if [ "${MUTATION_TYPE}" = "indels" ]; then
    DATA_DIR="${BASE}/data/DMS_ProteinGym_indels"
    REF="${BASE}/refs/DMS_indels.csv"
    PERF_FLAG="--indel_mode"
    MERGE_TYPE="indels"
else
    DATA_DIR="${BASE}/data/DMS_ProteinGym_substitutions"
    REF="${BASE}/refs/DMS_substitutions.csv"
    PERF_FLAG="--performance_by_depth"
    MERGE_TYPE="substitutions"
fi

for p in "${PG_REPO}/proteingym/merge.py" "${CONFIG}" "${DATA_DIR}" "${REF}"; do
    [ -e "${p}" ] || { echo "ERROR: missing ${p}" >&2; exit 1; }
done
mkdir -p "${RESULTS}" "${SCORES}"

export PYTHONPATH="${PG_REPO}:${PYTHONPATH:-}"
echo "=== ProteinGym official aggregation (${MUTATION_TYPE}) @ $(date -Is) ==="
echo "scores: ${SCORES}"
echo "config: ${CONFIG}"

"${PYTHON}" "${PG_REPO}/proteingym/merge.py" \
    --DMS_assays_location "${DATA_DIR}" \
    --model_scores_location "${SCORES}" \
    --merged_scores_dir merged \
    --mutation_type "${MERGE_TYPE}" --dataset DMS \
    --DMS_reference_file "${REF}" \
    --config_file "${CONFIG}"

"${PYTHON}" "${PG_REPO}/proteingym/performance_DMS_benchmarks.py" \
    --input_scoring_files_folder "${SCORES}/merged" \
    --output_performance_file_folder "${RESULTS}" \
    --DMS_reference_file_path "${REF}" \
    --DMS_data_folder "${DATA_DIR}" \
    --config_file "${CONFIG}" \
    ${PERF_FLAG}

echo "=== done; results under ${RESULTS} @ $(date -Is) ==="
