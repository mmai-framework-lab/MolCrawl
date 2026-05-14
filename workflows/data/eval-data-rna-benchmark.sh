#!/usr/bin/env bash
# Register an existing tokenised scRNA-seq JSONL for rna_benchmark.
#
# The tokenised cells JSONL is produced by the existing
# molcrawl/preparation/preparation_script_rna.py pipeline; we do not
# re-download raw expression here.  This workflow simply places (or
# symlinks) the JSONL into the standard eval directory and writes a
# manifest entry.
#
# Required environment:
#   RNA_BENCHMARK_SOURCE - path to the JSONL produced by the prep step
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/rna_benchmark/source.jsonl
#   manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init rna_benchmark

if [ -z "${RNA_BENCHMARK_SOURCE:-}" ]; then
    ed_skip_with_instructions \
        "set RNA_BENCHMARK_SOURCE to the JSONL produced by molcrawl/preparation/preparation_script_rna.py."
fi

dest="$(ed_dest)/source.jsonl"
cp "${RNA_BENCHMARK_SOURCE}" "${dest}"
ed_register_existing "source.jsonl" "${RNA_BENCHMARK_SOURCE}"

ed_finalize_manifest \
    "rna_benchmark JSONL (project-internal)" \
    "https://github.com/deskull-m/MolCrawl-private" \
    "Internal" \
    "$(date -u +%Y%m%d)"
