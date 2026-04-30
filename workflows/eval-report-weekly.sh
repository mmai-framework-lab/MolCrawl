#!/usr/bin/env bash
# Phase 6 - weekly evaluation snapshot.
#
# Walks experiment_data/eval (or INPUT_DIR) looking for metrics.json
# files emitted by the per-task evaluators and writes
#   snapshot_<date>.json + snapshot_<date>.md under OUTPUT_DIR.

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-experiment_data/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/evaluation}"
PREVIOUS="${PREVIOUS:-}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation._snapshot
     --input-dir "$INPUT_DIR"
     --output-dir "$OUTPUT_DIR")
if [[ -n "$PREVIOUS" ]]; then
    cmd+=(--previous "$PREVIOUS")
fi
"${cmd[@]}"
