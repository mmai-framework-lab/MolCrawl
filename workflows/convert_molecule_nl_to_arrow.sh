#!/bin/bash
#
# Convert molecule NL parquet file to split arrow files
#
# Usage:
#   bash workflows/convert_molecule_nl_to_arrow.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Set default LEARNING_SOURCE_DIR if not set
LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-learning_20251121}"

echo "=========================================="
echo "Convert Molecule NL Parquet to Arrow"
echo "=========================================="
echo "LEARNING_SOURCE_DIR: ${LEARNING_SOURCE_DIR}"
echo ""

PARQUET_FILE="${LEARNING_SOURCE_DIR}/molecule_nl/molecule_related_natural_language_tokenized.parquet"
OUTPUT_DIR="${LEARNING_SOURCE_DIR}/molecule_nl/arrow_splits"

if [ ! -f "${PARQUET_FILE}" ]; then
    echo "ERROR: Parquet file not found: ${PARQUET_FILE}"
    echo "Please run the preparation script first:"
    echo "  LEARNING_SOURCE_DIR='${LEARNING_SOURCE_DIR}' bash workflows/01_molecule-nl_prepare.sh"
    exit 1
fi

echo "Input:  ${PARQUET_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Activate conda environment
source miniconda/bin/activate

# Run conversion
python scripts/preparation/convert_parquet_to_arrow.py \
    "${PARQUET_FILE}" \
    "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Conversion completed!"
echo "=========================================="
echo "Arrow files saved to: ${OUTPUT_DIR}"
echo ""
echo "Files:"
ls -lh "${OUTPUT_DIR}"
