#!/usr/bin/env bash
# Build a small UniRef50 reference FASTA for protein_foldability.
#
# The full UniRef50 fasta is multiple GB; we fetch the seqres helper
# from RCSB which gives a few hundred MB of representative sequences.
# For larger experiments, override REFERENCE_URL to point at a local
# UniRef50 mirror.
#
# Optional:
#   REFERENCE_URL     - alternate FASTA URL
#   REFERENCE_NAME    - filename to use (default: derived from URL)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init protein_foldability

URL="${REFERENCE_URL:-https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz}"
NAME="${REFERENCE_NAME:-$(basename "${URL}")}"

ed_download "${URL}" "${NAME}"

ed_finalize_manifest \
    "RCSB pdb_seqres reference" \
    "${URL}" \
    "Public Domain (RCSB)" \
    "$(date -u +%Y%m%d)"

cat <<EOF

Next step:
  Decompress ${NAME} (gunzip) and pass its path as
  --reference-fasta to molcrawl.tasks.evaluation.protein_foldability.
  Override REFERENCE_URL to use UniRef50 instead.
EOF
