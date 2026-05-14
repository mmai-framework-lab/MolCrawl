#!/usr/bin/env bash
# Download a chromosome-scoped slice of gnomAD genomes (v4.1).
#
# Pulling the full release is hundreds of GB; this script downloads one
# chromosome per invocation so users can pick a tractable scope.  Use
# GNOMAD_CHROM (default: 22) and GNOMAD_VERSION to control the slice.
#
# Required environment:
#   none
# Optional:
#   GNOMAD_CHROM     - chromosome label (default: chr22)
#   GNOMAD_VERSION   - release version (default: v4.1)
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/gnomad_af_correlation/
#     gnomad.<version>.<chrom>.vcf.bgz
#     gnomad.<version>.<chrom>.vcf.bgz.tbi
#     manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init gnomad_af_correlation

VERSION="${GNOMAD_VERSION:-4.1}"   # path segment, e.g. "4.1"
CHROM="${GNOMAD_CHROM:-chr22}"
# gnomAD's filenames prefix the version with "v" (e.g. v4.1) while the
# URL path segment itself is just the numeric version (4.1). Keep both
# forms independent so future releases with different conventions are
# easy to parameterise.
FILE_VERSION="${GNOMAD_FILE_VERSION:-v${VERSION}}"
BASE="https://storage.googleapis.com/gcp-public-data--gnomad/release/${VERSION}/vcf/genomes"
NAME="gnomad.genomes.${FILE_VERSION}.sites.${CHROM}.vcf.bgz"

ed_download "${BASE}/${NAME}" "${NAME}"
ed_download "${BASE}/${NAME}.tbi" "${NAME}.tbi"

ed_finalize_manifest \
    "gnomAD genomes (${VERSION}, ${CHROM})" \
    "https://gnomad.broadinstitute.org/" \
    "CC0" \
    "${VERSION}/${CHROM}"

cat <<EOF

Next step:
  Extract (reference_sequence, variant_sequence, allele_frequency) from
  ${NAME} into a CSV consumed by
  molcrawl.tasks.evaluation.gnomad_af_correlation.
EOF
