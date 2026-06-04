#!/bin/bash

# Prepare genome_sequence training data from a pre-staged local FASTA
<<<<<<< HEAD
=======
# (e.g. files curated under /lustre/home/kojima-t/data/species_links/).
>>>>>>> 22b5ecd (feat(genome): add preparation pipeline for pre-staged FASTA files)
#
# Sibling of 01-genome_sequence-prepare.sh, which downloads via ncbi_genome_download.
#
# Usage:
#   LEARNING_SOURCE_DIR=learning_source_20260518_genome \
#   bash workflows/01-genome_sequence-prepare-local.sh \
<<<<<<< HEAD
#     --input  Homo_sapiens/GCF_000001405.40_GRCh38.p14_genomic_no_chr22.fna.gz \
=======
#     --input  /lustre/home/kojima-t/data/species_links/Homo_sapiens/GCF_000001405.40_GRCh38.p14_genomic_no_chr22.fna.gz \
>>>>>>> 22b5ecd (feat(genome): add preparation pipeline for pre-staged FASTA files)
#     --species homo_sapiens \
#     --group   vertebrate_mammalian
#
# Additional flags supported by preparation_local.py:
#   --force, --skip-stats, --only-stage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs/"

LOG="${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome-sequence-preparation-local-$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Launching local prep, log: ${LOG}"
nohup "$PYTHON" molcrawl/data/genome_sequence/preparation_local.py \
    assets/configs/genome_sequence.yaml \
    "$@" \
    > "${LOG}" 2>&1 &
echo "PID: $!"
