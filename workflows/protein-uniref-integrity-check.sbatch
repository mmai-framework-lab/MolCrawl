#!/bin/bash
#SBATCH --job-name=mc-protein-uniref-integ
#SBATCH --partition=gpu
#SBATCH --account=rkp00024
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Is the downloaded uniref50.fasta.gz truncated? protein-uniref-judgment-2026-10-27
# section 3.1 (gzip integrity) and 3.2 (first/last FASTA headers). No GPU: it is
# a gzip -t over 8.7 GiB plus a head/tail of the 16.9 GiB FASTA, which does not
# belong on the login node.
#
#   GZ=<.../uniref50.fasta.gz> FASTA=<.../uniref50.fasta> OUT=<out_dir> \
#     sbatch workflows/protein-uniref-integrity-check.sbatch
set -uo pipefail

: "${GZ:?set GZ to uniref50.fasta.gz}"
: "${FASTA:?set FASTA to uniref50.fasta}"
: "${OUT:?set OUT to the output directory}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p workflows/slurm-logs "${OUT}"
echo "=== uniref50 integrity on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "GZ=${GZ}"
echo "FASTA=${FASTA}"

# ---- 3.1 gzip -t ----
echo "=== 3.1 gzip -t ==="
gzip -t -v "${GZ}"
gzrc=$?
echo "gzip_test_exit=${gzrc}"
echo "${gzrc}" > "${OUT}/gzip_test_exit.txt"

# ---- 3.2 first and last FASTA header ----
echo "=== 3.2 first header ==="
first=$(grep -m1 '^>' "${FASTA}")
echo "FIRST_HEADER: ${first}"
# Last header: read the tail of the file (entries can be long; 64 MiB is ample)
# and take the last line beginning with '>'.
echo "=== 3.2 last header ==="
last=$(tail -c 67108864 "${FASTA}" | grep '^>' | tail -n1)
echo "LAST_HEADER: ${last}"
# Also the very last line of the file, to show it ends on a complete sequence line.
echo "=== 3.2 last line of file ==="
lastline=$(tail -n1 "${FASTA}")
echo "LAST_LINE_LEN: ${#lastline}"

{
  echo "gzip_test_exit	${gzrc}"
  echo "first_header	${first}"
  echo "last_header	${last}"
  echo "last_line_len	${#lastline}"
  echo "fasta_bytes	$(stat -c %s "${FASTA}")"
  echo "gz_bytes	$(stat -c %s "${GZ}")"
} > "${OUT}/integrity.tsv"

echo "=== wrote ${OUT}/integrity.tsv @ $(date -Is) ==="
