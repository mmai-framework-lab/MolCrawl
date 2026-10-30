#!/bin/bash
#SBATCH --job-name=mc-fingerprint-datasets
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Record what is on disk for the 67 training_ready_* datasets, so that a
# deletion or a silent rewrite becomes detectable.
#
# genome, protein and rna were all built before #134 seeded the split, so a
# rebuild does not reproduce them -- the copy on disk is the only original.
# Losing one would take every result measured on it with it. This is the ledger
# that lets us say later whether a directory is still the one that was trained on.
#
# One GPU because the partition requires one, not because the work uses it.
# The job is pure Lustre I/O: 16,489 files, three 1 MiB windows each, ~48 GiB
# read in total across 32 threads.
#
# Resumable -- the ledger is JSONL, appended a line per directory, and a re-run
# skips directories already present. Interrupting it loses at most one directory.

set -euo pipefail
: "${ROOT:?set ROOT to the main checkout whose trees are recorded}"
OUT="${OUT:-${ROOT}/tmp/dataset-fingerprints-$(date +%F).jsonl}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=== root: ${ROOT}  ledger: ${OUT} ==="

python3 scripts/fingerprint_datasets.py \
    --root "${ROOT}" \
    --out "${OUT}" \
    --workers 32
