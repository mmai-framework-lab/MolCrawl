#!/bin/bash
#SBATCH --job-name=mc-preserve-learning-source
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Full MD5 of every file under learning_source*, with no exceptions.
#
# Nothing in these trees can be regenerated. The datasets predate #134, which
# seeded the split, so a rebuild draws a different one; and no checkpoint records
# which dataset directory it read, so a run cannot say what it trained on. What is
# on disk is the only original, and this is the record that lets us prove later
# that a given directory is still the one that was trained on.
#
# Duplicates are hashed too -- five groups are known to be byte-identical, 328 GiB
# of rna among them. Deciding which copy is redundant is a separate question and
# not one to settle by leaving a copy unrecorded.
#
# One GPU because the partition requires one, not because the work uses it.
# 48 workers over 32 cores, one directory per worker. A single Lustre stream reads at
# ~150 MiB/s, so throughput is set by how many reads are in flight. Hashing the files
# within one directory -- four, in a checkpoint directory -- capped the first attempt at
# four streams and 611 MiB/s, unchanged from 16 workers to 48. Spreading the workers
# across the 2,906 directories is what actually raises it. Oversubscribing threads to
# cores is deliberate: md5 runs ~500 MB/s per core, so the waiting dominates.
#
# Resumable. The ledger is JSONL, appended a line per directory, and a re-run
# skips what is already there -- so a timeout costs at most one directory.

set -euo pipefail
: "${ROOT:?set ROOT to the main checkout whose trees are recorded}"
OUT="${OUT:-${ROOT}/tmp/learning-source-md5-$(date +%F).jsonl}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=== root: ${ROOT}  ledger: ${OUT} ==="

python3 scripts/preserve_learning_source.py \
    --root "${ROOT}" \
    --glob 'learning_source*' \
    --out "${OUT}" \
    --workers 48
