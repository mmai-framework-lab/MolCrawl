#!/bin/bash
#SBATCH --job-name=mc-molnet-fetch
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Compute-node wrapper for eval-moleculenet-fetch.sh — downloads belong off the
# login node. --gpus=1 only because `gpu` is the sole partition here.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch workflows/eval-moleculenet-fetch.sbatch
set -e
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p workflows/slurm-logs
echo "=== moleculenet fetch ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
bash workflows/eval-moleculenet-fetch.sh
echo "=== rc=$? @ $(date -Is) ==="
