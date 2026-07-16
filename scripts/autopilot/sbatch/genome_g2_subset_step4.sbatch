#!/bin/bash
#SBATCH --job-name=g2-split
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-g2-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-g2-%x-%j.out

# Genome G2 subset pipeline step 4 (parquet → contig-unit split → HF Arrow).
# Uses the STREAMING _contig_unit_split from Condition 1 (F2 worktree). Applies
# TARGET_TOTAL_WINDOWS (F2-c trim) computed by aggregate_realized_windows.py
# after all subset step-3 runs finish.
#
# Usage:
#   sbatch --job-name=g2-split-mammal_centered \
#          --export=SUBSET=mammal_centered,TARGET_TOTAL_WINDOWS=42500000000 \
#          tmp/scripts/autopilot/sbatch/genome_g2_subset_step4.sbatch

set -euo pipefail

WORKTREE=/lustre/home/matsubara/riken-genome-f2-worktree
cd "${WORKTREE}"

: "${SUBSET:?SUBSET must be set}"
: "${TARGET_TOTAL_WINDOWS:?TARGET_TOTAL_WINDOWS must be set (int, autopilot picks this)}"

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_genome_v2}"
PY="${MOLCRAWL_PYTHON:-/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python}"

echo "=== autopilot G2 subset step 4 ==="
echo "node       : $(hostname)"
echo "SLURM_JOB  : ${SLURM_JOB_ID}"
echo "SUBSET     : ${SUBSET}"
echo "TARGET     : ${TARGET_TOTAL_WINDOWS}"
echo "date       : $(date -Iseconds)"

"${PY}" - <<PYEOF
import os, sys
sys.path.insert(0, "${WORKTREE}")
os.chdir("${WORKTREE}")
from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig
from molcrawl.data.genome_sequence.preparation import (
    resolve_subset_paths,
    process4_subset_parquet_to_arrow,
    setup_logging,
)
from molcrawl.core.paths import GENOME_SEQUENCE_DIR

cfg = GenomeSequenceConfig.from_file("assets/configs/genome_sequence.yaml").data_preparation
setup_logging(GENOME_SEQUENCE_DIR)
resolve_subset_paths(cfg, "${SUBSET}")
base_dir = cfg.output_dir

target = int("${TARGET_TOTAL_WINDOWS}")
ok = process4_subset_parquet_to_arrow(
    base_dir,
    ["bert", "gpt2"],
    valid_size=50_000,
    test_size=50_000,
    valid_frac=0.005,
    test_frac=0.005,
    remove_parquet=False,
    target_total_windows=target,
    split_seed=42,
)
if not ok:
    sys.exit("Step4 FAIL")
print(f"OK ${SUBSET} step 4 complete (target={target:,}): {base_dir}")
PYEOF

echo "=== step 4 finished ==="
echo "exit=$?, date=$(date -Iseconds)"
