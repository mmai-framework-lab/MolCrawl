#!/bin/bash
#SBATCH --job-name=g2-prep
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-g2-%x-%j.out
#SBATCH --error=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/slurm-g2-%x-%j.out

# Genome G2 subset pipeline steps 1-3 (download → FASTA→raw → raw→parquet).
# NO step 4 (parquet→arrow / split) — that runs separately after realized
# window counts across all 21 subsets are aggregated and target chosen
# (charter Condition 2: measure → minimize → trim).
#
# Runs in the F2 worktree because Condition 1's streaming code lives there.
# Requests 1 GPU only because the pipeline is CPU/IO bound; the other 7 GPUs
# on the node stay free for training jobs.
#
# Usage:
#   sbatch --job-name=g2-mammal_centered \
#          --export=SUBSET=mammal_centered \
#          tmp/scripts/autopilot/sbatch/genome_g2_subset_steps1_3.sbatch

set -euo pipefail

WORKTREE=/lustre/home/matsubara/riken-genome-f2-worktree
cd "${WORKTREE}"

: "${SUBSET:?SUBSET must be set (e.g. mammal_centered)}"

# autopilot output tree — separate v2 dir so existing subset dirs stay intact.
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/learning_source_20260710_genome_v2}"
mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs"

# Pick python from the molcrawl env (matches common_functions.sh selection).
PY="${MOLCRAWL_PYTHON:-/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python}"

echo "=== autopilot G2 subset steps 1-3 ==="
echo "node       : $(hostname)"
echo "SLURM_JOB  : ${SLURM_JOB_ID}"
echo "SUBSET     : ${SUBSET}"
echo "worktree   : ${WORKTREE}"
echo "LSD        : ${LEARNING_SOURCE_DIR}"
echo "PY         : ${PY}"
echo "date       : $(date -Iseconds)"

# Step 4 is disabled by writing a sentinel that make process4 skip; then run
# the normal --subset flow which iterates steps 1-4. We intercept via a
# controlled invocation: run steps 1-3 manually via a Python one-liner.

"${PY}" - <<PYEOF
import os, sys
sys.path.insert(0, "${WORKTREE}")
os.chdir("${WORKTREE}")
from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig
from molcrawl.data.genome_sequence.preparation import (
    resolve_subset_paths,
    process1_subset_download,
    process2_subset_fasta_to_raw,
    process3_subset_raw_to_parquet,
    setup_logging,
)
from molcrawl.core.paths import GENOME_SEQUENCE_DIR

cfg = GenomeSequenceConfig.from_file("assets/configs/genome_sequence.yaml").data_preparation
setup_logging(GENOME_SEQUENCE_DIR)
resolve_subset_paths(cfg, "${SUBSET}")
base_dir = cfg.output_dir

ok = process1_subset_download(base_dir, cfg.path_species, cfg.num_worker, verify_md5=getattr(cfg, "verify_md5", True))
if not ok:
    sys.exit("Step1 FAIL")
ok = process2_subset_fasta_to_raw(base_dir, cfg.num_worker, getattr(cfg, "min_segment_len", 100))
if not ok:
    sys.exit("Step2 FAIL")
ok = process3_subset_raw_to_parquet(
    base_dir,
    ["bert", "gpt2"],
    getattr(cfg, "bert_chunk_size", 510),
    getattr(cfg, "gpt2_chunk_size", 1024),
    cfg.num_worker,
)
if not ok:
    sys.exit("Step3 FAIL")
print(f"OK ${SUBSET} steps 1-3 complete: {base_dir}")
PYEOF

echo "=== steps 1-3 finished ==="
echo "exit=$?, date=$(date -Iseconds)"
