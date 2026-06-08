#!/bin/bash
#SBATCH --job-name=genome-prep-local
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=96:00:00
#SBATCH --output=../learning_source_20260518_genome/genome_sequence/logs/slurm-%j.out
#SBATCH --error=../learning_source_20260518_genome/genome_sequence/logs/slurm-%j.out

# Per-slice batch job for the species_links Stage 4 (tokenise → parquet).
#
# Earlier history:
#   18082 / 18141 — single-job runs. 18082 OOM'd at 64G on species 4; 18141
#                   bumped --mem to 256G and got 35/262 done before the
#                   --time=24:00:00 cutoff (~45 min per species sequential).
#
# To finish the remaining ~227 species without taking another week, this
# script is parameterised by SPECIES_RANGE so 4 concurrent SLURM jobs can
# split the work alphabetically. The Stage 3 shared tokenizer must already
# exist before parallel jobs run; otherwise each job would race-train its
# own incompatible tokenizer from its slice.
#
# Submit example:
#   sbatch --export=SPECIES_RANGE=0:65    workflows/01-genome_sequence-prepare-local-slurm.sh
#   sbatch --export=SPECIES_RANGE=65:130  workflows/01-genome_sequence-prepare-local-slurm.sh
#   sbatch --export=SPECIES_RANGE=130:195 workflows/01-genome_sequence-prepare-local-slurm.sh
#   sbatch --export=SPECIES_RANGE=195:262 workflows/01-genome_sequence-prepare-local-slurm.sh
#
# An unset SPECIES_RANGE processes the full 0:262 range (back-compat with
# single-job submits).
#
# The GPU is reserved only because the h200-long partition requires a GRES
# allocation; SentencePiece and HF tokenisation are both pure CPU+RAM.
#
# LEARNING_SOURCE_DIRs live one directory up from the repo, out of
# riken-dataset-fundational-model/, so VSCode does not have to enumerate the
# bloated dataset trees.

set -e

# Capture an externally provided PYTHON before sourcing common_functions.sh,
# which auto-detects PYTHON via `conda run` (see the PYTHON export below).
_PYTHON_OVERRIDE="${PYTHON:-}"

# Repo root: honor an explicit REPO_ROOT, else the sbatch submit dir (cwd at
# submit time), else derive from this script's location for plain `bash`.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/workflows/common_functions.sh" ]; then
        REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/workflows/common_functions.sh"

# Relative path resolves against REPO_ROOT (we just cd'd there). Override via
# the environment to point at a different learning-source root.
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260518_genome}"
mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs/"

# Override the auto-detected PYTHON. common_functions.sh tries `conda run -n
# molcrawl` which fails in non-interactive SLURM shells (no conda init), so
# it falls back to system python which has no molcrawl package. Job 18140
# died for exactly this reason. Honor an externally provided PYTHON (captured
# above, before the helper ran), else default to `python` on PATH.
export PYTHON="${_PYTHON_OVERRIDE:-python}"

RANGE_ARGS=()
if [ -n "${SPECIES_RANGE:-}" ]; then
    RANGE_ARGS=(--species-range "${SPECIES_RANGE}")
fi

echo "=== SLURM job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} ==="
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  mem-per-node=${SLURM_MEM_PER_NODE}"
echo "SPECIES_RANGE=${SPECIES_RANGE:-<unset, full 0:262>}"
echo "start: $(date)"

"$PYTHON" molcrawl/data/genome_sequence/preparation_local.py \
    assets/configs/genome_sequence.yaml \
    --input-dir "${SPECIES_LINKS_DIR:-/lustre/home/kojima-t/data/species_links}" \
    --stage-workers 8 \
    "${RANGE_ARGS[@]}"

echo "end: $(date)"
