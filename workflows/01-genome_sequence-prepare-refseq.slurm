#!/bin/bash
#SBATCH --job-name=refseq-gpt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --time=4-00:00:00
#SBATCH --output=refseq-%j.log

# Submit from the repository root. Site-specific account/partition are passed
# to sbatch, for example: sbatch -A PROJECT -p PARTITION THIS_FILE pilot
set -euo pipefail

PROJECT="${MOLCRAWL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$PROJECT/molcrawl/preparation/genome/parallel.py" ]]; then
    echo "Submit from the MolCrawl repository root, or set MOLCRAWL_PROJECT_DIR." >&2
    exit 2
fi
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${MOLCRAWL_PYTHON:-python}"
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-$PROJECT/learning_source}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_NUM_THREADS=1
export PYTHONUNBUFFERED=1

MODE="${1:-full}"
BASE="${REFSEQ_BASE:-$LEARNING_SOURCE_DIR/refseq_genomes}"
# Use the YAML entry point when a configuration file is supplied.
if [[ "$MODE" == *.yaml || "$MODE" == *.yml ]]; then
    "$PYTHON" -u -m molcrawl.preparation.preparation_script_genome_sequence \
        "$MODE" "${@:2}"
    exit $?
fi

case "$MODE" in
  pilot)
    OPTIONS=(--output "$BASE/processed_gpt2_n10_pilot" --workers 1 --shards 8 --pilot 32)
    ;;
  full)
    OPTIONS=(--output "$BASE/processed_gpt2_n10_parallel" --workers "${REFSEQ_WORKERS:-8}" --shards 64)
    ;;
  *) echo "Usage: sbatch workflows/01-genome_sequence-prepare-refseq.slurm [pilot|full]" >&2; exit 2 ;;
esac

"$PYTHON" -c 'import pyarrow; print("Python environment OK")'
"$PYTHON" -u -m molcrawl.preparation.genome.parallel \
    --input "${REFSEQ_INPUT:-$BASE/fasta}" "${OPTIONS[@]}"
