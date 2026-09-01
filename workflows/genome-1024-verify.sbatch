#!/bin/bash
#SBATCH --job-name=mc-gen-1024verify
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Full source check on the rebuilt 1,024 data, including the chromosome scan
# that walks every row. The 512 build passed the same three gates; the 1,024
# build has to clear them on its own before anything trains on it, because the
# windows were cut again and the split was reassigned.
#
# Required: ROOT. Optional: SUBSET (default: all subsets under ROOT).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${ROOT:?set ROOT to the genome_sequence dir to check}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== node=$(hostname) root=${ROOT} subset=${SUBSET:-all} ==="
if [ -n "${SUBSET:-}" ]; then
    "${PY}" scripts/verify_genome_training_source.py --source-dir "${ROOT}" --subset "${SUBSET}" --model bert
else
    "${PY}" scripts/verify_genome_training_source.py --source-dir "${ROOT}" --all
fi
echo "---- rc=$? ----"
echo "=== done @ $(date -Is) ==="
