#!/bin/bash
#SBATCH --job-name=mc-gen-heldout
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Confirm the held-out human chromosomes never reached the training data.
#
# The ClinVar evaluation scores variants on chr21/22/X/Y. A model that read
# those contigs during pretraining is being asked about sequence it already saw,
# which is a different failure from anything in the pretraining comparison and
# cannot be repaired afterwards -- only the evaluation set can move. Cheaper to
# know now.
#
# The hold-out is an accession-prefix rule (NC_000021..NC_000024) naming human
# chromosomes, so only a subset carrying the human genome can fail it. Of the 21,
# that is mammal_centered alone; a clean result elsewhere means the human genome
# is absent, not that a filter worked. The report prints accession counts beside
# the verdict so the two cases stay distinguishable.
#
# Required: ROOT. Optional: MODELS, SUBSETS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${ROOT:?set ROOT to the genome_sequence dir to audit}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="tmp/heldout-audit-${SLURM_JOB_ID:-0}.json"
mkdir -p tmp
echo "=== node=$(hostname) root=${ROOT} models=${MODELS:-gpt2,bert} ==="

"${PY}" scripts/audit_genome_heldout.py "${ROOT}" \
    --models "${MODELS:-gpt2,bert}" \
    ${SUBSETS:+--subsets "${SUBSETS}"} \
    --json-out "${OUT}"
rc=$?
echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
