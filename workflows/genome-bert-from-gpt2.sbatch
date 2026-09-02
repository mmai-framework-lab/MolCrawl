#!/bin/bash
#SBATCH --job-name=mc-gen-bertfromgpt2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Derive the BERT dataset from the GPT-2 one rather than cutting windows again.
#
# Re-cutting at 1,022 produced a split that shared two contigs out of five
# thousand with GPT-2's, because the split fills a 50,000-*window* quota and the
# window count of a contig depends on the chunk length. Wrapping GPT-2's rows in
# [CLS]/[SEP] keeps its split intact, so the two models see the same bases.
#
# Verification runs inside the conversion and the job stops if it fails.
#
# Required: SRC_ROOT, DST_ROOT. Optional: SUBSETS (default: all under SRC_ROOT).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC_ROOT:?set SRC_ROOT to the genome_sequence dir holding training_ready_hf_dataset_gpt2}"
: "${DST_ROOT:?set DST_ROOT to the genome_sequence dir to write into}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome-1024"

if [ -n "${SUBSETS:-}" ]; then LIST="${SUBSETS//,/ }"
else LIST=$(for d in "${SRC_ROOT}"/*/; do
                [ -d "${d}/training_ready_hf_dataset_gpt2" ] && basename "${d}"; done | sort); fi

echo "=== node=$(hostname) subsets=$(echo ${LIST} | wc -w) ==="
rc=0
for s in ${LIST}; do
    echo ""
    echo "########## ${s} ##########"
    mkdir -p "${DST_ROOT}/${s}"
    "${PY}" scripts/build_genome_bert_from_gpt2.py \
        "${SRC_ROOT}/${s}/training_ready_hf_dataset_gpt2" \
        "${DST_ROOT}/${s}/training_ready_hf_dataset_bert" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" || { echo "  FAILED: ${s}"; rc=1; }
done
echo ""
echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
