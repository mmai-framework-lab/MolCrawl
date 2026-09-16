#!/bin/bash
#SBATCH --job-name=bert-speed-keys
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# No GPU. Executes every base BERT config and prints what it resolves to.
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LEARNING_SOURCE_DIR
# molnl builds a GPT-2 tokenizer and the genome subset config wants its subset
# name. Both are pre-existing requirements of those configs; defaulted here so the
# check covers every modality rather than reporting their absence as a finding.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${REPO}/assets/tokenizers/gpt2}"
export GENOME_SUBSET="${GENOME_SUBSET:-mammal_centered}"
C=molcrawl/tasks/pretrain/configs
echo "=== bert speed keys on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/verify_bert_speed_keys.py \
    ${C}/rna/bert_{small,medium,large}.py \
    ${C}/protein_sequence/bert_{small,medium,large}.py \
    ${C}/compounds/bert_{small,medium,large,xl}.py \
    ${C}/molecule_nat_lang/bert_{small,medium,large}.py || rc=$?
# genome_sequence/bert_small_subset.py is left out: its data sits under a
# different learning-source root, so it cannot be resolved in the same pass. It
# already carries bf16, four workers and pinning and is not touched here.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
