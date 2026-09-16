#!/bin/bash
#SBATCH --job-name=tput-arm-config-check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# No GPU. Executes the arm configs so the resolved values can be compared;
# a compute node because the configs open the dataset and write the tokenizer.
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LEARNING_SOURCE_DIR
cd "${REPO}"; mkdir -p workflows/slurm-logs
D=molcrawl/tasks/pretrain/configs/rna
echo "=== arm config check on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/verify_tput_arm_configs.py \
    "${D}/bert_small.py" \
    "${D}/bert_small_tput1_asis.py" \
    "${D}/bert_small_tput2_bf16.py" \
    "${D}/bert_small_tput3_bf16_workers.py" \
    "${D}/bert_small_tput3b_tf32.py" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
