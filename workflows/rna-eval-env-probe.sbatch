#!/bin/bash
# 評価用の依存が、学習側の環境に同居させずに解決できるかを見る。導入はしない。aarch64 では wheel が無いパッケージがあり、どれが依存解決を壊すかを 1 つずつ切り分ける。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=rna-eval-env-probe
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${B200_DIR:?export B200_DIR to the learning_source root holding rna/download_dir and rna/metadata_preparation_dir}"
: "${RNA_WORK_DIR:?export RNA_WORK_DIR to a writable directory for intermediates (outside any input tree)}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${RNA_WORK_DIR}/tmp}"
mkdir -p "${TMPDIR}" "${RNA_WORK_DIR}" workflows/slurm-logs
echo "=== ${SLURM_JOB_NAME:-job} @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PY}" scripts/rna_eval_env_probe.py --work-dir "${RNA_WORK_DIR}" \
  ${PACKAGES:+--packages "${PACKAGES}"}
