#!/bin/bash
# GPT-2 の埋め込みで細胞種を線形分類する（Tabula Sapiens の評価器）。
#
# 分割は JSONL が持つものを使う -- 無作為に細胞を切ると同じドナーが両側に入り、
# 線形プローブが細胞種ではなくドナー固有の信号を読める。
# 窓は 1,024。既定の 512 では 1 細胞の半分が黙って落ちる。
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=rna-celltype-probe
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${MODEL_PATH:?export MODEL_PATH to the GPT-2 checkpoint directory}"
: "${EVAL_JSONL:?export EVAL_JSONL to the evaluation JSONL}"
: "${PROBE_OUT_DIR:?export PROBE_OUT_DIR to the directory results should be written to}"
# Passed rather than derived: core.paths builds it from LEARNING_SOURCE_DIR, so
# an unset root resolves to a relative path and the load fails 35s in, looking
# like a network error rather than a missing root.
: "${TOKENIZER_DIR:?export TOKENIZER_DIR to the tokenizer directory}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
mkdir -p workflows/slurm-logs "${PROBE_OUT_DIR}"
echo "=== rna-celltype-probe @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "=== outputs: ${PROBE_OUT_DIR} ==="
echo "model: ${MODEL_PATH}"
echo "jsonl: ${EVAL_JSONL}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
"${PY}" -m molcrawl.tasks.evaluation.tabula_sapiens \
  --model-path "${MODEL_PATH}" \
  --jsonl-path "${EVAL_JSONL}" \
  --output-dir "${PROBE_OUT_DIR}" \
  --tokenizer-path "${TOKENIZER_DIR}" \
  --arch gpt2 --modality rna \
  ${MAX_CELLS:+--max-cells "${MAX_CELLS}"}
echo "=== done @ $(date -Is) ==="
