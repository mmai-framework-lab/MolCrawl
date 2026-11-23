#!/bin/bash
#SBATCH --job-name=mc-prot-bert-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Smoke one GENERATED protein BERT grid config before the 39-arm launch: run a few
# steps with evaluation firing, so the fixed-mask path (FixedEvalMaskCollator wrapping
# the protein ProteinSequenceDataCollator -- a combination no run has used) and the
# judged-metric keys are exercised end to end. Trains nothing worth keeping; the output
# goes to a throwaway dir that is removed at the end.
#
#   sbatch --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/protein_sequence/bert_small_b256_lr9p49e5.py,\
#          LEARNING_SOURCE_DIR=...,MODEL_OUTPUT_ROOT=... workflows/protein-bert-config-smoke.sbatch
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${CONFIG:?CONFIG=path to a generated config}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT

if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}" || exit 1
mkdir -p workflows/slurm-logs
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

OUT="${MODEL_OUTPUT_ROOT}/smoke-$(basename "${CONFIG}" .py)-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "=== smoke ${CONFIG} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "code: $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"

# max_steps 20, warmup 2, eval every 10 so two eval points fire; save past the end.
"${TORCHRUN}" --standalone --nproc_per_node=4 \
    molcrawl/models/bert/main.py "${CONFIG}" \
    --model_path="${OUT}" \
    --max_steps=20 --warmup_steps=2 --log_interval=10 --save_steps=1000 \
    2>&1 | tee "${OUT}/smoke.log"
RC=${PIPESTATUS[0]}

echo
echo "=== eval lines produced ==="
grep -iE "eval_loss_mask|eval_loss|Fixed evaluation mask" "${OUT}/smoke.log" | tail -10
echo
if [ "${RC}" -eq 0 ] && grep -q "eval_loss_mask" "${OUT}/smoke.log"; then
  echo "SMOKE OK: config trained ${CONFIG##*/} and produced eval_loss_mask"
else
  echo "SMOKE FAILED (rc=${RC}); see ${OUT}/smoke.log"
  tail -30 "${OUT}/smoke.log"
fi
rm -rf "${OUT}"
echo "=== done @ $(date -Is) ==="
