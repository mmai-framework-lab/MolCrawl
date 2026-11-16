#!/bin/bash
#SBATCH --job-name=mc-es-probe
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Does early stopping stop GPT-2 and BERT where its rule says it should?
#
# No production run has had it on, so the path has not run on this cluster. Eight
# cases, each launched through the real entry point on 4 GPUs (DDP), compounds
# small, lr 0 and a fixed evaluation so the loss cannot move and the stop step is
# known in advance. Cases, expected steps and the reasoning behind each are in
# scripts/early_stopping_probe.py (CASES):
#
#   g_on          GPT-2, on, patience 3                    -> stops at 30
#   g_off         GPT-2, off                               -> runs to 100
#   g_resume      GPT-2, stop at 20 then resume            -> stops at 20 (counter carried)
#   b_on          BERT, on, save every eval                -> stops at 40
#   b_off         BERT, off                                -> runs to 100
#   b_save20      BERT, save every 2nd eval                -> stops at 50 (best set at saves only)
#   b_save20_soi  as b_save20 with save_on_improve         -> stops at 40
#   b_resume      BERT, stop at 30 then resume             -> stops at 60 (counter reset)
#
# A rank left waiting at a collective appears as rc 124 (the per-launch timeout).
#
# Submit from the repository root (main checkout or a worktree):
#   sbatch --export=ALL,LEARNING_SOURCE_DIR=<compounds packed root>,PROBE_ROOT=<dir> \
#     workflows/early-stopping-probe.sbatch
# LEARNING_SOURCE_DIR must hold compounds/organix13/training_ready_hf_dataset_{gpt2,bert}.
# Outputs go to ${PROBE_ROOT}/early-stopping-<jobid>/; weights are deleted after the
# check unless KEEP_WEIGHTS=1 (trainer_state.json, manifests and logs stay).
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the root holding compounds/}"
: "${PROBE_ROOT:?export PROBE_ROOT to the directory the probe writes under}"
for d in training_ready_hf_dataset_gpt2 training_ready_hf_dataset_bert; do
  [ -d "${LEARNING_SOURCE_DIR}/compounds/organix13/${d}" ] || {
    echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no compounds/organix13/${d}" >&2; exit 1; }
done
CASE_TIMEOUT="${CASE_TIMEOUT:-1500}"  # s per launch; healthy launches take a few minutes
KEEP_WEIGHTS="${KEEP_WEIGHTS:-0}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

OUT="${PROBE_ROOT}/early-stopping-${SLURM_JOB_ID:-manual}"
[ ! -e "${OUT}" ] || { echo "${OUT} already exists; refusing to mix runs" >&2; exit 1; }
mkdir -p "${OUT}/logs" "${OUT}/rc"

export LEARNING_SOURCE_DIR
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_CACHE="${OUT}/hf-cache"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== early-stopping probe ${SLURM_JOB_ID:-} on ${SLURMD_NODENAME:-?} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
echo "data   : ${LEARNING_SOURCE_DIR}"
echo "outputs: ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

echo "--- unit tests for the rules the check replays ---"
"${PYTHON}" -m pytest -q tests/unit/test_early_stopping_probe.py tests/unit/test_bert_early_stopping.py

"${PYTHON}" scripts/early_stopping_probe.py write-configs "${OUT}"

prev_case="" prev_rc=0
while IFS=$'\t' read -r case phase arch cfg run_dir; do
  if [ "${case}" = "${prev_case}" ] && [ "${prev_rc}" != 0 ]; then
    echo "skip ${case}-${phase}: phase before it failed (rc ${prev_rc})"
    echo "skipped" > "${OUT}/rc/${case}-${phase}"
    continue
  fi
  if [ "${arch}" = gpt2 ]; then entry=molcrawl/models/gpt2/train.py; else entry=molcrawl/models/bert/main.py; fi
  echo "--- ${case}-${phase} (${arch}) @ $(date -Is) ---"
  rc=0
  timeout --kill-after=60 "${CASE_TIMEOUT}" \
    "${TORCHRUN}" --standalone --nproc_per_node=4 "${entry}" "${cfg}" \
    > "${OUT}/logs/${case}-${phase}.log" 2>&1 || rc=$?
  echo "${rc}" > "${OUT}/rc/${case}-${phase}"
  echo "    rc=${rc}; last lines:"; tail -n 3 "${OUT}/logs/${case}-${phase}.log" | sed 's/^/    /'
  prev_case="${case}" prev_rc="${rc}"
done < "${OUT}/plan.tsv"

echo "--- check @ $(date -Is) ---"
check_rc=0
"${PYTHON}" scripts/early_stopping_probe.py check "${OUT}" || check_rc=$?

if [ "${KEEP_WEIGHTS}" != 1 ]; then
  # Only inside this job's own fresh directory. Keeps trainer_state.json, the
  # manifests, the logs and the per-rank RNG files; drops the tensors.
  find "${OUT}/runs" -type f \( -name '*.safetensors' -o -name 'optimizer.pt' \
       -o -name 'pytorch_model.bin' -o -name 'training_state.bin' -o -name 'ckpt.pt' \
       -o -name 'scheduler.pt' \) -delete
  rm -rf "${OUT}/hf-cache"
  echo "weights removed (KEEP_WEIGHTS=1 keeps them); $(du -sh "${OUT}" | cut -f1) left"
fi
echo "=== done check_rc=${check_rc} @ $(date -Is) ==="
exit "${check_rc}"
