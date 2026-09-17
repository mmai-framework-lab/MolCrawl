#!/bin/bash
# DeepSpeed training on N GB200 nodes, one Slurm task per GPU (DeepSpeed order §8).
#
# Resources are not written here: scripts/submit_deepspeed.py builds them from the
# node count and the model config, and prints the full sbatch command (dry run by
# default). The legacy torchrun workflows are untouched; multi-node runs go only
# through this path (order §8.2).
#
# Required at submit time (exported by the submitter):
#   ARCH              gpt2 | bert
#   MODEL_CONFIG      path to the model config, relative to the repository root
#   RUN_NAME          output directory name under RUNS_ROOT
#   RUNS_ROOT         output root; no default, so a worktree never receives outputs
#   LEARNING_SOURCE_DIR  data root the configs read
# Optional:
#   DS_CONFIG         defaults to <MODEL_CONFIG without .py>.deepspeed.json
#   EXTRA_ARGS        further --key=value overrides for the trainer
#   EXPECT_COMMIT     refuse to start unless HEAD matches
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"

: "${ARCH:?set ARCH to gpt2 or bert}"
: "${MODEL_CONFIG:?set MODEL_CONFIG to the model config path}"
: "${RUN_NAME:?set RUN_NAME}"
: "${RUNS_ROOT:?set RUNS_ROOT to the output root (outside any worktree)}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root}"
[ -d "${LEARNING_SOURCE_DIR}" ] || { echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} does not exist" >&2; exit 1; }
[ -f "${MODEL_CONFIG}" ] || { echo "MODEL_CONFIG=${MODEL_CONFIG} not found under ${REPO}" >&2; exit 1; }
DS_CONFIG="${DS_CONFIG:-${MODEL_CONFIG%.py}.deepspeed.json}"
[ -f "${DS_CONFIG}" ] || { echo "DS_CONFIG=${DS_CONFIG} not found" >&2; exit 1; }

case "${ARCH}" in
  gpt2) TRAINER=molcrawl/models/gpt2/train.py; OUT_KEY=out_dir ;;
  bert) TRAINER=molcrawl/models/bert/main.py;  OUT_KEY=model_path ;;
  *) echo "ARCH=${ARCH}: expected gpt2 or bert" >&2; exit 1 ;;
esac
# The trainer has to accept the DeepSpeed backend before this launcher can drive it.
grep -qE '^\s*deepspeed_config\s*(:[^=]*)?=' "${TRAINER}" || {
  echo "${TRAINER} does not declare deepspeed_config yet; refusing to start" >&2; exit 1; }

# Interpreter from the main checkout (a worktree has no miniconda).
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no interpreter at ${PY}" >&2; exit 1; }

# Outputs never inside a worktree.
if [ "$(git -C "${REPO}" rev-parse --git-dir)" != "$(git -C "${REPO}" rev-parse --git-common-dir)" ]; then
  case "$(cd "${RUNS_ROOT}" 2>/dev/null && pwd || echo "${RUNS_ROOT}")/" in
    "${REPO}"/*) echo "RUNS_ROOT=${RUNS_ROOT} is inside the worktree ${REPO}" >&2; exit 1 ;;
  esac
fi
OUT="${RUNS_ROOT}/${RUN_NAME}"
mkdir -p "${OUT}" workflows/slurm-logs

if [ -n "${EXPECT_COMMIT:-}" ]; then
  HEAD="$(git -C "${REPO}" rev-parse HEAD)"
  case "${HEAD}" in "${EXPECT_COMMIT}"*) ;; *)
    echo "HEAD ${HEAD} does not match EXPECT_COMMIT ${EXPECT_COMMIT}" >&2; exit 1 ;; esac
fi

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

: "${SLURM_JOB_NUM_NODES:?run under sbatch}"
: "${SLURM_CPUS_PER_TASK:?submit with --cpus-per-task}"
mapfile -t SRUN < <("${PY}" -m molcrawl.models._launcher srun-args \
  --nodes "${SLURM_JOB_NUM_NODES}" --cpus-per-task "${SLURM_CPUS_PER_TASK}")

TRAIN_CMD=("${PY}" "${TRAINER}" "${MODEL_CONFIG}" "--${OUT_KEY}=${OUT}" "--deepspeed_config=${DS_CONFIG}")
# shellcheck disable=SC2206  # EXTRA_ARGS is a list of --key=value words by design
[ -n "${EXTRA_ARGS:-}" ] && TRAIN_CMD+=(${EXTRA_ARGS})

export MOLCRAWL_LAUNCHER=srun_deepspeed
export MOLCRAWL_LAUNCHER_ARGS="${SRUN[*]} ${PY} -m molcrawl.models._launcher exec -- ${TRAIN_CMD[*]}"
MOLCRAWL_LAUNCHER_SCRIPT="$(scontrol show job -o "${SLURM_JOB_ID}" | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
export MOLCRAWL_LAUNCHER_SCRIPT

echo "=== deepspeed-train ${ARCH} ${MODEL_CONFIG} on ${SLURM_JOB_NODELIST} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"
echo "ds config: ${DS_CONFIG}"
echo "=== outputs: ${OUT} ==="
echo "step: ${MOLCRAWL_LAUNCHER_ARGS}"

rc=0
"${SRUN[@]}" "${PY}" -m molcrawl.models._launcher exec -- "${TRAIN_CMD[@]}" || rc=$?
if [ "${rc}" -ne 0 ]; then
  "${PY}" -m molcrawl.models._launcher record-failure --out-dir "${OUT}" --exit-code "${rc}" || true
fi
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
