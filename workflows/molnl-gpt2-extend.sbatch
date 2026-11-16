#!/bin/bash
#SBATCH --job-name=mc-molnl-gpt2-ext
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# mol_nl GPT-2: continue a finished run past its schedule, to where the validation loss
# stops falling (2026-11-16 order §1).
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<corpus root> sbatch \
#     --export=ALL,EXTEND=1,CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small_1500_lr1p2e3_ext.py \
#     workflows/molnl-gpt2-extend.sbatch
#
# Why this is not molnl-gpt2-grid.sbatch with a flag. That script passes
# --init_from=scratch and refuses a directory that already holds a checkpoint, which is
# the right rule for a grid arm and the exact opposite of what an extension does. The two
# refusals would have to be inverted, so they live in separate files and each one states
# which it is.
#
# What an extension may and may not change. The model, the corpus, the global batch, the
# seed and the peak rate are the finished run's, taken from the config it imports. The
# rate past the original schedule is settled by leaving lr_decay_iters where it was:
# get_lr returns min_lr for every iteration past it (models/gpt2/train.py), and the
# cosine is already at min_lr when it arrives there, so the extension runs at a constant
# peak/10 with no step change at the resume point. Moving lr_decay_iters with max_iters
# recomputes the rate at every iteration from the first and is a different run, not a
# continuation -- so this script refuses unless lr_decay_iters is at or before the step
# being resumed from.
#
# The copy. The run being extended keeps its own directory: retention ranks the best ten
# checkpoints and an extension writing into that directory would prune the ones the
# reported numbers refer to. workflows/molnl-extend-prepare.sbatch copies the final
# checkpoint and the run's logging CSV into this run's directory -- the CSV because
# train.py reads the evaluations back from it (eval_history), and without it the earlier
# checkpoints rank as unscored and are prunable.
set -eo pipefail
: "${CONFIG:?CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_<size>_1500_lr<rate>_ext.py}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ "${EXTEND:-0}" = "1" ] || {
  echo "REFUSING: this script resumes a finished run into a larger max_iters. Say EXTEND=1" \
       "so that is a decision rather than a default." >&2; exit 1; }
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="${CONFIG}"
CFG_NAME="$(basename "${CFG}" .py)"

# A git worktree holds only tracked files, so miniconda and assets/ are not inside it.
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || {
  echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun" >&2; exit 1; }
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# Outputs must not land in a worktree: learning_source* is gitignored, so each checkout
# has its own and a worktree is meant to be disposable. The log says which root was used.
ARTIFACT_ROOT="${MOLNL_ARTIFACT_ROOT:-${PYROOT}/learning_source_20260803_molnl_train}"
RUNS_ROOT="${RUNS_ROOT:-${ARTIFACT_ROOT}/molecule_nat_lang}"
OUT="${RUNS_ROOT}/gpt2-output-grid/${CFG_NAME}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs
[ -f "${CFG}" ] || { echo "no config at ${CFG} (relative to ${REPO})" >&2; exit 1; }

"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "out_dir" || exit 1

# Where this resumes from. train.py prefers the newest checkpoint-*/training_state.bin,
# so the step is readable from the directory name and nothing has to be loaded here.
_last=$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "${_last}" ] || [ ! -f "${OUT}/checkpoint-${_last}/training_state.bin" ]; then
  echo "REFUSING: ${OUT} holds no checkpoint-*/training_state.bin to resume from, so" \
       "train.py would start from scratch and spend the allocation re-running step 0." \
       "Run workflows/molnl-extend-prepare.sbatch first; it copies the finished run's" \
       "final checkpoint and logging CSV in here." >&2
  exit 1
fi

# An early stop is the answer, not an interruption. The marker is written at the end of
# this script when the run ended below its ceiling, and refusing on it keeps a resubmit
# -- a chained segment, a retry by hand -- from walking past the step the stop chose.
if [ -f "${OUT}/EARLY_STOPPED" ]; then
  echo "REFUSING: ${OUT}/EARLY_STOPPED says this run already stopped because the" \
       "validation loss stopped falling:" >&2
  sed 's/^/    /' "${OUT}/EARLY_STOPPED" >&2
  echo "    Remove the file only to reopen a decision that was already reached." >&2
  exit 1
fi

_cfg_dump=$("${PYTHON}" - "${CFG}" <<'PYEOF' 2>/dev/null
import runpy, sys
g = runpy.run_path(sys.argv[1])
for k in ("max_iters", "lr_decay_iters", "warmup_iters", "learning_rate", "min_lr", "seed",
          "batch_size", "gradient_accumulation_steps", "block_size", "eval_interval",
          "eval_sequences", "early_stopping", "early_stopping_patience", "init_from",
          "save_checkpoint_steps", "max_checkpoints", "dtype", "dataset_dir"):
    print(f"{k}={g.get(k)}")
PYEOF
) || { echo "could not read ${CFG}" >&2; exit 1; }
_val() { printf '%s\n' "${_cfg_dump}" | sed -n "s/^$1=//p"; }
_max_iters=$(_val max_iters); _decay=$(_val lr_decay_iters)
_es=$(_val early_stopping); _init=$(_val init_from)

[ "${_init}" = "resume" ] || {
  echo "REFUSING: ${CFG} resolves init_from=${_init:-None}. An extension continues the" \
       "copied checkpoint; init_from=scratch would train a second run over it." >&2; exit 1; }
[ "${_es}" = "True" ] || {
  echo "REFUSING: ${CFG} resolves early_stopping=${_es:-None}. The order of 2026-11-16" \
       "runs these to where the validation loss stops falling, and the grid configs set" \
       "early_stopping = False -- \`import *\` carries that in, so an extension config has" \
       "to state True itself." >&2; exit 1; }
[ "${_max_iters}" -gt "${_last}" ] || {
  echo "REFUSING: ${CFG} says max_iters=${_max_iters} and the checkpoint is already at" \
       "${_last}. There is nothing to extend." >&2; exit 1; }
[ "${_decay}" -le "${_last}" ] || {
  echo "REFUSING: ${CFG} says lr_decay_iters=${_decay}, past the step ${_last} this" \
       "resumes from, so the cosine would still be running and the rate at every" \
       "iteration -- including the ones already trained -- would be a different schedule" \
       "from the one the checkpoint was written under. Leave lr_decay_iters at the" \
       "original schedule's length; get_lr returns min_lr past it." >&2; exit 1; }

echo "${SLURM_JOB_ID:-local} extend-from-${_last} $(date -Is)" >> "${OUT}/segments.log"
echo "=== mol_nl ${CFG_NAME} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified; the schedule is not overridden here)"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}  (config resolves dataset_dir)"
echo "extension: resuming from checkpoint-${_last}, ceiling max_iters=${_max_iters}," \
     "constant rate past lr_decay_iters=${_decay}"
echo "--- config values as resolved ---"
printf '%s\n' "${_cfg_dump}" | sed 's/^/    /'
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CFG}" \
  --out_dir="${OUT}" --tensorboard_dir="${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="

# Record an early stop where the next submission will see it. The run's own CSVs say
# where it ended; a last evaluation short of the ceiling after a clean exit is the stop.
if [ "${rc}" = "0" ]; then
  _end=$(cat "${OUT}"/logging_*.csv 2>/dev/null | awk -F, '/^[0-9]/ {print $1}' | sort -n | tail -1)
  if [ -n "${_end}" ] && [ "${_end}" -lt "${_max_iters}" ]; then
    printf 'job %s ended at iteration %s of max_iters %s on %s: early stopping (patience %s evaluations)\n' \
      "${SLURM_JOB_ID:-local}" "${_end}" "${_max_iters}" "$(date -Is)" "$(_val early_stopping_patience)" \
      > "${OUT}/EARLY_STOPPED"
    echo "early stop recorded in ${OUT}/EARLY_STOPPED"
  fi
fi
exit "${rc}"
