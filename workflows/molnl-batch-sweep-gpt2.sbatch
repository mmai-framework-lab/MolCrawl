#!/bin/bash
#SBATCH --job-name=mc-molnl-bsweep-gpt2
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# One arm of the molnl GPT-2 batch-size grid, run from its config and nothing else.
#
#   sbatch --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small_b256_lr1p9e4.py,\
# LEARNING_SOURCE_DIR=<input root>,MODEL_OUTPUT_ROOT=<output root> \
#       workflows/molnl-batch-sweep-gpt2.sbatch
#
# Why this is not molnl-gpt2-grid.sbatch. That script builds the output directory from
# the config's file name under MOLNL_ARTIFACT_ROOT, which is the tree molnl's input and
# output already share. This grid's configs name their own out_dir through
# get_gpt2_output_path, so it resolves under MODEL_OUTPUT_ROOT and the 20 arms land
# outside the input tree -- and the config, not the launcher, is the answer to where an
# arm wrote. Nothing about the schedule is overridden here either way.
#
# 4 GPUs. nanoGPT divides gradient_accumulation_steps by the world size, so the
# effective batch is micro x accumulation whatever the GPU count; train.py stops the run
# when that product disagrees with the config's expected_global_batch (§5).
set -eo pipefail
: "${CONFIG:?CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_<size>_b<batch>_lr<rate>.py}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the output root for this grid, outside the input tree}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
case "${CONFIG}" in
  */configs/molecule_nat_lang/gpt2_*) ;;
  *) echo "CONFIG=${CONFIG} is not a molecule_nat_lang GPT-2 config" >&2; exit 1;;
esac
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
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

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs
[ -f "${CFG}" ] || { echo "no config at ${CFG} (relative to ${REPO})" >&2; exit 1; }

# The grid configs import their size's base rather than restating it, so the values
# cannot be read out of the file with grep. Run it the way train.py will and read what
# it resolves to: the log then answers what the run was without the config being
# reopened later, and the guards below look at the same directory the run will.
_cfg_dump=$("${PYTHON}" - "${CFG}" <<'PYEOF' 2>/dev/null
import runpy, sys
g = runpy.run_path(sys.argv[1])
keys = ("max_iters", "lr_decay_iters", "warmup_iters", "learning_rate", "min_lr",
        "seed", "batch_size", "gradient_accumulation_steps", "expected_global_batch",
        "block_size", "eval_interval", "log_interval", "eval_sequences",
        "deterministic_val_eval", "early_stopping", "init_from", "dtype",
        "save_checkpoint_steps", "max_checkpoints", "always_save_checkpoint",
        "dataset_dir", "out_dir", "tensorboard_dir")
for k in keys:
    print(f"{k}={g.get(k)}")
PYEOF
) || { echo "could not read ${CFG}" >&2; exit 1; }
OUT=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^out_dir=//p')
[ -n "${OUT}" ] || { echo "no out_dir in ${CFG}" >&2; exit 1; }
case "${OUT}" in
  "${MODEL_OUTPUT_ROOT%/}"/*) ;;
  *) echo "out_dir ${OUT} from ${CFG} is not under MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}" >&2; exit 1;;
esac
case "$(basename "${OUT}")" in
  *-b*-lr*) ;;
  *) echo "out_dir ${OUT} is not named for a batch and a rate; an arm must not share its size's directory" >&2; exit 1;;
esac
_init=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^init_from=//p')
[ "${_init}" = "scratch" ] || {
  echo "REFUSING: ${CFG} resolves init_from=${_init}. molnl's GPT-2 base says 'resume';" \
       "a grid arm must start from scratch or it continues whatever is in its directory." >&2; exit 1; }
_es=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^early_stopping=//p')
[ "${_es}" = "False" ] || {
  echo "REFUSING: ${CFG} resolves early_stopping=${_es}; §8 runs every arm to its budget." >&2; exit 1; }

# Ask the guard itself rather than re-deriving its rule here, so the two cannot drift
# and so nanoGPT -- which does not call it anywhere -- is covered too.
"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "out_dir" || exit 1

# Starting over a directory that already holds a checkpoint would train a second run on
# top of the first: init_from=scratch ignores what is there and the files interleave.
if [ -e "${OUT}/ckpt.pt" ] || [ -n "$(find "${OUT}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
  echo "REFUSING: ${OUT} already holds a checkpoint. This arm starts from scratch, so it" \
       "would write a second run into the first one's directory. Move it aside." >&2
  exit 1
fi

echo "=== molnl ${CFG_NAME} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified; the schedule is not overridden here)"
echo "outputs: ${OUT}   (MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}  (config resolves dataset_dir; see \"Loading dataset\" below)"
echo "--- config values as resolved ---"
printf '%s\n' "${_cfg_dump}" | sed 's/^/    /'
# -i 0 rather than a pipe into head: under `set -eo pipefail` the pipe makes the
# job exit 141 whenever head closes the pipe before nvidia-smi is done writing,
# which killed one probe of eight two seconds in (job 374112).
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CFG}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
