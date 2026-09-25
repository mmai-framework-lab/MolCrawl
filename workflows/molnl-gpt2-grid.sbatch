#!/bin/bash
#SBATCH --job-name=mc-molnl-gpt2-grid
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# mol_nl GPT-2 learning-rate grid on the shuffled corpus.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch \
#     --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small_1500_lr6e4.py \
#     workflows/molnl-gpt2-grid.sbatch
#
# What this does differently from molnl-gpt2-ladder.sbatch, and why there are two:
#
# The ladder script passes --max_iters, --lr_decay_iters and --eval_interval on the
# command line, so gpt2_small.py reads max_iters = 373 while the run on disk did 3,000.
# Which schedule a directory was trained under is then answerable only from the job log
# (2026-09-25 order §7). Here the schedule lives in the config, nothing about it is
# overridden, and run_manifest.json records what the run resolved to.
#
# Only the output location is passed, and it is derived from the config name, so the
# three learning rates of one size cannot write into one directory -- where the first
# would be resumed by the second instead of it starting.
#
# 4 GPUs. nanoGPT divides gradient_accumulation_steps by the world size, so the
# effective batch is GPU-count-independent at 2,560 sequences; the count changes the
# time, not the model.
set -eo pipefail
: "${CONFIG:?CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_<size>_1500_lr<rate>.py}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
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

# Inputs and outputs are separate trees; see molnl-bert-ladder.sbatch for what putting
# them in one cost. MOLNL_ARTIFACT_ROOT names where mol_nl's artifacts already live.
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

# Ask the guard itself rather than re-deriving its rule here, so the two cannot drift
# and so nanoGPT -- which does not call it anywhere -- is covered too.
"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "out_dir" || exit 1

# Starting over a directory that already holds a checkpoint would train a second run on
# top of the first: init_from=scratch ignores what is there and the files interleave.
if [ -e "${OUT}/ckpt.pt" ] || [ -n "$(find "${OUT}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
  echo "REFUSING: ${OUT} already holds a checkpoint. This script starts from scratch, so" \
       "it would write a second run into the first one's directory. Move it aside." >&2
  exit 1
fi

# The grid configs import their base rather than restating it, so the values cannot be
# read out of the file with grep. Run it the way train.py will and print what it resolves
# to -- the log then answers what the run was without the config being reopened later.
echo "=== mol_nl ${CFG_NAME} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified; the schedule is not overridden here)"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}  (config resolves dataset_dir; see \"Loading dataset\" below)"
echo "--- config values as resolved ---"
"${PYTHON}" - "${CFG}" <<'PYEOF' | sed 's/^/    /'
import runpy, sys
g = runpy.run_path(sys.argv[1])
for k in ("max_iters", "lr_decay_iters", "warmup_iters", "learning_rate", "min_lr", "seed",
          "batch_size", "gradient_accumulation_steps", "block_size", "eval_interval",
          "eval_sequences", "early_stopping", "dtype"):
    print(f"{k}={g.get(k)}")
PYEOF
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CFG}" \
  --init_from=scratch --out_dir="${OUT}" --tensorboard_dir="${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
