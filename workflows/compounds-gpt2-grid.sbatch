#!/bin/bash
#SBATCH --job-name=mc-cmp-gpt2-grid
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# One compounds GPT-2 sweep point, run from the config and nothing else.
#
#   sbatch --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/compounds/gpt2_xl_lr8e4.py,\
# LEARNING_SOURCE_DIR=<input root>,MODEL_OUTPUT_ROOT=<output root> \
#       workflows/compounds-gpt2-grid.sbatch
#
# Nothing is overridden at launch. compounds-lr-sweep.sbatch passes the rate, the seed
# and the length in, which is why the ladder configs still read 1,558 and 6e-4 for runs
# that trained at 4,674 and 1.6e-3 -- the file stopped describing the run. These configs
# carry every value, so the config is the answer to what a run was.
#
# out_dir comes from the config, which builds it from get_gpt2_output_path: that
# resolves under MODEL_OUTPUT_ROOT when set, and main.py's output guard refuses a path
# inside the input tree, which is where compounds' 1.4T of output already sits.
#
# 4 GPUs. nanoGPT divides gradient_accumulation_steps by the world size, so the
# effective batch is GPU-count-independent at 2,560 sequences; train.py stops the run
# when the product disagrees with the config's expected_global_batch.
set -eo pipefail
: "${CONFIG:?CONFIG=molcrawl/tasks/pretrain/configs/compounds/gpt2_<size>_lr<rate>.py}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the packed compounds learning_source root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "${REPO}/${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

# A git worktree holds only tracked files, so miniconda is not inside it.
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== compounds GPT-2 ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CONFIG}"
rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 molcrawl/models/gpt2/train.py "${CONFIG}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
