#!/bin/bash
#SBATCH --job-name=mc-cmp-lrsweep
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Vary the learning rate on one ladder size with everything else held, so the spread
# between sizes can be read against the spread a single size shows across learning
# rates. The ladder configs set a different rate per size (small 6e-4 / medium 3e-4 /
# large 2.5e-4 / xl 2e-4), and nothing has ever measured whether those choices, rather
# than capacity, are what orders the four runs.
#
#   sbatch --export=ALL,SIZE=medium,LR=1e-3,SEED=20,... workflows/compounds-lr-sweep.sbatch
#
# The seed is held at the ladder value so the existing run serves as a fourth point on
# the same curve. Results go to their own directory; the ladder outputs are untouched.
# TimeLimit is set for the slowest size this script is used with, not the fastest:
# xl measures 29.6 GPU-h over 4 GPUs, i.e. 7.4 hours of wall, and large 3.1. A 3-hour
# limit killed xl before it wrote a single eval point and cut large off at iter 1500
# of 1558. Sizes that finish sooner are not charged for the unused reservation, so the
# only cost of asking for more is queue position.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${LR:?set LR to the max learning rate for this point}"
: "${SEED:?set SEED to the seed to hold across the sweep}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the packed compounds learning_source root}"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to a runs root outside any worktree}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

TORCHRUN="${TORCHRUN:-${REPO}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun"; exit 1; }

CONFIG="molcrawl/tasks/pretrain/configs/compounds/gpt2_${SIZE}.py"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }
OUT="${RUNS_ROOT}/lrsweep-gpt2-${SIZE}-lr${LR}-seed${SEED}"
mkdir -p "${OUT}"

echo "=== compounds lr sweep ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "size   : ${SIZE}   lr: ${LR}   seed: ${SEED}"
echo "outputs: ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CONFIG}" \
  --init_from=scratch --seed="${SEED}" --learning_rate="${LR}" \
  --out_dir="${OUT}" --compile=False --tensorboard=False && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? could only ever be 0.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
