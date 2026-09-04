#!/bin/bash
#SBATCH --job-name=mc-cmp-seedvar
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Repeat one ladder size with the seed changed and nothing else, so the spread between
# sizes can be judged against the spread between runs. Without this number the ladder
# cannot be argued about in either direction: the four shipped runs already use four
# different seeds (small 21 / medium 20 / large 19 / xl 22), so size and seed are
# confounded in what has been measured so far.
#
#   sbatch --export=ALL,SIZE=small,SEED=101,... workflows/compounds-seed-variance.sbatch
#
# Everything except `seed` matches the ladder config, and the run starts from scratch:
# a resumed run cannot be compared with a scratch one because the sampler's RNG state
# is not restored exactly.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${SEED:?set SEED to the seed for this repeat}"
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
# Never the ladder output directory: these are repeats, not replacements.
OUT="${RUNS_ROOT}/seedvar-gpt2-${SIZE}-seed${SEED}"
mkdir -p "${OUT}"

echo "=== compounds seed variance ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "size   : ${SIZE}   seed: ${SEED}"
echo "data   : ${LEARNING_SOURCE_DIR}"
echo "outputs: ${OUT}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CONFIG}" \
  --init_from=scratch --seed="${SEED}" \
  --out_dir="${OUT}" --compile=False --tensorboard=False && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? could only ever be 0.
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
