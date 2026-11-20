#!/bin/bash
#SBATCH --job-name=mc-molnl-bsweep-shape
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# How fast one size runs at each candidate for one GPU block count (§4, 2026-11-20).
#
# §4 asks for the fastest shape per size, measured, and constrains the candidates: one
# GPU block count times 4 GPUs must divide 256, so 4, 8, 16, 32 and 64 are what there is
# to choose from. It also says not to leave BERT at 8, which is where every molnl BERT
# config sits today, and GPT-2 large likewise.
#
# The one molnl measurement on record does not settle it. Jobs 75095-75097 compared
# 8x80, 32x20 and 64x10 on medium and found 35.57 / 35.16 / 35.15 s/step, a 1.2 %
# spread. They ran on 2026-09-01 and bf16 reached the BERT base configs on 2026-09-17
# (7a13dc0f); production medium now runs near 7.6 s/step. A shape comparison made on a
# 4.5x slower run does not carry over, because what it was measuring was mostly the
# part that got faster.
#
#   sbatch --export=ALL,ARCH=bert,SIZE=medium,SHAPES="8 16 32 64",STEPS=40,\
# LEARNING_SOURCE_DIR=<input root>,MODEL_OUTPUT_ROOT=<probe root> \
#       workflows/molnl-batch-sweep-shape.sbatch
#
# One job is one (ARCH, SIZE), and it runs every shape in SHAPES one after another in
# the same allocation, on the same node, in the order given. That is deliberate: the
# comparison is between the shapes, and the node, the filesystem and whatever else is
# running are the things most likely to move a throughput number between two
# allocations.
#
# The global batch is held at 2,560 for every shape, which is what the size base
# configs declare, so main.py and train.py check the product themselves and the shape
# cannot drift into a different model. The grid runs at 256 to 4,096 instead, and the
# per-step time there is not this number: what transfers is which decomposition is
# fastest, which is a property of one GPU's share.
#
# NOT a production s/step. STEPS is passed as the schedule length, so the whole probe
# sits inside a warmup that production leaves behind. Every shape is wrong in the same
# way and the ratio is what this is for.
set -eo pipefail
: "${ARCH:?ARCH=bert or gpt2}"
: "${SIZE:?SIZE=small, medium, large or xl}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a probe output root, outside the input tree}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
case "${ARCH}" in bert|gpt2) ;; *) echo "ARCH must be bert or gpt2" >&2; exit 1;; esac
case "${SIZE}" in small|medium|large|xl) ;; *) echo "SIZE must be small, medium, large or xl" >&2; exit 1;; esac

GLOBAL="${GLOBAL:-2560}"
# The candidates §4 leaves open. xl is given the smaller end of the range by default:
# one GPU holding 64 blocks of 1,024 tokens at 1.47 G parameters is the shape most
# likely to run out of memory, and a probe that dies tells us less than one that runs.
if [ -z "${SHAPES:-}" ]; then
  case "${ARCH}-${SIZE}" in
    bert-xl|gpt2-xl) SHAPES="4 8 16 32" ;;
    *)               SHAPES="8 16 32 64" ;;
  esac
fi
# Long enough that the per-step number is not one step's noise, short enough that four
# shapes of the largest size fit in one allocation.
if [ -z "${STEPS:-}" ]; then
  case "${SIZE}" in
    small)  STEPS=50 ;;
    medium) STEPS=40 ;;
    large)  STEPS=25 ;;
    xl)     STEPS=20 ;;
  esac
fi
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || { echo "no torchrun at ${TORCHRUN}" >&2; exit 1; }
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs

CFG="molcrawl/tasks/pretrain/configs/molecule_nat_lang/${ARCH}_${SIZE}.py"
[ -f "${CFG}" ] || { echo "no config at ${CFG}" >&2; exit 1; }
PROBE_ROOT="${MODEL_OUTPUT_ROOT%/}/molecule_nat_lang/batch-shape-probe-20261120/${ARCH}-${SIZE}"
mkdir -p "${PROBE_ROOT}"

echo "=== molnl ${ARCH} ${SIZE} batch-shape probe ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config : ${CFG}"
echo "shapes : ${SHAPES}   (one GPU block count; x 4 GPUs must divide 256)"
echo "steps  : ${STEPS} per shape, global batch ${GLOBAL} throughout"
echo "probes : ${PROBE_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

rc=0
for micro in ${SHAPES}; do
  if [ "${ARCH}" = "bert" ]; then
    divisor=$(( micro * 4 ))
  else
    divisor="${micro}"   # nanoGPT divides the accumulation by the world size itself
  fi
  if [ $(( GLOBAL % divisor )) -ne 0 ]; then
    echo "-- skipping ${micro}: it cannot build the global batch ${GLOBAL}"
    continue
  fi
  if [ $(( 256 % (micro * 4) )) -ne 0 ]; then
    echo "-- skipping ${micro}: ${micro} x 4 GPUs cannot build the grid smallest batch 256 (§4)"
    continue
  fi
  accum=$(( GLOBAL / divisor ))
  OUT="${PROBE_ROOT}/${micro}x${accum}"
  LOG="${PROBE_ROOT}/${micro}x${accum}.log"
  rm -rf "${OUT}"; mkdir -p "${OUT}"
  echo
  echo "########## ${ARCH} ${SIZE}: one GPU ${micro} blocks x ${accum} = ${GLOBAL} ##########"
  echo "-- started $(date -Is), log ${LOG}"
  step=0
  if [ "${ARCH}" = "bert" ]; then
    # One evaluation, at the end, so the timing reader has one number to subtract. The
    # periodic save is pushed past the end and save_on_improve is off: a checkpoint
    # write sits inside train_runtime and is not training.
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
      molcrawl/models/bert/main.py "${CFG}" \
      --model_path="${OUT}" \
      --batch_size="${micro}" --gradient_accumulation_steps="${accum}" \
      --per_device_eval_batch_size=8 \
      --max_steps="${STEPS}" --log_interval="${STEPS}" \
      --save_steps=$(( STEPS * 10 )) --save_on_improve=False \
      > "${LOG}" 2>&1 || step=$?
  else
    # No evaluation inside the window: eval_interval past the end leaves the one at
    # iteration 0, which the reader drops with the warmup.
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
      molcrawl/models/gpt2/train.py "${CFG}" \
      --init_from=scratch --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
      --batch_size="${micro}" --gradient_accumulation_steps="${accum}" \
      --max_iters="${STEPS}" --lr_decay_iters="${STEPS}" --warmup_iters=5 \
      --eval_interval=$(( STEPS + 1 )) --log_interval=1 \
      --always_save_checkpoint=False --save_checkpoint_steps=$(( STEPS * 10 )) \
      > "${LOG}" 2>&1 || step=$?
  fi
  echo "-- finished $(date -Is), rc=${step}"
  if [ "${step}" -ne 0 ]; then
    echo "-- ${micro}x${accum} FAILED; its last 20 lines:"
    tail -20 "${LOG}" | sed 's/^/     /'
    # A shape that does not fit is an answer, not a stop: the rest still get measured.
    echo "SHAPE_RESULT arch=${ARCH} label=${SIZE}-${micro}x${accum} per_step_s= failed=${step}"
    rc="${step}"
    continue
  fi
  "${PYTHON}" scripts/molnl_batch_shape_timing.py --arch "${ARCH}" \
      --steps "${STEPS}" --label "${SIZE}-${micro}x${accum}" "${LOG}" \
    || { echo "-- could not read a time out of ${LOG}"; rc=1; }
  # The checkpoints a probe leaves are worth nothing and a size's four of them are not
  # small; the log is the measurement and it stays.
  rm -rf "${OUT}"
done

echo
echo "=== the shapes, as measured (one SHAPE_RESULT line per shape, above) ==="
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
