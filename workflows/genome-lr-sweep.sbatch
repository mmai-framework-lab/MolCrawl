#!/bin/bash
#SBATCH --job-name=mc-gen-lr
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# §5 learning-rate pre-check for GPT-2, one LR per job.
#
# The config records that on the pre-G2 corpus 6e-4 spiked back up around step
# 6k while 6e-5 stayed monotone, and asks for that to be re-checked on the
# current data before the 21-subset launch. The shipped default is 6e-4.
#
# Running this the obvious way would not answer the question. Setting
# SMOKE_MAX_STEPS also compresses lr_decay_iters to the same value, so an
# 8k-step run decays over 8k instead of the production 54,634 and step 6000
# lands at a completely different point on the schedule:
#
#   production (warmup 1092, decay 54634)   5.889e-04
#   SMOKE_MAX_STEPS=8000 alone              2.601e-04   44% of production
#   + SMOKE_WARMUP_ITERS=1092               1.642e-04   28% -- worse
#   + lr_decay_iters=54634 as well          5.889e-04   matches
#
# Fixing warmup alone makes it worse, because the shorter warmup starts the
# decay earlier; the decay horizon dominates. So both are pinned to production
# values here, which makes these 8k steps an exact prefix of the production
# trajectory. Overriding lr_decay_iters is a research value and was approved for
# this purpose.
#
# Staged on node-local NVMe: 5.83 -> 0.71 s/iter, so 8k steps costs about 1.6 h
# rather than 13.
#
# Required: LR. Optional: TAG (checkpoint dir suffix, defaults from LR).

set -uo pipefail
# Submitted from the repo root by the submit-* wrappers; SLURM records that as
# SLURM_SUBMIT_DIR. Falling back to the current directory keeps a direct sbatch
# from the repo root working too.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${LR:?set LR via --export, e.g. LR=0.0006}"
TAG="${TAG:-lr${LR}}"
STEPS="${STEPS:-8000}"

# mammal_centered production schedule (max_iters from 3 epochs at global 2560).
PROD_MAX_ITERS=54634
PROD_WARMUP=1092

SRC="${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}/genome_sequence/mammal_centered"
LOCAL_ROOT=/tmp/genome-local
DST="${LOCAL_ROOT}/genome_sequence/mammal_centered"

export GENOME_SUBSET=mammal_centered
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) LR=${LR} tag=${TAG} steps=${STEPS} ==="
echo "=== schedule pinned to production: warmup=${PROD_WARMUP} decay=${PROD_MAX_ITERS} ==="

echo "########## staging to local NVMe ##########"
mkdir -p "${DST}"
t0=$(date +%s)
cp -r "${SRC}/training_ready_hf_dataset_gpt2" "${DST}/" || { echo "stage failed"; exit 1; }
echo "=== staged in $(( $(date +%s) - t0 )) s ==="

OUT="$PWD/learning_source_genome_runs/lrsweep-${TAG}"

# Finer evals than production so a spike near 6k is visible rather than
# straddled by the 1000-step production interval.
LEARNING_SOURCE_DIR="${LOCAL_ROOT}" \
SUBSET_GPT2_LR="${LR}" \
GPT2_LR_TAG="${TAG}" \
SMOKE_MAX_STEPS="${STEPS}" \
SMOKE_WARMUP_ITERS="${PROD_WARMUP}" \
SMOKE_EVAL_INTERVAL=250 \
./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
    --out_dir="${OUT}" \
    --tensorboard_dir="${OUT}" \
    --lr_decay_iters="${PROD_MAX_ITERS}" \
    --batch_size=128 --gradient_accumulation_steps=20 \
    --compile=False --log_interval=50
echo "---- ${TAG} rc=$? ----"

rm -rf "${LOCAL_ROOT}"
echo "=== done @ $(date -Is) ==="
