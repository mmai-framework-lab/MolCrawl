#!/bin/bash
#SBATCH --job-name=mc-gen-stab
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=16:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Stability sweep over the full production length, one point per job.
#
# All 21 production subsets diverged in the back half: best-val landed at step
# 11-14k and train and val then rose together to the end, +0.46 on average. Not
# overfitting -- the two move in step, so this is the run destabilising.
#
# The earlier §5 check ran 8,000 steps because the v1 note said the spike showed
# up "around step 6k". Onset here was 15,000-20,000, outside what was measured.
# Hence full length: the failure has to be inside the window.
#
# What the runs point at is how long the schedule sits at a high learning rate.
# Divergence began just as LR fell through ~5e-4, and every GPT-2 config sets
# lr_decay_iters = max_iters, so the schedule is normalised to run length and the
# high-LR stretch is always the first ~28% -- but its absolute length scales:
#
#   genome    54,634 steps -> 15,345 above 5e-4   diverged 21/21
#   rna       40,320       -> 11,325              unmeasured
#   protein   11,177       ->  3,140              completed clean, best == final
#
# Two ways to shorten that stretch, and they trade off differently, so the sweep
# varies them separately rather than together:
#
#   LR         lower the peak. Cheap to state, but lowers LR everywhere, and §5
#              found 6e-4 beat 6e-5 over the first 8k, so early progress suffers.
#   DECAY_FRAC decay sooner (lr_decay_iters = frac * max_iters). get_lr pins LR
#              at min_lr past lr_decay_iters, so this shortens the high-LR
#              stretch while leaving the early schedule alone.
#
# Judge on whether val turns back up anywhere in the run, not at some point
# defined relative to the high-LR stretch -- DECAY_FRAC moves that stretch, and a
# yardstick that moves with the treatment cannot compare the treatments. Residency
# is recorded per point as the explanatory variable, and doubles as calibration:
# 3,140 is known good and 15,345 known bad, with nothing measured in between.
#
# Required: TAG. Optional: LR (default 6e-4), DECAY_FRAC (default 1.0),
# GENOME_SUBSET (default mammal_centered).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${TAG:?set TAG via --export, e.g. TAG=D-decay025}"
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"

SUBSET="${GENOME_SUBSET:-mammal_centered}"
LR="${LR:-0.0006}"
DECAY_FRAC="${DECAY_FRAC:-1.0}"
MB="${MB:-320}"
GA="${GA:-8}"

LOCAL_ROOT=/tmp/genome-stab
DST="${LOCAL_ROOT}/genome_sequence/${SUBSET}"
OUT="$PWD/learning_source_genome_runs/stab-${TAG}"

export GENOME_SUBSET="${SUBSET}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) tag=${TAG} subset=${SUBSET} LR=${LR} decay_frac=${DECAY_FRAC} ==="

echo "########## staging to local NVMe ##########"
mkdir -p "${DST}" "${LOCAL_ROOT}/genome_sequence"
# Staging repoints LEARNING_SOURCE_DIR, and the tokenizer path moves with it; the
# config would otherwise have four ranks racing to build it on a fresh root.
for aux in custom_tokenizer_bert_single_nuc; do
    [ -d "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" ] && \
        cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" "${LOCAL_ROOT}/genome_sequence/" || true
done
t0=$(date +%s)
cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${SUBSET}/training_ready_hf_dataset_gpt2" "${DST}/" || {
    echo "stage failed"; exit 1; }
echo "=== staged in $(( $(date +%s) - t0 )) s ==="

# max_iters comes from the realized row count, so read it from the config rather
# than hardcoding a per-subset number, then derive lr_decay_iters from it.
read -r MAX_ITERS DECAY_ITERS RESIDENCY <<EOF
$(LEARNING_SOURCE_DIR="${LOCAL_ROOT}" GENOME_SUBSET="${SUBSET}" ./miniconda/bin/python - "${LR}" "${DECAY_FRAC}" <<'PY' 2>/dev/null
import math, sys, runpy
lr, frac = float(sys.argv[1]), float(sys.argv[2])
ns = runpy.run_path("molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py")
mx, warm = ns["max_iters"], ns["warmup_iters"]
dec = max(warm + 1, int(mx * frac))
mn = lr / 10
n = 0
for it in range(mx):
    if it < warm: cur = lr * it / warm
    elif it > dec: cur = mn
    else: cur = mn + 0.5 * (1 + math.cos(math.pi * (it - warm) / (dec - warm))) * (lr - mn)
    if cur >= 5e-4: n += 1
print(mx, dec, n)
PY
)
EOF

[ -n "${MAX_ITERS:-}" ] || { echo "could not derive schedule from config"; exit 1; }
echo "=== max_iters=${MAX_ITERS} lr_decay_iters=${DECAY_ITERS} residency(LR>=5e-4)=${RESIDENCY} ==="

LEARNING_SOURCE_DIR="${LOCAL_ROOT}" \
SUBSET_GPT2_LR="${LR}" \
GPT2_LR_TAG="stab-${TAG}" \
./miniconda/bin/torchrun --standalone --nproc_per_node=4 \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
    --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
    --lr_decay_iters="${DECAY_ITERS}" \
    --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
    --compile=False --log_interval=100
rc=$?

rm -rf "${LOCAL_ROOT}"
echo "---- ${TAG} rc=${rc} residency=${RESIDENCY} ----"
echo "=== done @ $(date -Is) ==="
