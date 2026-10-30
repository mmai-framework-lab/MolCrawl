#!/bin/bash
#SBATCH --job-name=mc-gen-stab2
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Second round of the stability sweep. Round 1 varied peak LR and decay length
# and settled one thing: peak LR is what decides whether the run survives.
# Shortening the high-LR stretch did not help -- D held 6e-4 for only 3,740
# steps, close to protein's 3,140 which finishes clean, and still reversed
# +0.0931. Everything at 6e-4 diverged regardless of decay; 3e-4 reversed
# +0.0262; 1e-4 reversed +0.0013.
#
# That killed the residency measure this sweep was built around, so round 2
# drops it and asks three narrower questions.
#
#   E  1e-4 run 1.5x longer. 1e-4 was still improving when it hit the end --
#      best at 48,000 of 54,634, where every other point bottomed out by 18,000.
#      Calling it "stable but reaches less than D" assumes it had converged, and
#      it plainly had not. Same trap the compounds notice describes: judging a
#      run while it is still on the plateau.
#
#   F  2e-4 at production length. 1e-4 is stable, 3e-4 mildly reverses, and
#      nothing sits between them. If 2e-4 holds it likely reaches further than
#      1e-4, and the production setting should be the highest LR that is stable
#      rather than the lowest one tried.
#
#   G  6e-4 with tighter gradient clipping. grad_clip is already 1.0, so this is
#      not about clipping being absent -- divergence happened with it on. The
#      question is whether 0.5 is enough to hold 6e-4 together, which would keep
#      its early speed instead of trading it away.
#
# What round 1 left unexplained and this does not settle: D kept degrading after
# its LR was pinned at 6e-5, with train and val falling together. Low LR did not
# repair whatever the 6e-4 phase did. G speaks to it indirectly; if clipping
# rescues 6e-4 the damage was gradient-scale, and if it does not the cause lies
# elsewhere.
#
# Required: TAG. Optional: LR, DECAY_FRAC, STEPS_MULT, GRAD_CLIP, GENOME_SUBSET.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
# Outputs go to the main checkout, not to a worktree that is meant to be deleted.
RUNS_ROOT="${RUNS_ROOT:-${PYROOT}/learning_source_genome_runs}"
echo "=== outputs under: ${RUNS_ROOT} ==="

: "${TAG:?set TAG via --export, e.g. TAG=E-lr1e4-long}"
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"

SUBSET="${GENOME_SUBSET:-mammal_centered}"
LR="${LR:-0.0006}"
DECAY_FRAC="${DECAY_FRAC:-1.0}"
STEPS_MULT="${STEPS_MULT:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MB="${MB:-320}"
GA="${GA:-8}"

LOCAL_ROOT=/tmp/genome-stab2
DST="${LOCAL_ROOT}/genome_sequence/${SUBSET}"
OUT="${RUNS_ROOT}/stab2-${TAG}"

export GENOME_SUBSET="${SUBSET}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) tag=${TAG} LR=${LR} decay_frac=${DECAY_FRAC} steps_mult=${STEPS_MULT} grad_clip=${GRAD_CLIP} ==="

echo "########## staging to local NVMe ##########"
mkdir -p "${DST}" "${LOCAL_ROOT}/genome_sequence"
# Staging repoints LEARNING_SOURCE_DIR and the tokenizer path moves with it;
# without this all four ranks race to build it on a fresh root.
for aux in custom_tokenizer_bert_single_nuc; do
    [ -d "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" ] && \
        cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${aux}" "${LOCAL_ROOT}/genome_sequence/" || true
done
t0=$(date +%s)
cp -r "${GENOME_SOURCE_ROOT}/genome_sequence/${SUBSET}/training_ready_hf_dataset_gpt2" "${DST}/" || {
    echo "stage failed"; exit 1; }
echo "=== staged in $(( $(date +%s) - t0 )) s ==="

# Derive the schedule from the config's own max_iters so this works for any
# subset, then apply the multiplier and decay fraction on top.
read -r MAX_ITERS DECAY_ITERS <<EOF
$(LEARNING_SOURCE_DIR="${LOCAL_ROOT}" GENOME_SUBSET="${SUBSET}" "${PYROOT}/miniconda/bin/python" - "${STEPS_MULT}" "${DECAY_FRAC}" <<'PY' 2>/dev/null
import sys, runpy
mult, frac = float(sys.argv[1]), float(sys.argv[2])
ns = runpy.run_path("molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py")
mx = int(ns["max_iters"] * mult)
print(mx, max(ns["warmup_iters"] + 1, int(mx * frac)))
PY
)
EOF

[ -n "${MAX_ITERS:-}" ] || { echo "could not derive schedule from config"; exit 1; }
echo "=== max_iters=${MAX_ITERS} lr_decay_iters=${DECAY_ITERS} ==="

LEARNING_SOURCE_DIR="${LOCAL_ROOT}" \
SUBSET_GPT2_LR="${LR}" \
GPT2_LR_TAG="stab2-${TAG}" \
"${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py \
    --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
    --max_iters="${MAX_ITERS}" \
    --lr_decay_iters="${DECAY_ITERS}" \
    --grad_clip="${GRAD_CLIP}" \
    --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
    --compile=False --log_interval=100
rc=$?

rm -rf "${LOCAL_ROOT}"
echo "---- ${TAG} rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
