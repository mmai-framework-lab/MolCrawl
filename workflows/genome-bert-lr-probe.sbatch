#!/bin/bash
#SBATCH --job-name=mc-gen-bert-lrprobe
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# genome BERT learning-rate probe.
#
# The question this settles: bert_small_subset.py records that on the pre-G2
# data 1e-4 collapsed to the degenerate level and only 1e-5 trained, while the
# standing directive is 1e-4 for every modality. The config itself says not to
# use its default blind. GPT-2 on the same genome data turned out to have a
# narrow LR tolerance (6e-4 diverged 21/21, 1e-4 was stable), so the pre-G2
# observation should not be written off as stale.
#
# THE SCHEDULE MUST KEEP ITS PRODUCTION SHAPE. Truncating max_steps to 8,000
# would compress the whole decay into the probe: LR would be down to ~26% of
# peak by step 6,000, and an arm could pass here while the real run fails.
# Under the production shape, step 8,000 is only 5.3% into the decay, so the
# probe spends essentially all of its time at the production peak LR.
#
#   HARD_MAX_STEPS_OVERRIDE does NOT do this. It rewrites max_steps itself,
#   and main.py hands max_steps to TrainingArguments, which is what HF builds
#   the scheduler from. It preserves warmup_steps and compresses the decay --
#   the exact failure mode above. SMOKE_MAX_STEPS is worse still (40% warmup).
#
# So max_steps stays at the production value and the run is stopped from the
# outside once it has passed PROBE_STOP_STEPS. The stop signal is the arrival
# of checkpoint-<PROBE_STOP_STEPS> on disk (save_steps=1000): HF evaluates
# before it saves, so by the time that directory exists the eval at that step
# is already in the log. No repo change is needed for any of this.
#
# No NVMe staging. The BERT dataset is 230G per subset and the probe touches
# about 7% of it, while node-local staging measured only 1.3x for BERT (versus
# 8.2x for GPT-2). Copying 230G would cost more than it returns here.
#
# Required: GENOME_SOURCE_ROOT.
# Optional: LR, TAG, GENOME_SUBSET, PROBE_STOP_STEPS, MB, GA.
#
#   GENOME_SOURCE_ROOT=/path/to/learning_source LR=0.0001 TAG=lr1e4 \
#     sbatch workflows/genome-bert-lr-probe.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
# Outputs go to the main checkout, not to a worktree that is meant to be deleted.
RUNS_ROOT="${RUNS_ROOT:-${PYROOT}/learning_source_genome_runs}"
echo "=== outputs under: ${RUNS_ROOT} ==="

: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
[ -d "${GENOME_SOURCE_ROOT}/genome_sequence" ] || {
    echo "GENOME_SOURCE_ROOT=${GENOME_SOURCE_ROOT} has no genome_sequence/ under it" >&2; exit 1; }

LR="${LR:-0.0001}"
TAG="${TAG:-lr${LR}}"
SUBSET="${GENOME_SUBSET:-mammal_centered}"
STOP="${PROBE_STOP_STEPS:-8000}"
MB="${MB:-8}"
GA="${GA:-80}"

OUT="${RUNS_ROOT}/bert-lrprobe-${SUBSET}-${TAG}"
CONFIG=molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py

export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
export GENOME_SUBSET="${SUBSET}"
export SUBSET_BERT_LR="${LR}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subset=${SUBSET} LR=${LR} stop_at=${STOP} ==="
echo "=== commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
echo "=== global batch = ${MB} x ${GA} x 4 GPU = $((MB * GA * 4)) (HF counts both per-device) ==="

########## pre-flight: derive the real schedule from the config itself ##########
# dataset_info.json is stale on this tree -- it reports 46,720,417 rows for
# mammal_centered, which would give max_steps 54,751 against a verified runtime
# value of 54,634 for GPT-2. The config reads the arrow row count through
# load_from_disk (memory-mapped, metadata only), so running the config is the
# ground truth. This also produces the production max_steps that the BERT cost
# estimate needs.
"${PYROOT}/miniconda/bin/python" - "${STOP}" "${LR}" <<'PY'
import runpy
import sys

stop, lr = int(sys.argv[1]), float(sys.argv[2])
ns = runpy.run_path("molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py")
ms, warm = ns["max_steps"], ns["warmup_steps"]
frac = (stop - warm) / (ms - warm)
print("########## pre-flight ##########")
print(f"  train rows          : {ns['_train_n']:,}")
print(f"  max_steps (arrow)   : {ms:,}    <- production value, NOT truncated")
print(f"  warmup_steps        : {warm:,}")
print(f"  learning_rate       : {ns['learning_rate']:.2e}")
print(f"  stop step           : {stop:,}  ({stop / ms * 100:.1f}% of the run)")
print(f"  decay progress there: {frac * 100:.2f}%")
print(f"  LR at the stop step : {lr * (1 - frac):.3e}  ({(1 - frac) * 100:.1f}% of peak)")
if (1 - frac) < 0.90:
    print("  !! schedule looks compressed -- the probe would not test production conditions")
    raise SystemExit(1)
print(f"  degenerate line     : eval_loss_mask 1.3703 (measured); ln(4)=1.3863 is the uniform bound")
print("  judge on eval_loss_mask, never the overall eval_loss (copy positions dilute it ~0.077)")
PY
[ $? -eq 0 ] || { echo "pre-flight failed"; exit 1; }

########## train, stopped from the outside at PROBE_STOP_STEPS ##########
mkdir -p "${OUT}"
t0=$(date +%s)

# setsid so the whole worker group can be signalled without pkill, which would
# also hit any other BERT job sharing this node.
setsid "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
    molcrawl/models/bert/main.py "${CONFIG}" \
    --model_path="${OUT}" \
    --batch_size="${MB}" \
    --gradient_accumulation_steps="${GA}" &
TRPID=$!

CKPT="${OUT}/checkpoint-${STOP}"
REACHED=0
while kill -0 "${TRPID}" 2>/dev/null; do
    if [ -d "${CKPT}" ]; then
        REACHED=1
        echo ""
        echo "=== reached step ${STOP} after $(( $(date +%s) - t0 ))s -- stopping the run ==="
        kill -TERM -"${TRPID}" 2>/dev/null
        for _ in $(seq 1 20); do kill -0 "${TRPID}" 2>/dev/null || break; sleep 3; done
        kill -KILL -"${TRPID}" 2>/dev/null
        break
    fi
    sleep 20
done
wait "${TRPID}" 2>/dev/null
rc=$?
elapsed=$(( $(date +%s) - t0 ))

########## summary ##########
echo ""
echo "########## ${TAG} summary ##########"
echo "  reached step ${STOP}: $([ "${REACHED}" -eq 1 ] && echo YES || echo "NO -- the run ended on its own, rc=${rc}")"
echo "  wall: ${elapsed}s   GPU-h: $(awk -v s="${elapsed}" 'BEGIN{printf "%.1f", s/3600*4}')"
[ "${REACHED}" -eq 1 ] && echo "  s/step: $(awk -v s="${elapsed}" -v n="${STOP}" 'BEGIN{printf "%.3f", s/n}')"

echo ""
echo "  eval_loss_mask trajectory (degenerate line 1.3703):"
grep -o "'eval_loss_mask': [0-9.]*" "workflows/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null \
  | awk -F': ' 'NR%10==1 || NR<=3 {printf "    %s\n", $2}' | tail -20

echo ""
echo "=== done @ $(date -Is) rc=${rc} ==="
