#!/bin/bash
#SBATCH --job-name=mc-rna-gpt2-tput
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# How long one GPT-2 step costs at one micro-batch, for one rna size.
#
# §4 of the rna batch-size directive asks for one GPU's block count to be the
# fastest shape that can build every batch of the grid, measured per size rather
# than carried over. The global batch is held while the micro-batch moves, so what
# changes is how much is resident at once and how often the ranks synchronise --
# an execution choice, and the one that decides whether 68 arms take weeks or
# months.
#
#   SIZE=large MB=16 PROBE_OUT=... LEARNING_SOURCE_DIR=... \
#     sbatch workflows/rna-gpt2-tput-arm.sbatch
#
# Env:
#   SIZE                 small|medium|large|xl
#   MB                   one GPU's block count to time
#   GLOBAL               the global batch to hold (default 256, the grid's smallest)
#   STEPS                iterations to run (default 300)
#   PROBE_OUT            scratch root outside every input tree
#   LEARNING_SOURCE_DIR  the learning_source root holding rna/
#   RUN_STEPS            optional: iterations of a real arm, for the GPU-h figure
#
# One arm per job, never two side by side: two arms on one node read the same
# Lustre objects and one warms them for the other (measured on BERT in job 117320).
# Chain them with --dependency.
#
# nanoGPT divides the accumulation count across ranks, so the global batch is
# micro x accumulation with no GPU factor, and the accumulation count has to stay
# divisible by the 4 ranks. compile stays off: the production rna GPT-2 runs pass
# --compile=False (no Triton on these nodes), so timing it on would measure a
# shape nothing runs.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${SIZE:?set SIZE to small|medium|large|xl}"
: "${MB:?set MB to the per-GPU block count}"
: "${PROBE_OUT:?set PROBE_OUT to a scratch root outside any input tree}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
[ -d "${LEARNING_SOURCE_DIR}/rna" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no rna/ under it" >&2; exit 1; }

GLOBAL="${GLOBAL:-256}"
STEPS="${STEPS:-300}"
RANKS=4
CONFIG="molcrawl/tasks/pretrain/configs/rna/gpt2_${SIZE}.py"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

if [ $((GLOBAL % (MB * RANKS))) -ne 0 ]; then
  echo "global ${GLOBAL} is not a multiple of micro ${MB} x ${RANKS} ranks:" \
       "the accumulation count would not divide across the ranks" >&2
  exit 1
fi
GA=$((GLOBAL / MB))

if [ -x "./miniconda/bin/torchrun" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
PYBIN="${PYROOT}/miniconda/bin/python"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda" >&2; exit 1; }

LABEL="gpt2-${SIZE}-mb${MB}-g${GLOBAL}"
OUT="${PROBE_OUT}/${LABEL}-${SLURM_JOB_ID:-0}"
LOG="${PROBE_OUT}/${LABEL}-${SLURM_JOB_ID:-0}.log"
mkdir -p "${OUT}" "${PROBE_OUT}" workflows/slurm-logs

export LEARNING_SOURCE_DIR PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HOME="${PROBE_OUT}/hf" XDG_CACHE_HOME="${PROBE_OUT}/xdg" TMPDIR="${OUT}/tmp"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}"

echo "=== ${LABEL}: micro ${MB} x accum ${GA} = ${GLOBAL} blocks, ${STEPS} iters ==="
echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=${RANKS} \
  molcrawl/models/gpt2/train.py "${CONFIG}" \
  --compile=False --early_stopping=False --init_from=scratch \
  --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
  --batch_size="${MB}" --gradient_accumulation_steps="${GA}" \
  --expected_global_batch="${GLOBAL}" \
  --max_iters="${STEPS}" --lr_decay_iters="${STEPS}" --warmup_iters=1 \
  --eval_interval="$((STEPS + 10))" \
  > "${LOG}" 2>&1 || rc=$?

if [ "${rc}" -ne 0 ]; then
  why="rc=${rc}"; grep -qi "out of memory" "${LOG}" && why="OOM"
  echo "VERDICT ${LABEL}: FAILED (${why})"
  grep -iE "error|out of memory|assert" "${LOG}" | head -5 | sed 's/^/  /'
  rm -rf "${OUT}"; exit 0   # a shape that does not fit is an answer, not a failure
fi

# nanoGPT prints one line per iteration with its own millisecond figure. The first
# iterations carry the allocator warming up, so the window starts at 50 and the
# median is taken rather than the mean -- one slow iteration from a checkpoint or a
# page fault should not move the figure.
"${PYBIN}" - "${LOG}" "${LABEL}" "${MB}" "${GLOBAL}" "${RUN_STEPS:-0}" <<'PYTPUT'
import re
import statistics
import sys

path, label, mb, global_batch, run_steps = sys.argv[1:6]
mb, global_batch, run_steps = int(mb), int(global_batch), int(run_steps)
times = [(int(i), float(ms)) for i, ms in
         re.findall(r"iter (\d+): loss [\d.]+, time ([\d.]+)ms", open(path, errors="ignore").read())]
window = [ms for i, ms in times if i >= 50]
if len(window) < 3:
    print(f"VERDICT {label}: no usable timing samples ({len(times)} iteration lines)")
    raise SystemExit
sper = statistics.median(window) / 1000.0
tokens = global_batch * 1024
print(f"VERDICT {label}: OK  {sper:.3f} s/iter  ({len(window)} samples from iter 50)")
print(f"  {tokens / sper / 1e6:.3f} M tokens/s at global {global_batch}")
print(f"  equivalent at global 2,560: {sper * 2560 / global_batch:.3f} s/iter")
if run_steps:
    hours = run_steps * sper / 3600
    print(f"  one {run_steps:,}-iteration arm: {hours:.1f} h, {hours * 4:,.0f} GPU-h")
PYTPUT
rm -rf "${OUT}"
echo "=== done @ $(date -Is) ==="
