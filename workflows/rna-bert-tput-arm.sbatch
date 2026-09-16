#!/bin/bash
#SBATCH --job-name=rna-bert-tput-arm
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# One throughput arm, one job. Replaces the five-arms-in-one-job harness.
#
# That harness disagreed with itself: its own first/last check on the shipped
# shape read 21.550 against 11.950, 1.80x. The cause is not the node. Every arm
# shares seed 106 and global batch 2,560, so all five draw the SAME first 30 x
# 2,560 = 76,800 rows -- 600.3 MiB of a 262.6 GiB split, 0.223 % of one epoch,
# small enough to stay in page cache. Arm 1 paid the cold read; arms 2-5 did
# not. Measured in job 117247, not inferred.
#
# Two things follow, and both are built in here:
#   - one arm per job, so each arm gets its own allocation and its own cold cache
#   - 300 steps, not 30. The sustained figure was 18.8 s/step out to step 352;
#     a 20-step window does not represent a 40,320-step run.
#
# The first/last same-shape check survives as an operator step: submit the
# as-is arm again after the last one and compare. Across jobs that check now
# measures what it always claimed to -- whether the machine moved under us.
#
# Required: LEARNING_SOURCE_DIR, TPUT_OUT_BASE, ARM_CONFIG. Optional: STEPS, LABEL.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the RNA learning_source root}"
: "${TPUT_OUT_BASE:?set TPUT_OUT_BASE to a scratch root outside any input tree}"
: "${ARM_CONFIG:?set ARM_CONFIG to a configs/rna/bert_small_tput*.py}"
[ -f "${ARM_CONFIG}" ] || { echo "no config at ${ARM_CONFIG}" >&2; exit 1; }

if [ -x "./miniconda/bin/torchrun" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
PYBIN="${PYROOT}/miniconda/bin/python"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda" >&2; exit 1; }

STEPS="${STEPS:-300}"
LABEL="${LABEL:-$(basename "${ARM_CONFIG}" .py)}"
OUT="${TPUT_OUT_BASE}/bert-tput-arm/${LABEL}-${SLURM_JOB_ID:-0}"
LOG="${TPUT_OUT_BASE}/tput-arm-${SLURM_JOB_ID:-0}-${LABEL}.log"
mkdir -p "${OUT}" "${TPUT_OUT_BASE}" workflows/slurm-logs

# The config writes the tokenizer to LEARNING_SOURCE_DIR/rna/custom_tokenizer_bert
# and reads it straight back. That path is shared, so arms running at the same
# time write over each other's directory mid-read: arms 1 and 2 of the first
# attempt (117320, 117321) both died in AutoTokenizer.from_pretrained while arm 3
# was writing. Pre-writing it once per job does not help -- the collision is
# between jobs, not between the four ranks.
#
# So each job gets its own view of the tree: every entry of rna/ symlinked
# through, except custom_tokenizer_bert, which is a real directory this job alone
# writes. The data is not copied and the shared tree is not written to.
OVERLAY="${TPUT_OUT_BASE}/ls-overlay-${SLURM_JOB_ID:-0}"
rm -rf "${OVERLAY}"; mkdir -p "${OVERLAY}/rna"
for e in "${LEARNING_SOURCE_DIR}"/rna/*; do
    b="$(basename "${e}")"
    [ "${b}" = "custom_tokenizer_bert" ] && continue
    ln -s "${e}" "${OVERLAY}/rna/${b}"
done
mkdir -p "${OVERLAY}/rna/custom_tokenizer_bert"
echo "--- per-job learning source overlay: ${OVERLAY} ---"
export LEARNING_SOURCE_DIR="${OVERLAY}"

export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${TPUT_OUT_BASE}/hf-cache-rna"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== arm=${LABEL} node=$(hostname) job=${SLURM_JOB_ID:-none} steps=${STEPS} ==="
echo "=== config=${ARM_CONFIG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
free -g | awk 'NR<=2{print "  mem: "$0}'

# Four ranks writing one directory is also a race -- it killed the first arm of
# job 117065 when rank 1 read a half-written file. Write it once, single process,
# before torchrun. With the overlay above this is now the only writer of that
# path anywhere.
echo "--- materialising the tokenizer once ---"
"${PYBIN}" - "${ARM_CONFIG}" <<'PYTOK'
import sys
src = open(sys.argv[1], encoding="utf-8").read()
exec(compile(src.split("max_steps")[0], sys.argv[1], "exec"), {"__name__": "__main__"})
print("  tokenizer written")
PYTOK

# save_steps must stay a multiple of the eval interval or load_best_model_at_end
# refuses to build TrainingArguments; pushing both past the step count is what
# keeps eval and checkpointing out of the timing.
"${TORCHRUN}" --standalone --nproc_per_node=4 \
    molcrawl/models/bert/main.py "${ARM_CONFIG}" \
    --model_path="${OUT}" \
    --max_steps="${STEPS}" \
    --warmup_steps=10 \
    --log_interval="$((STEPS + 100))" \
    --save_steps="$((STEPS + 100))" \
    > "${LOG}" 2>&1
rc=$?

if [ ${rc} -ne 0 ]; then
    why="rc=${rc}"; grep -qi "out of memory" "${LOG}" && why="OOM"
    echo "FAILED (${why})"; grep -i "error\|out of memory" "${LOG}" | head -5 | sed 's/^/  /'
    rm -rf "${OUT}" "${OVERLAY}"; exit "${rc}"
fi

# Seconds per step from tqdm's own clock, from step 50 to the last step. genome's
# harness took it from step 10; at 300 steps there is room to leave more of the
# warm-up out, and the sustained figure is the one being checked.
"${PYBIN}" - "${LOG}" "${STEPS}" "${LABEL}" <<'PYTPUT'
import re
import sys

path, steps, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pat = re.compile(rf"(\d+)/{steps} \[(\d+):(\d+)(?::(\d+))?<")
pts = {}
for m in pat.finditer(open(path, errors="ignore").read().replace("\r", "\n")):
    a, b, c = int(m.group(2)), int(m.group(3)), m.group(4)
    pts[int(m.group(1))] = a * 3600 + b * 60 + int(c) if c else a * 60 + b
ks = [k for k in sorted(pts) if k >= 50]
if len(ks) < 2:
    print("  no usable timing samples"); raise SystemExit
sper = (pts[ks[-1]] - pts[ks[0]]) / (ks[-1] - ks[0])
print(f"  {label}: {sper:.3f} s/step  (steps {ks[0]}-{ks[-1]})")
print(f"  one 40,320-step run: {40320 * sper / 3600:.1f} h, "
      f"{40320 * sper / 3600 * 4:,.0f} GPU-h")
print(f"  nine runs at this rate: {40320 * sper / 3600 * 4 * 9:,.0f} GPU-h "
      f"(same s/step for all three sizes -- a rough ceiling, measure each)")
# The shape of the curve says whether the window was long enough to settle.
seen = [k for k in sorted(pts) if k >= 50]
for k in seen[:: max(1, len(seen) // 6)]:
    print(f"    step {k:>4}  {pts[k]:>6}s elapsed")
PYTPUT
rm -rf "${OUT}" "${OVERLAY}"
echo "=== done @ $(date -Is) ==="
