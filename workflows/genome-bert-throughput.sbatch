#!/bin/bash
#SBATCH --job-name=mc-gen-bert-tput
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Why the genome BERT micro-batch matters.
#
# The LR probe measured 3.7 s/it at the shipped 8 x 80. Per token that is ~10x
# slower than genome GPT-2 on the same cluster, which the model size does not
# explain -- both are 12 layers / 768 wide and the vocab is 10, so the embedding
# is negligible. What differs is the shape of one forward pass: 8 x 512 = 4,096
# tokens here against 320 x 1024 = 327,680 for GPT-2, about 1/80. GPT-2 had the
# same defect and got 4x back by moving from 8 x 80 to 320 x 8.
#
# At 3.7 s/it the 21-subset BERT campaign is ~9,600 GPU-h against GPT-2's ~830,
# and one subset needs 4.76 days, past the 4-day MaxTime. So this is not about
# the probe; it decides the shape of the campaign.
#
# Every arm holds the effective global batch at 2,560 (MB x GA x 4 GPU), so the
# training trajectory is nominally identical and only execution efficiency
# changes. Nothing here is a research value.
#
# Measurement discipline -- micro-benchmarks have misdiagnosed this workstream
# twice before:
#   - real config, real training entry point, not a synthetic loop
#   - every arm on ONE node, run sequentially, so nothing runs alongside
#   - the shipped 8 x 80 runs FIRST and LAST; if those two disagree the node
#     drifted and the whole comparison is void
#   - timing is taken between step 10 and the last step, so model init, dataset
#     open and the first-step overhead are excluded
#   - SMOKE_MAX_STEPS keeps eval from firing (eval_interval becomes 50 > steps),
#     so this measures training throughput alone
#
# Required: GENOME_SOURCE_ROOT. Optional: GENOME_SUBSET, STEPS.

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

SUBSET="${GENOME_SUBSET:-mammal_centered}"
STEPS="${STEPS:-40}"
CONFIG=molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py
OUT_BASE="${RUNS_ROOT}/bert-tput"

export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
export GENOME_SUBSET="${SUBSET}"
export SUBSET_BERT_LR=0.0001
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subset=${SUBSET} steps=${STEPS} ==="
echo "=== all arms hold global batch = MB x GA x 4 = 2,560 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

RESULTS=()
run_arm () {
    local label="$1" mb="$2" ga="$3"
    local out="${OUT_BASE}/${label}"
    local log="tmp/tput-${SLURM_JOB_ID:-0}-${label}.log"
    mkdir -p "${out}" tmp
    echo ""
    echo "########## ${label}: MB=${mb} GA=${ga} -> global $((mb * ga * 4)) ##########"

    SMOKE_MAX_STEPS="${STEPS}" "${PYROOT}/miniconda/bin/torchrun" --standalone --nproc_per_node=4 \
        molcrawl/models/bert/main.py "${CONFIG}" \
        --model_path="${out}" \
        --batch_size="${mb}" \
        --gradient_accumulation_steps="${ga}" \
        > "${log}" 2>&1
    local rc=$?

    if [ ${rc} -ne 0 ]; then
        local why="rc=${rc}"
        grep -qi "out of memory\|CUDA out of memory" "${log}" && why="OOM"
        echo "  FAILED (${why})"
        grep -i "error\|out of memory" "${log}" | head -3 | sed 's/^/    /'
        RESULTS+=("${label}|${mb}|${ga}|FAILED(${why})|-")
        rm -rf "${out}"
        return
    fi

    # Seconds per step between step 10 and the final step, from tqdm's own clock.
    # tqdm writes "  17/40 [00:52<01:12,  3.03s/it]"; take the elapsed field of
    # the first sample at/after step 10 and of the last, and divide by the steps
    # between them. Doing it in python keeps the MM:SS / HH:MM:SS split honest.
    local sper
    sper=$("${PYROOT}/miniconda/bin/python" - "${log}" "${STEPS}" <<'PYTPUT'
import re
import sys

path, steps = sys.argv[1], int(sys.argv[2])
pat = re.compile(rf"(\d+)/{steps} \[(\d+):(\d+)(?::(\d+))?<")
pts = {}
for m in pat.finditer(open(path, errors="ignore").read().replace("\r", "\n")):
    a, b, c = int(m.group(2)), int(m.group(3)), m.group(4)
    pts[int(m.group(1))] = a * 3600 + b * 60 + int(c) if c else a * 60 + b
ks = [k for k in sorted(pts) if k >= 10]
if len(ks) >= 2:
    print(f"{(pts[ks[-1]] - pts[ks[0]]) / (ks[-1] - ks[0]):.3f}")
PYTPUT
)
    [ -n "${sper}" ] || sper="?"
    echo "  ${sper} s/step  (steps 10-${STEPS})"
    RESULTS+=("${label}|${mb}|${ga}|OK|${sper}")
    rm -rf "${out}"
}

# Shipped setting first and last: if they disagree, the node drifted.
run_arm "a-shipped-8x80"  8   80
run_arm "b-40x16"         40  16
run_arm "c-160x4"         160 4
run_arm "d-320x2"         320 2
run_arm "e-repeat-8x80"   8   80

echo ""
echo "########## results ##########"
printf "  %-16s %6s %5s %-16s %10s %12s\n" arm MB GA status "s/step" "21本 GPU-h"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r label mb ga status sper <<< "${r}"
    gpuh="-"
    [ "${sper}" != "-" ] && [ "${sper}" != "?" ] && \
        gpuh=$(awk -v s="${sper}" 'BEGIN{printf "%.0f", 111230*s/3600*4*21}')
    printf "  %-16s %6s %5s %-16s %10s %12s\n" "${label}" "${mb}" "${ga}" "${status}" "${sper}" "${gpuh}"
done
echo ""
echo "  a と e が食い違う場合、ノードがドリフトしているので比較は無効"
echo "=== done @ $(date -Is) ==="
