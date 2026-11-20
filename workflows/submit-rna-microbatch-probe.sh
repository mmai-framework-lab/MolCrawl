#!/bin/bash
# Time one GPU's block count, per size, for both frameworks -- §4 and §9 of the
# rna batch-size directive of 2026-11-20.
#
# §4 fixes the global batch and asks which micro-batch runs it fastest, measured
# per size rather than carried over from another corpus, and says not to leave BERT
# at 8. The answer decides the accumulation count of all 68 arms, and on compounds
# the wrong shape cost 1.84 to 1.97 times the step time.
#
# Measured at global batch 256, the smallest of the grid's five. Per-token cost is
# a property of the micro shape, and the synchronisation cost per token at a given
# batch is the same whatever the micro-batch, so the ranking carries to the other
# four batches; 256 is where a step is cheapest to measure and where the grid runs
# the most steps. Each arm prints its equivalent at 2,560 as well, which is what
# the existing rna runs and the compounds measurement are quoted in.
#
# Usage:
#   LEARNING_SOURCE_DIR=... PROBE_OUT=... [SUBMIT_RECORD=...] [DRY_RUN=1] \
#     workflows/submit-rna-microbatch-probe.sh
#
# Env:
#   LEARNING_SOURCE_DIR  the learning_source root holding rna/
#   PROBE_OUT            scratch root for probe outputs, outside every input tree
#   SUBMIT_RECORD        TSV of what was submitted (default under PROBE_OUT)
#   BERT_MICROS_<SIZE>   override a size's candidate list, e.g. BERT_MICROS_XL="4 8"
#   GPT2_MICROS_<SIZE>   the same for GPT-2
#   STEPS                steps per arm (default 300)
#   DRY_RUN              1 to print the sbatch lines and submit nothing
#
# One arm at a time within a framework, chained with afterany: two arms on one node
# read the same objects and one warms them for the other (measured on BERT in job
# 117320), and an arm that does not fit in memory is an answer, so the chain must
# not stop on it. BERT and GPT-2 run as two chains side by side because they read
# different files -- BERT the Arrow dataset, GPT-2 the uint16 memmap.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${PROBE_OUT:?set PROBE_OUT to a scratch root outside every input tree}"
STEPS="${STEPS:-300}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_RECORD="${SUBMIT_RECORD:-${PROBE_OUT}/rna-microbatch-probe.tsv}"

# The candidates. Every value listed can build 256 at 4 GPUs: BERT needs
# micro x 4 to divide 256, and so does GPT-2, since nanoGPT splits the
# accumulation count across the ranks. The current value of each size is in its
# list, so the table says what staying put costs. The largest value per size is
# capped where memory makes it pointless to queue -- xl holds 1,561M parameters
# against large's 606M, and an arm that does not fit reports that and moves on.
BERT_MICROS_SMALL="${BERT_MICROS_SMALL:-8 16 32 64}"
BERT_MICROS_MEDIUM="${BERT_MICROS_MEDIUM:-8 16 32 64}"
BERT_MICROS_LARGE="${BERT_MICROS_LARGE:-8 16 32 64}"
BERT_MICROS_XL="${BERT_MICROS_XL:-4 8 16 32}"
GPT2_MICROS_SMALL="${GPT2_MICROS_SMALL:-8 16 32 64}"
GPT2_MICROS_MEDIUM="${GPT2_MICROS_MEDIUM:-8 16 32 64}"
GPT2_MICROS_LARGE="${GPT2_MICROS_LARGE:-4 8 16 32}"
GPT2_MICROS_XL="${GPT2_MICROS_XL:-4 8 16}"

# The adopted arm of each size, so the probe runs the production shape and not a
# base config nothing runs.
bert_config() {
  case "$1" in
    small)  echo "molcrawl/tasks/pretrain/configs/rna/bert_small_lr3e4.py" ;;
    medium) echo "molcrawl/tasks/pretrain/configs/rna/bert_medium_lr3e4.py" ;;
    large)  echo "molcrawl/tasks/pretrain/configs/rna/bert_large_lr1p7e4.py" ;;
    xl)     echo "molcrawl/tasks/pretrain/configs/rna/bert_xl_lr2p5e5.py" ;;
  esac
}

# §2's step count at batch 256, for the GPU-h figure each arm prints.
RUN_STEPS=1209600
GLOBAL=256
GPUS=4

mkdir -p "${PROBE_OUT}" workflows/slurm-logs
if [ "${DRY_RUN}" != "1" ] && [ ! -f "${SUBMIT_RECORD}" ]; then
  printf 'submitted_at\tarch\tsize\tmicro\tglobal\taccum\tsteps\tjob_id\tdepends_on\tcommit\n' \
    > "${SUBMIT_RECORD}"
fi
COMMIT="$(git rev-parse HEAD)"

submit() {  # arch size micro accum dep -> job id on stdout
  local arch="$1" size="$2" micro="$3" accum="$4" dep="$5" jid
  local name="mc-rna-tput-${arch}-${size}-mb${micro}"
  local -a line
  # The values go into the environment of the sbatch call and are carried by
  # --export=ALL, not written into --export's own comma-separated list: that list
  # splits on commas, and EXTRA_ARGS also holds spaces. A path with a comma in it
  # silently truncated a launch on 2026-11-10 and the same shape would mangle this.
  if [ "${arch}" = "bert" ]; then
    line=(env
      MODALITY=rna
      LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}"
      TPUT_OUT_BASE="${PROBE_OUT}"
      ARM_CONFIG="$(bert_config "${size}")"
      STEPS="${STEPS}"
      RUN_STEPS="${RUN_STEPS}"
      LABEL="${size}-mb${micro}"
      EXTRA_ARGS="--batch_size=${micro} --gradient_accumulation_steps=${accum} --expected_global_batch=${GLOBAL}"
      sbatch --job-name="${name}" ${dep:+--dependency=afterany:${dep}} --export=ALL
      workflows/bert-tput-arm.sbatch)
  else
    line=(env
      SIZE="${size}"
      MB="${micro}"
      GLOBAL="${GLOBAL}"
      STEPS="${STEPS}"
      PROBE_OUT="${PROBE_OUT}"
      LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}"
      RUN_STEPS="${RUN_STEPS}"
      sbatch --job-name="${name}" ${dep:+--dependency=afterany:${dep}} --export=ALL
      workflows/rna-gpt2-tput-arm.sbatch)
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    printf '  %s\n' "${line[*]}" >&2
    echo "DRY"
    return
  fi
  jid="$("${line[@]}" | awk '{print $NF}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${arch}" "${size}" "${micro}" "${GLOBAL}" "${accum}" "${STEPS}" \
    "${jid}" "${dep:-}" "${COMMIT}" >> "${SUBMIT_RECORD}"
  echo "${jid}"
}

echo "=== rna micro-batch probe: global ${GLOBAL}, ${STEPS} steps per arm,"
echo "    ${GPUS} GPUs, commit ${COMMIT:0:8}, record ${SUBMIT_RECORD}"
[ "${DRY_RUN}" = "1" ] && echo "    DRY_RUN: nothing is submitted"

total=0
for arch in bert gpt2; do
  dep=""
  echo "--- ${arch} (one arm at a time)"
  for size in small medium large xl; do
    upper="$(echo "${size}" | tr '[:lower:]' '[:upper:]')"
    if [ "${arch}" = "bert" ]; then
      eval "micros=\${BERT_MICROS_${upper}}"
    else
      eval "micros=\${GPT2_MICROS_${upper}}"
    fi
    for micro in ${micros}; do
      if [ $((GLOBAL % (micro * GPUS))) -ne 0 ]; then
        echo "    ${size} micro ${micro}: skipped, ${micro} x ${GPUS} does not divide ${GLOBAL}"
        continue
      fi
      accum=$((GLOBAL / (micro * GPUS)))
      [ "${arch}" = "gpt2" ] && accum=$((GLOBAL / micro))
      jid="$(submit "${arch}" "${size}" "${micro}" "${accum}" "${dep}")"
      echo "    ${size} micro ${micro} x accum ${accum}: job ${jid}${dep:+ (after ${dep})}"
      dep="${jid}"
      total=$((total + 1))
    done
  done
done
echo "=== ${total} arms"
