#!/bin/bash
# Exercise workflows/bert-grid.sbatch's segment guards with fake checkpoints, no GPU.
#
# The order submits every segment up front with --dependency=afterany, so segments beyond
# the one that finishes the run will start. This checks what they do, and what an honest
# continuation does, by running the real launcher with torchrun replaced by a no-op and
# checkpoints written by hand under a scratch MODEL_OUTPUT_ROOT.
#
#   bash scripts/check_bert_grid_segments.sh <main checkout> <scratch dir> <TIMEOUT job id> <COMPLETED job id>
set -u
MAIN="$1"; SCRATCH="$2"; TIMEOUT_JOB="$3"; COMPLETED_JOB="$4"
REPO="$(pwd)"
CFG=molcrawl/tasks/pretrain/configs/compounds/bert_small_lr1e3.py
ROOT="${SCRATCH}/segtest-$$"
OUT="${ROOT}/compounds/bert-output/compounds-small-lr1e3"
STUB="${ROOT}/bin"; mkdir -p "${STUB}"
printf '#!/bin/sh\necho "torchrun (stub) $*"\n' > "${STUB}/torchrun"
printf '#!/bin/sh\necho "GPU (stub)"\n' > "${STUB}/nvidia-smi"
chmod +x "${STUB}/torchrun" "${STUB}/nvidia-smi"

ckpt() {  # $1 step, $2 max_steps
  mkdir -p "${OUT}/checkpoint-$1"
  printf '{"global_step": %s, "max_steps": %s, "log_history": []}\n' "$1" "$2" > "${OUT}/checkpoint-$1/trainer_state.json"
}
run() {  # name, SEGMENT, extra env
  local name="$1" seg="$2"; shift 2
  local log="${ROOT}/${name}.log" t0 rc
  t0=$(date +%s)
  env "$@" SLURM_SUBMIT_DIR="${REPO}" SLURM_JOB_ID="test-${name}" PATH="${STUB}:${PATH}" TORCHRUN="${STUB}/torchrun" \
      MODALITY=compounds CONFIG="${CFG}" SEGMENT="${seg}" \
      LEARNING_SOURCE_DIR="${MAIN}/learning_source_20260805_compounds_packed" MODEL_OUTPUT_ROOT="${ROOT}" \
      GPT2_TOKENIZER_DIR="${MAIN}/assets/tokenizers/gpt2" \
      bash workflows/bert-grid.sbatch > "${log}" 2>&1
  rc=$?
  printf '%-44s rc=%s  %ss  %s\n' "${name}" "${rc}" "$(( $(date +%s) - t0 ))" \
    "$( (grep -hE 'REFUSING' "${log}" || grep -hE 'torchrun \(stub\)' "${log}") | head -1 | cut -c1-130)"
}

rm -rf "${OUT}"
run "1 fresh start, empty dir"                 1
ckpt 5000 15000
run "2 continuation after TIMEOUT"             2 PREV_JOB="${TIMEOUT_JOB}"
run "3 continuation after COMPLETED"           2 PREV_JOB="${COMPLETED_JOB}"
ckpt 15000 15000
run "4 extra segment, run already at max_steps" 3 PREV_JOB="${TIMEOUT_JOB}"
rm -f "${OUT}/segments.log"
run "5 segment with no record of the previous" 4
run "6 SEGMENT=1 over existing checkpoints"    1
rm -rf "${ROOT}"
