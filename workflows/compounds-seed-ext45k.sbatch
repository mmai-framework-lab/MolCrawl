#!/bin/bash
#SBATCH --job-name=mc-seed-ext45k
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Seed the 45,000-step output directories from the 30,000-step ones.
#
# An extension resumes from a COPY of the last checkpoint with scheduler.pt removed.
# HF restores the scheduler from that file and LambdaLR.load_state_dict does
# self.__dict__.update(state_dict), which brings back base_lrs and last_epoch -- so a
# resume that keeps it ignores the config's learning_rate and warmup_steps entirely.
# optimizer.pt is kept: the moments are what a continuation needs.
#
# The copy, rather than extending in place, is what keeps the 30,000-step record. The
# values reported on 2026-10-27 point at checkpoints in those directories, and
# retention keeps only the best ten -- continuing in the same directory would prune
# them as new ones improved.
#
# Nothing here needs a GPU. It runs on a compute node because a multi-gigabyte copy
# does not belong on a login node; xlarge's checkpoint alone is about 17 GB.
#
# Submit with:
#   MODEL_OUTPUT_ROOT=<root> sbatch --account=<account> \
#     workflows/compounds-seed-ext45k.sbatch
set -euo pipefail

: "${MODEL_OUTPUT_ROOT:?MODEL_OUTPUT_ROOT=<the learning_source_*_runs root>}"
BERT_OUT="${MODEL_OUTPUT_ROOT}/compounds/bert-output"
FROM_STEP="${FROM_STEP:-30000}"

# arm : the directory holding the finished run : the directory to create
ARMS="small:compounds-small-lr1e3-seed42-ext30k:compounds-small-lr1e3-seed42-ext45k
medium:compounds-medium-lr2p8e4-seed42-ext30k:compounds-medium-lr2p8e4-seed42-ext45k
large:compounds-large-lr1p5e4-seed42-ext30k:compounds-large-lr1p5e4-seed42-ext45k
xlarge:compounds-xl-lr4p5e5-seed42-ext30k:compounds-xl-lr4p5e5-seed42-ext45k"

echo "=== seeding the 45,000-step directories from step ${FROM_STEP} @ $(date -Is) ==="
echo "root: ${BERT_OUT}"
echo

# Refuse the whole job before copying anything if any arm is not ready. A partial
# seeding would leave some arms launchable and some not, which is the state that is
# hardest to read afterwards.
for spec in ${ARMS}; do
  arm=${spec%%:*}; rest=${spec#*:}; src=${rest%%:*}; dst=${rest##*:}
  src_ckpt="${BERT_OUT}/${src}/checkpoint-${FROM_STEP}"
  [ -d "${src_ckpt}" ] || { echo "REFUSING: ${src_ckpt} does not exist" >&2; exit 1; }
  [ -f "${src_ckpt}/optimizer.pt" ] || {
    echo "REFUSING: ${src_ckpt}/optimizer.pt is missing, so there are no moments to" \
         "continue from" >&2; exit 1; }
  [ -f "${src_ckpt}/trainer_state.json" ] || {
    echo "REFUSING: ${src_ckpt}/trainer_state.json is missing, so the schedule this" \
         "checkpoint was trained under cannot be read" >&2; exit 1; }
  # All four ranks wrote their RNG state. A missing rng_state_0.pth is the signature
  # of the rank divergence that killed six runs in October.
  for r in 0 1 2 3; do
    [ -f "${src_ckpt}/rng_state_${r}.pth" ] || {
      echo "REFUSING: ${src_ckpt}/rng_state_${r}.pth is missing; this checkpoint was" \
           "not written by all four ranks" >&2; exit 1; }
  done
  if [ -e "${BERT_OUT}/${dst}" ]; then
    echo "REFUSING: ${BERT_OUT}/${dst} already exists. Move it aside under a changed" \
         "name rather than writing into it." >&2
    exit 1
  fi
  echo "ready: ${arm}  ${src}/checkpoint-${FROM_STEP} -> ${dst}/checkpoint-${FROM_STEP}"
done
echo

for spec in ${ARMS}; do
  arm=${spec%%:*}; rest=${spec#*:}; src=${rest%%:*}; dst=${rest##*:}
  src_ckpt="${BERT_OUT}/${src}/checkpoint-${FROM_STEP}"
  dst_ckpt="${BERT_OUT}/${dst}/checkpoint-${FROM_STEP}"

  echo "--- ${arm}: $(du -sh "${src_ckpt}" | cut -f1)"
  mkdir -p "${BERT_OUT}/${dst}"
  cp -a "${src_ckpt}" "${dst_ckpt}"
  rm -f "${dst_ckpt}/scheduler.pt"

  # State what was produced rather than trusting that it was.
  [ -f "${dst_ckpt}/scheduler.pt" ] && {
    echo "FAILED: ${dst_ckpt}/scheduler.pt survived the removal" >&2; exit 1; }
  [ -f "${dst_ckpt}/optimizer.pt" ] || {
    echo "FAILED: ${dst_ckpt}/optimizer.pt did not copy" >&2; exit 1; }
  cmp -s "${src_ckpt}/trainer_state.json" "${dst_ckpt}/trainer_state.json" || {
    echo "FAILED: ${dst_ckpt}/trainer_state.json differs from its source" >&2; exit 1; }
  echo "    copied, scheduler.pt removed, optimizer.pt kept"
  echo "    $(ls "${dst_ckpt}" | tr '\n' ' ')"
done

echo
echo "=== seeded @ $(date -Is) ==="
