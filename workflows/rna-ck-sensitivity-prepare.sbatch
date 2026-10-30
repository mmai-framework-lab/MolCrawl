#!/bin/bash
#SBATCH --job-name=mc-rna-ck-prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Seed a new output directory with a copy of one checkpoint, so a resume can start
# from it. all-checkpoint-metric-judgment-2026-10-30-v3 §9.
#
# main.py resolves the resume point itself: it lists model_path, sorts by step and
# takes the highest. There is no way to ask it for an earlier one, and the runs being
# re-run hold checkpoints up to 120,960. Copying the step the re-run starts from into
# a directory of its own is what makes it the highest -- and it leaves the original
# run untouched, which a symlink would not: BestNCheckpointRetention prunes by step
# and would follow the link out of the new directory.
#
# No GPU. A compute node because nothing heavier than a shell runs on the login node,
# and this copies several GB.
#
#   sbatch --export=ALL,SRC_RUN=<original run dir>,DEST_RUN=<new run dir>,STEP=115300 \
#          workflows/rna-ck-sensitivity-prepare.sbatch
set -eo pipefail

: "${SRC_RUN:?set SRC_RUN to the run directory holding the checkpoint to copy}"
: "${DEST_RUN:?set DEST_RUN to the new run directory (must not exist yet)}"
: "${STEP:?set STEP to the checkpoint step to copy}"

SRC="${SRC_RUN}/checkpoint-${STEP}"
DEST="${DEST_RUN}/checkpoint-${STEP}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== rna ck prepare ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "=== outputs: ${DEST_RUN} ==="
echo "source : ${SRC}"

[ -d "${SRC}" ] || { echo "no such checkpoint: ${SRC}" >&2; exit 1; }
# A directory left behind by a cancelled run holds rng_state_*.pth and nothing to
# resume from; 2026-10-19 produced six of those. Refuse early rather than at step 0.
[ -f "${SRC}/model.safetensors" ] || [ -f "${SRC}/pytorch_model.bin" ] || {
    echo "${SRC} holds no weights" >&2; exit 1; }
[ -f "${SRC}/trainer_state.json" ] || { echo "${SRC} holds no trainer_state.json" >&2; exit 1; }
[ -e "${DEST_RUN}" ] && { echo "${DEST_RUN} already exists; refusing to write into it" >&2; exit 1; }

mkdir -p "${DEST_RUN}"
cp -a "${SRC}" "${DEST}"

# What was copied, verified rather than assumed: the resume reads all three.
for f in trainer_state.json optimizer.pt scheduler.pt; do
    if [ -e "${SRC}/${f}" ]; then
        cmp -s "${SRC}/${f}" "${DEST}/${f}" || { echo "copy differs: ${f}" >&2; exit 1; }
    else
        echo "note: ${f} is not in the source checkpoint"
    fi
done
src_size="$(du -sb "${SRC}" | cut -f1)"
dest_size="$(du -sb "${DEST}" | cut -f1)"
[ "${src_size}" = "${dest_size}" ] || {
    echo "size differs: ${src_size} vs ${dest_size}" >&2; exit 1; }

echo "copied ${src_size} bytes to ${DEST}"
ls -la "${DEST}" | head -20
echo "=== rc=0 @ $(date -Is) ==="
