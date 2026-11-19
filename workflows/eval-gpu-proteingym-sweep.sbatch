#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-proteingym-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# ProteinGym full-assay sweep wrapper. Defaults to the
# ``protein_sequence × gpt2 × medium`` model so a single sbatch run
# produces the canonical 217-assay ProteinGym table. Override with
# MODEL_PATH / MODEL_SLUG / ARCH if you want a different combo.

set -uo pipefail
# shellcheck disable=SC1091
source workflows/common_functions.sh

# Require LEARNING_SOURCE_DIR to be set in the environment (typically by
# common-env / per-user bashrc). check_learning_source_dir exits with a
# clear error message if it is unset, instead of silently defaulting to
# a path that only exists on one developer's machine.
check_learning_source_dir

# Defaults: protein × gpt2 × medium (the canonical ProteinGym sweep target).
export MODEL_PATH="${MODEL_PATH:-${LEARNING_SOURCE_DIR}/protein_sequence/gpt2-output/protein_sequence-medium/ckpt.pt}"
export MODEL_SLUG="${MODEL_SLUG:-protein_sequence-gpt2-medium}"
export ARCH="${ARCH:-gpt2}"
export MODALITY="${MODALITY:-protein_sequence}"
export DEVICE="${DEVICE:-cuda}"
export BOOTSTRAP="${BOOTSTRAP:-100}"
# ``-`` (no colon) so an explicitly-empty MAX_EXAMPLES (passed as
# --export=...,MAX_EXAMPLES= for a FULL-assay run) stays empty instead of
# falling back to the 200-variant subsample. Unset still defaults to 200.
export MAX_EXAMPLES="${MAX_EXAMPLES-200}"
# Forward the official-run knobs when set (empty = evaluator/sweep defaults):
# PG_DIR selects the assay CSV directory (official vs unpacked); SCORING_METHOD
# picks pll_delta vs masked_marginals.
export PG_DIR="${PG_DIR:-}"
export SCORING_METHOD="${SCORING_METHOD:-}"

echo "[pg-sweep-gpu] node:        $(hostname)"
echo "[pg-sweep-gpu] LSD:         ${LEARNING_SOURCE_DIR}"
echo "[pg-sweep-gpu] MODEL_PATH:  ${MODEL_PATH}"
echo "[pg-sweep-gpu] MODEL_SLUG:  ${MODEL_SLUG}"
nvidia-smi 2>/dev/null | head -10 || rocm-smi 2>/dev/null | head -10 || true
echo

bash workflows/eval-proteingym-sweep.sh

echo
echo "[pg-sweep-gpu] DONE. Rebuild dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
