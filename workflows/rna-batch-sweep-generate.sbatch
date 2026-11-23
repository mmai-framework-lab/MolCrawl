#!/bin/bash
#SBATCH --job-name=mc-rna-bsweep-generate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Build the 68 configs of the rna batch-size grid from the measured block counts,
# check them, and print the launch plan.
#
#   TABLE=... sbatch --dependency=afterok:<probe table job> \
#     workflows/rna-batch-sweep-generate.sbatch
#
# Queued behind the probe so the grid is ready to launch whenever the measurement
# lands, rather than waiting for someone to notice that it has. It writes configs
# and reads logs; it starts no training. The launch itself stays a separate,
# approved step -- see workflows/submit-rna-batch-sweep.sh.
#
# Env:
#   TABLE            the probe table TSV the block counts come from
#   TABLE2           a second table, when the two frameworks were measured apart
#                    (BERT on 2026-11-20, GPT-2 on 2026-11-23); it wins on overlap
#   ARMS_OUT         where to write the arm inventory (default: beside this log)
#   LEARNING_SOURCE_DIR, MODEL_OUTPUT_ROOT, SLURM_ACCOUNT
#                    passed through to the dry run of the submitter, which needs
#                    them to print the plan
#
# One GPU because the partition requires one; this writes text files.

set -u
: "${TABLE:?export TABLE to the probe table TSV}"
[ -f "${TABLE}" ] || { echo "no such table: ${TABLE}" >&2; exit 1; }
ARMS_OUT="${ARMS_OUT:-workflows/slurm-logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-0}-arms.tsv}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== built:    $(date -Is) ==="
echo "=== table:    ${TABLE} ==="
rc=0

echo
echo "--- 1/5 the arm inventory, at the measured block counts"
"${PYTHON}" scripts/rna_batch_sweep_arms.py --micro-batch-set measured \
  --measured-from "${TABLE}" ${TABLE2:+--measured-from "${TABLE2}"} \
  --out "${ARMS_OUT}" || rc=$?

echo
echo "--- 2/5 the 68 configs"
[ "${rc}" -eq 0 ] && { "${PYTHON}" scripts/rna_batch_sweep_configs.py \
  --arms "${ARMS_OUT}" || rc=$?; }

echo
echo "--- 3/5 disk against the inventory"
[ "${rc}" -eq 0 ] && { "${PYTHON}" scripts/rna_batch_sweep_configs.py \
  --arms "${ARMS_OUT}" --check || rc=$?; }

echo
echo "--- 4/5 unit tests"
"${PYTHON}" -m pytest tests/unit/test_rna_batch_sweep.py -q -p no:cacheprovider || rc=$?

echo
echo "--- 5/5 the launch plan (dry run; nothing is queued)"
if [ -n "${LEARNING_SOURCE_DIR:-}" ] && [ -n "${MODEL_OUTPUT_ROOT:-}" ] \
   && [ -n "${SLURM_ACCOUNT:-}" ] && [ "${rc}" -eq 0 ]; then
  ARMS="${ARMS_OUT}" PROBE_TABLE="${TABLE}" DRY_RUN=1 \
    workflows/submit-rna-batch-sweep.sh | grep -vE "^    \[dry-run\]" || rc=$?
else
  echo "    skipped: set LEARNING_SOURCE_DIR, MODEL_OUTPUT_ROOT and SLURM_ACCOUNT"
fi

echo
echo "=== generate rc=${rc}; inventory ${ARMS_OUT} ==="
echo "=== nothing was launched. To launch, with approval:"
echo "===   ARMS=${ARMS_OUT} PROBE_TABLE=${TABLE} ... workflows/submit-rna-batch-sweep.sh"
exit ${rc}
