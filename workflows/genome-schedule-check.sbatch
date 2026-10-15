#!/bin/bash
#SBATCH --job-name=mc-gen-sched-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Print the schedule a genome config resolves to, without training.
#
# WHY. max_iters, lr_decay_iters and warmup_iters are derived inside the config
# from the dataset's row count and from env vars. A run that resolves them wrong
# does not fail -- it trains on the wrong schedule for a day and a half and the
# mistake is found afterwards in a log. This resolves the config the same way
# train.py does and prints the three numbers, so a launch can be checked first.
#
# Required: GENOME_SOURCE_ROOT, GENOME_SUBSET.
# Optional: LR, EPOCHS, DECAY_EPOCHS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
: "${GENOME_SUBSET:?set GENOME_SUBSET to the subset to resolve}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LEARNING_SOURCE_DIR="${GENOME_SOURCE_ROOT}"
export SUBSET_GPT2_LR="${LR:-0.0001}"
export SUBSET_GPT2_EPOCHS="${EPOCHS:-3}"
[ -n "${DECAY_EPOCHS:-}" ] && export SUBSET_GPT2_LR_DECAY_EPOCHS="${DECAY_EPOCHS}"

echo "=== node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
echo "=== subset=${GENOME_SUBSET} epochs=${SUBSET_GPT2_EPOCHS} decay_epochs=${SUBSET_GPT2_LR_DECAY_EPOCHS:-same as epochs} ==="

"${PYTHON}" - <<'PY'
import runpy
import os

ns = runpy.run_path(
    "molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py")
keys = ("max_iters", "lr_decay_iters", "warmup_iters", "learning_rate",
        "min_lr", "eval_interval", "eval_iters", "batch_size", "block_size",
        "gradient_accumulation_steps", "expected_global_batch", "seed")
for k in keys:
    v = ns.get(k)
    print(f"  {k:30s} {v:,}" if isinstance(v, int) else f"  {k:30s} {v}")
print(f"  {'data_seed declared':30s} {'data_seed' in ns}")
print(f"  {'out_dir':30s} {ns.get('out_dir')}")
d, m = ns["lr_decay_iters"], ns["max_iters"]
print(f"\n  lr_decay_iters / max_iters = {d:,} / {m:,} = {d / m:.4f}")
print(f"  steps held at min_lr after the cosine ends: {max(0, m - d):,}")
PY
rc=$?
echo "---- rc=${rc} ----"
exit ${rc}
