#!/bin/bash
#SBATCH --job-name=mc-probe-extend-resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Settle what an extension's resume restores, on a compute node.
#
# scripts/probe_extend_resume_scheduler.py runs a tiny CPU model three times:
# resuming from a checkpoint whose scheduler.pt is kept, removed, or rewritten.
# It reports, for each, whether the Adam moments arrived, whether the run uses
# the extension's own learning rate, and whether the warmup ramped.
#
# This exists because bert-grid.sbatch instructs "remove scheduler.pt -- keep
# optimizer.pt", while in transformers 4.45.1 one `if` guards both loads. If the
# probe says the moments do not arrive, every extension taken that way --
# compounds and molecule_nat_lang so far -- continued without them.
#
# One GPU because the partition requires one; the probe runs on CPU.

set -u
: "${MOLCRAWL_PYTHON:?set MOLCRAWL_PYTHON to the research env python}"
[ -x "${MOLCRAWL_PYTHON}" ] || { echo "MOLCRAWL_PYTHON=${MOLCRAWL_PYTHON} is not executable" >&2; exit 1; }

OUT_JSON="${PROBE_JSON:-tmp/probe-extend-resume-$(date +%Y%m%d-%H%M%S).json}"
mkdir -p "$(dirname "${OUT_JSON}")"

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
echo "=== json:     ${OUT_JSON} ==="
"${MOLCRAWL_PYTHON}" -c 'import transformers, torch; print("transformers", transformers.__version__, "torch", torch.__version__)'

PYTHONPATH="$PWD" "${MOLCRAWL_PYTHON}" scripts/probe_extend_resume_scheduler.py \
    --pre-steps 20 --add-steps 10 --peak 1e-3 --warmup 5 \
    --json-out "${OUT_JSON}"
rc=$?
echo "=== probe rc=${rc} ==="
exit ${rc}
