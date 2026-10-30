#!/bin/bash
#SBATCH --job-name=mc-gen-figs
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Regenerate the genome GPT-2 figures. No GPU work happens here -- gpu is simply
# the only partition on this cluster, and the login node is off limits for
# anything that runs.
#
# Run from the checkout that holds workflows/slurm-logs/ (the scripts read the
# training logs by relative path and write into tmp/figs/ there). FIG_SCRIPTS
# points at wherever the scripts live, which is normally the worktree that owns
# the figure branch.
#
#   FIG_SCRIPTS=<worktree>/scripts/figs \
#     sbatch workflows/genome-regen-figs.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

FIG_SCRIPTS="${FIG_SCRIPTS:-scripts/figs}"
[ -d "${FIG_SCRIPTS}" ] || { echo "FIG_SCRIPTS=${FIG_SCRIPTS} is not a directory"; exit 1; }
[ -d workflows/slurm-logs ] || { echo "run this from a checkout that has workflows/slurm-logs/"; exit 1; }

mkdir -p tmp/figs
echo "=== node=$(hostname) cwd=$PWD scripts=${FIG_SCRIPTS} ==="

rc=0
for s in plot_genome_b21_curves.py plot_genome_schedule_comparison.py; do
    echo "########## ${s} ##########"
    "${PYROOT}/miniconda/bin/python" "${FIG_SCRIPTS}/${s}" || rc=1
done

ls -la tmp/figs/
echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
