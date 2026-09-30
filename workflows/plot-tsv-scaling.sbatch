#!/bin/bash
#SBATCH --job-name=mc-tsv-figs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Draw scaling figures from a curve TSV another workstream published.
#
# No GPU is requested -- it reads a TSV and writes PNGs -- but compute does not
# belong on the login node here. The config and the output directory are BOTH
# outside any checkout: they name run paths and this repository is public.
#
#   sbatch --export=ALL,TSV_CONFIG=<file.json>,TSV_OUT_DIR=<dir> \
#          workflows/plot-tsv-scaling.sbatch
set -eo pipefail

: "${TSV_CONFIG:?set TSV_CONFIG to the JSON describing the figures}"
: "${TSV_OUT_DIR:?set TSV_OUT_DIR to the directory the PNGs are written into}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "${REPO}"
"${PYROOT}/miniconda/bin/python" scripts/plot_tsv_scaling.py --config "${TSV_CONFIG}" --out-dir "${TSV_OUT_DIR}"
