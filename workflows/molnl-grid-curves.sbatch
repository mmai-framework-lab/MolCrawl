#!/bin/bash
#SBATCH --job-name=molnl-grid-curves
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Draw the molecule_nat_lang loss curves and write their TSVs: the BERT learning-rate
# grid, and the GPT-2 ladder's three passes over the corpus.
#
# No GPU: this reads each run's trainer_state.json and draws. It is here rather than on
# the login node because compute there is not allowed, matplotlib included.
#
# Submit from the repository root:
#   sbatch --export=ALL,OUT_DIR=<report dir> workflows/molnl-grid-curves.sbatch
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# A git worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# The same two-step default as workflows/molnl-bert-ladder.sbatch, so this reads the
# directory that launcher wrote. Override either when the runs live elsewhere.
ARTIFACT_ROOT="${MOLNL_ARTIFACT_ROOT:-${PYROOT}/learning_source_20260803_molnl_train}"
RUNS_ROOT="${RUNS_ROOT:-${ARTIFACT_ROOT}/molecule_nat_lang}"
RUNS_DIR="${RUNS_ROOT}/bert-output-shuffled"
OUT_DIR="${OUT_DIR:-${PYROOT}/tmp/molnl-bert-grid-curves}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs "${OUT_DIR}"

echo "=== grid curves on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "runs: ${RUNS_DIR} (bert), ${RUNS_ROOT}/gpt2-output* (gpt2)"
echo "out:  ${OUT_DIR}"
"${PYTHON}" scripts/molnl_bert_grid_curves.py --runs-root "${RUNS_DIR}" --out-dir "${OUT_DIR}"
echo
"${PYTHON}" scripts/molnl_gpt2_ladder_curves.py --runs-root "${RUNS_ROOT}" --out-dir "${OUT_DIR}"
echo "=== done @ $(date -Is) ==="
