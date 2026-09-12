#!/bin/bash
#SBATCH --job-name=mc-unit-tests
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Run the unit tests on a compute node.
#
# They do not belong on the login node: the suite imports torch and transformers, and
# compute there is not allowed here. No GPU is requested -- nothing under tests/unit
# needs one, and the partition accepts a job without any.
#
# Submit from the repository root:
#   sbatch workflows/run-unit-tests.sbatch
#   sbatch --export=ALL,TESTS="tests/unit/test_run_provenance.py" workflows/run-unit-tests.sbatch
#
# rc is non-zero while tests/unit/test_compounds.py's two benchmark tests error for want
# of pytest-benchmark; read the summary line rather than the exit code alone.
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

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== unit tests on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "tests: ${TESTS:-tests/unit}"
rc=0
"${PYTHON}" -m pytest ${TESTS:-tests/unit} -q -p no:cacheprovider || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
