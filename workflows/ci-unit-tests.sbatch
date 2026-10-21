#!/bin/bash
#SBATCH --job-name=mc-ci-unit-tests
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Run tests/unit the way ci-tests.yml does, on a compute node.
#
# MOLCRAWL_PYTHON is the research env whose versions ci-tests.yml pins to
# (numpy 1.26.4, pyarrow 23.0.1, scikit-learn 1.6.1). EXTRA_PYTHON, if set, is a
# second interpreter that runs only the tests named in EXTRA_TESTS -- used to
# check that a test does not depend on the pinned scikit-learn.
#
# One GPU because the partition requires one, not because the work uses it.

set -u
: "${MOLCRAWL_PYTHON:?set MOLCRAWL_PYTHON to the research env python (numpy 1.26 / pyarrow 23)}"
[ -x "${MOLCRAWL_PYTHON}" ] || { echo "MOLCRAWL_PYTHON=${MOLCRAWL_PYTHON} is not executable" >&2; exit 1; }

echo "=== checkout: $PWD ($(git rev-parse --short HEAD)) ==="
"${MOLCRAWL_PYTHON}" -c 'import sys, numpy, pyarrow, sklearn; print(sys.version.split()[0], "numpy", numpy.__version__, "pyarrow", pyarrow.__version__, "sklearn", sklearn.__version__)'
PYTHONPATH="$PWD" "${MOLCRAWL_PYTHON}" -m pytest tests/unit -q -p no:cacheprovider
rc=$?
echo "=== tests/unit rc=${rc} ==="

if [ -n "${EXTRA_PYTHON:-}" ]; then
    "${EXTRA_PYTHON}" -c 'import sys, numpy, sklearn; print(sys.version.split()[0], "numpy", numpy.__version__, "sklearn", sklearn.__version__)'
    PYTHONPATH="$PWD" "${EXTRA_PYTHON}" -m pytest ${EXTRA_TESTS:?set EXTRA_TESTS with EXTRA_PYTHON} -q -p no:cacheprovider
    echo "=== extra rc=$? ==="
fi
exit ${rc}
