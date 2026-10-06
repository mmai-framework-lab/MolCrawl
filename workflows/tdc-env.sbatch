#!/bin/bash
# Build the evaluation dependency for the downstream tasks in its own tree.
#
# PyTDC pulls its own pandas, scikit-learn and requests. Installing it beside the
# environment the training runs use would change the footing of a run already in
# flight, so it goes in a separate venv under envs/, the way envs/rna-eval does.
# The shared environment is not written to.
#
# PyTDC's dependency list reaches tiledb, which publishes no aarch64 wheel and whose
# source build fails here. tiledb backs a few of TDC's modules and none of the dataset
# loaders, so the install takes PyTDC without its dependency closure and then names the
# ones the loaders actually use. A dependency that is skipped on purpose is recorded in
# requirements-frozen.txt beside the ones that went in.
#
# The package index needs a working name lookup, which this cluster's resolv.conf
# does not provide; NETRUN points at the unprivileged bind-mount helper.
#
#   sbatch --export=ALL,TDC_ENV_DIR=<dir>,NETRUN=<wrapper> workflows/tdc-env.sbatch
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=mc-tdc-env
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -uo pipefail
: "${TDC_ENV_DIR:?export TDC_ENV_DIR to the directory the venv should live in}"
NETRUN="${NETRUN:?export NETRUN to the DNS wrapper}"
[ -x "${NETRUN}" ] || { echo "no wrapper at ${NETRUN}" >&2; exit 1; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
BASE_PY="${PYROOT}/miniconda/bin/python"
[ -x "${BASE_PY}" ] || { echo "no python under ${PYROOT}/miniconda" >&2; exit 1; }
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${TDC_ENV_DIR}")"
echo "=== tdc-env on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "base python (left untouched): $("${BASE_PY}" -V) at ${PYROOT}/miniconda"
echo "building into              : ${TDC_ENV_DIR}"

"${BASE_PY}" -m venv "${TDC_ENV_DIR}"
ENV_PY="${TDC_ENV_DIR}/bin/python"
"${NETRUN}" "${ENV_PY}" -m pip install --quiet --upgrade pip || exit 1
"${NETRUN}" "${ENV_PY}" -m pip install --quiet setuptools wheel || exit 1
# The loaders need these; tiledb and the modules behind it are left out.
"${NETRUN}" "${ENV_PY}" -m pip install \
    numpy pandas scikit-learn requests tqdm fuzzywuzzy seaborn huggingface_hub || exit 1
"${NETRUN}" "${ENV_PY}" -m pip install --no-deps PyTDC || exit 1

echo "--- 入ったもの ---"
"${ENV_PY}" -m pip list 2>/dev/null | grep -iE "^(PyTDC|pandas|numpy|scikit-learn|requests|fuzzywuzzy|tqdm|seaborn) " || true
echo "--- 取得経路が動くか（import だけ、ダウンロードはしない）---"
"${ENV_PY}" - <<'PYCHK'
import importlib, sys
ok = True
for mod in ("tdc", "tdc.single_pred", "tdc.utils"):
    try:
        importlib.import_module(mod)
        print(f"  {mod}: import できる")
    except Exception as exc:
        ok = False
        print(f"  {mod}: {type(exc).__name__}: {exc}")
# tdc exposes no __version__, so the installed distribution's metadata is the
# only place the version can be read from, and §4 asks for it to be recorded.
from importlib.metadata import version
print(f"  PyTDC 版: {version('PyTDC')}")
sys.exit(0 if ok else 1)
PYCHK
echo "--- 版の記録 ---"
"${NETRUN}" "${ENV_PY}" -m pip freeze > "${TDC_ENV_DIR}/requirements-frozen.txt"
wc -l < "${TDC_ENV_DIR}/requirements-frozen.txt" | xargs echo "  requirements-frozen.txt 行数:"
echo "--- 共有環境が変わっていないことの確認 ---"
"${BASE_PY}" -c "import importlib.util as u; print('  共有側の tdc:', 'あり（想定外）' if u.find_spec('tdc') else 'なし（想定どおり）')"
echo "=== rc=$? @ $(date -Is) ==="
