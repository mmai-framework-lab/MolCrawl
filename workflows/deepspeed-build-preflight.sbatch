#!/bin/bash
#SBATCH --job-name=ds-build-preflight
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# DeepSpeed order §5.1 / §5.2: install a candidate DeepSpeed into an isolated venv on a
# compute node and check that it imports without touching the MolCrawl environment.
#
# No GPU is requested: nothing here runs a kernel. DS_BUILD_OPS stays 0, so no CUDA op
# is compiled -- the GPT-2 backend uses the trainer's own torch AdamW at ZeRO stage 0
# and needs none. Loading and running on GB200 is a separate, later GPU job.
#
# The venv is created with --system-site-packages: torch, transformers and the rest are
# read from the MolCrawl interpreter as they are, and only DeepSpeed and the four
# packages it adds (einops, hjson, ninja, py-cpuinfo) go into the venv. Installation is
# offline from a wheelhouse downloaded beforehand on the login node (compute-node DNS
# is unreliable), with --no-deps so pip cannot upgrade anything.
#
# Required:
#   DS_VERSION   e.g. 0.19.5
#   WHEELHOUSE   directory holding deepspeed-${DS_VERSION}.tar.gz and the four wheels
# Optional:
#   CUDA_MODULE  environment module providing CUDA_HOME (default cuda/12.9.2, matching torch +cu129)
#   ENV_DIR      default <main checkout>/envs/deepspeed-${DS_VERSION}
#
# Submit from the repository root:
#   sbatch --account=<account> --export=ALL,DS_VERSION=0.19.5,WHEELHOUSE=<dir> workflows/deepspeed-build-preflight.sbatch
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"
: "${DS_VERSION:?set DS_VERSION, e.g. 0.19.5}"
: "${WHEELHOUSE:?set WHEELHOUSE to the directory downloaded on the login node}"
[ -f "${WHEELHOUSE}/deepspeed-${DS_VERSION}.tar.gz" ] || {
  echo "no deepspeed-${DS_VERSION}.tar.gz in ${WHEELHOUSE}" >&2; exit 1; }
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.2}"

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
BASE_PY="${PYROOT}/miniconda/bin/python"
[ -x "${BASE_PY}" ] || { echo "no interpreter at ${BASE_PY}" >&2; exit 1; }
ENV_DIR="${ENV_DIR:-${PYROOT}/envs/deepspeed-${DS_VERSION}}"
[ -e "${ENV_DIR}" ] && { echo "${ENV_DIR} already exists; remove it or set ENV_DIR" >&2; exit 1; }
mkdir -p workflows/slurm-logs

echo "=== deepspeed build preflight ${DS_VERSION} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"
echo "base interpreter: ${BASE_PY}"
echo "venv: ${ENV_DIR}"

snapshot() { "$1" -m pip list --format=freeze 2>/dev/null | sort; }
BEFORE="$(mktemp)"; AFTER="$(mktemp)"
snapshot "${BASE_PY}" > "${BEFORE}"

set +u
# shellcheck disable=SC1091
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CUDA_MODULE}"
set -u
echo "--- toolchain"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
command -v nvcc && nvcc --version | tail -2
gcc --version | head -1
"${BASE_PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda available', torch.cuda.is_available())"

echo "--- venv"
"${BASE_PY}" -m venv --system-site-packages "${ENV_DIR}"
PY="${ENV_DIR}/bin/python"
export DS_BUILD_OPS=0
"${PY}" -m pip install --no-index --find-links "${WHEELHOUSE}" --no-deps --no-build-isolation \
  einops hjson ninja py-cpuinfo "${WHEELHOUSE}/deepspeed-${DS_VERSION}.tar.gz"

echo "--- import"
"${PY}" - <<'PY'
import deepspeed, torch, transformers, accelerate
print("deepspeed", deepspeed.__version__, deepspeed.__file__)
print("torch", torch.__version__, torch.__file__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
from deepspeed.runtime.torch_autocast import autocast_if_enabled  # noqa: F401  used by the backend
print("torch_autocast support: ok")
PY
echo "--- ds_report"
"${ENV_DIR}/bin/ds_report" || echo "ds_report exited $?"
echo "--- pip check (venv)"
"${PY}" -m pip check || echo "pip check exited $?"

echo "--- base environment unchanged?"
snapshot "${BASE_PY}" > "${AFTER}"
if diff "${BEFORE}" "${AFTER}"; then echo "base environment: unchanged"; else echo "base environment: CHANGED" >&2; exit 1; fi
echo "=== done @ $(date -Is) ==="
