#!/bin/bash
#SBATCH --job-name=mc-zinc22-inventory
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Fix the list of ZINC-22 2D tranche files before any of them is fetched, so the
# acquisition can be checked against it afterwards. Reads directory indexes only;
# nothing is downloaded.
#
#   sbatch --export=ALL,ZINC22_AUTH=<user:password>,ZINC22_OUT=<file.tsv> \
#          workflows/zinc22-inventory.sbatch
#
# This cluster's /etc/resolv.conf points at a dead stub, so every outbound name
# lookup goes through the unprivileged bind-mount helper rather than resolving
# directly. NETRUN points at it.
set -uo pipefail
: "${ZINC22_AUTH:?export ZINC22_AUTH as user:password}"
: "${ZINC22_OUT:?export ZINC22_OUT to the TSV to write}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
NETRUN="${NETRUN:?export NETRUN to the DNS wrapper}"
[ -x "${NETRUN}" ] || { echo "no wrapper at ${NETRUN}" >&2; exit 1; }
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${ZINC22_OUT}")"
echo "=== ZINC-22 tranche inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${NETRUN}" "${PYTHON}" scripts/zinc22_tranche_inventory.py \
    --base-url "${ZINC22_BASE:-https://files.docking.org/zinc22/2d}" --out "${ZINC22_OUT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
