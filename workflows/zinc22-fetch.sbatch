#!/bin/bash
#SBATCH --job-name=mc-zinc22-fetch
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Fetch the ZINC-22 2D tranche files named in a fixed inventory. Resumable: a file
# already present at its listed size is skipped, so re-submitting after a timeout or
# a failure continues rather than starting over.
#
#   sbatch --export=ALL,ZINC22_AUTH=<user:password>,ZINC22_INVENTORY=<tsv>,\
#   ZINC22_DEST=<dir>,ZINC22_RECORD=<tsv>,ZINC22_RETRY=<txt>[,ZINC22_WORKERS=8] \
#          workflows/zinc22-fetch.sbatch
#
# Outbound name lookup goes through the unprivileged bind-mount helper because this
# cluster's /etc/resolv.conf points at a dead stub. NETRUN points at it.
set -uo pipefail
: "${ZINC22_AUTH:?export ZINC22_AUTH as user:password}"
: "${ZINC22_INVENTORY:?export ZINC22_INVENTORY to the tranche TSV}"
: "${ZINC22_DEST:?export ZINC22_DEST to the directory to fetch into}"
: "${ZINC22_RECORD:?export ZINC22_RECORD to the per-file record TSV to write}"
: "${ZINC22_RETRY:?export ZINC22_RETRY to the retry list to write}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
NETRUN="${NETRUN:?export NETRUN to the DNS wrapper}"
[ -x "${NETRUN}" ] || { echo "no wrapper at ${NETRUN}" >&2; exit 1; }
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${ZINC22_DEST}" "$(dirname "${ZINC22_RECORD}")"
echo "=== ZINC-22 fetch on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "dest: ${ZINC22_DEST}"
df -h "${ZINC22_DEST}" | tail -1
"${NETRUN}" "${PYTHON}" scripts/zinc22_fetch.py \
    --inventory "${ZINC22_INVENTORY}" \
    --base-url "${ZINC22_BASE:-https://files.docking.org/zinc22/2d}" \
    --dest "${ZINC22_DEST}" \
    --record "${ZINC22_RECORD}" \
    --retry-list "${ZINC22_RETRY}" \
    --workers "${ZINC22_WORKERS:-8}" \
    ${ZINC22_LIMIT:+--limit "${ZINC22_LIMIT}"}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
