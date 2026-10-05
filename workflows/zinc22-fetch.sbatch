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
#   sbatch --export=ALL,ZINC22_INVENTORY=<tsv>,ZINC22_DEST=<dir>,\
#   ZINC22_RECORD=<tsv>,ZINC22_RETRY=<txt>[,ZINC22_WORKERS=8][,ZINC22_AUTH=<user:password>] \
#          workflows/zinc22-fetch.sbatch
#
# The 2D tranches sit behind HTTP Basic. ZINC publishes what to use on its own index
# page at files.docking.org/zinc22/, stating the point is to keep the files out of
# search engines rather than to restrict access. Supply it either way:
#
#   ZINC22_AUTH=<user:password>        directly, or
#   ZINC22_AUTH_FILE=<path>           a file whose first line is user:password
#
# The file form keeps the value out of the submit line and the process list. Either
# way it is read at run time and never written to a tracked file or a log. With
# neither set the request goes out unauthenticated, which is what the public
# directories want.
#
# Outbound name lookup goes through the unprivileged bind-mount helper because this
# cluster's /etc/resolv.conf points at a dead stub. NETRUN points at it.
set -uo pipefail
: "${ZINC22_INVENTORY:?export ZINC22_INVENTORY to the tranche TSV}"
: "${ZINC22_DEST:?export ZINC22_DEST to the directory to fetch into}"
: "${ZINC22_RECORD:?export ZINC22_RECORD to the per-file record TSV to write}"
: "${ZINC22_RETRY:?export ZINC22_RETRY to the retry list to write}"
if [ -n "${ZINC22_AUTH_FILE:-}" ]; then
  [ -r "${ZINC22_AUTH_FILE}" ] || { echo "cannot read ${ZINC22_AUTH_FILE}" >&2; exit 1; }
  ZINC22_AUTH="$(head -n 1 "${ZINC22_AUTH_FILE}" | tr -d '\r\n')"
  export ZINC22_AUTH
fi
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
if [ -n "${ZINC22_AUTH:-}" ]; then echo "auth: supplied"; else echo "auth: none"; fi
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
