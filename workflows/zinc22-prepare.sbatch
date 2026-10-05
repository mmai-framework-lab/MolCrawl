#!/bin/bash
#SBATCH --job-name=mc-zinc22-prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%A_%a.out
#SBATCH --error=workflows/slurm-logs/%x-%A_%a.out
# Tokenize and pack the fetched ZINC-22 tranche files, one array task per input file.
# Each task writes its own shard and its own counts file, so tasks never contend and
# a re-run skips the files whose shard already exists.
#
#   ls <dest>/*/*.smi.gz > <list>
#   sbatch --array=1-$(wc -l < <list>) \
#          --export=ALL,ZINC22_FILELIST=<list>,ZINC22_SHARDS=<dir>,ZINC22_COUNTS=<dir> \
#          workflows/zinc22-prepare.sbatch
#
# No network: everything this reads is already on disk.
set -uo pipefail
: "${ZINC22_FILELIST:?export ZINC22_FILELIST to the file holding one .smi.gz path per line}"
: "${ZINC22_SHARDS:?export ZINC22_SHARDS to the directory for the parquet shards}"
: "${ZINC22_COUNTS:?export ZINC22_COUNTS to the directory for the per-file counts}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
VOCAB="${ZINC22_VOCAB:-${PYROOT}/assets/molecules/vocab.txt}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${ZINC22_SHARDS}" "${ZINC22_COUNTS}"
IDX="${SLURM_ARRAY_TASK_ID:-1}"
SMI="$(sed -n "${IDX}p" "${ZINC22_FILELIST}")"
[ -n "${SMI}" ] || { echo "no line ${IDX} in ${ZINC22_FILELIST}" >&2; exit 1; }
STEM="$(basename "${SMI}" .smi.gz)"
echo "=== ZINC-22 prepare ${STEM} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "vocab: ${VOCAB} ($(wc -l < "${VOCAB}") entries)"
"${PYTHON}" scripts/zinc22_prepare.py \
    --smi "${SMI}" --vocab "${VOCAB}" \
    --out "${ZINC22_SHARDS}/${STEM}.parquet" \
    --counts "${ZINC22_COUNTS}/${STEM}.json" \
    --block "${ZINC22_BLOCK:-1024}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
