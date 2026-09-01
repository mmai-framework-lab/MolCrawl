#!/bin/bash
#SBATCH --job-name=mc-cmp-corpus-audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Two read-only measurements on the compounds corpus, both asked for in the
# 2026-08-28 verdict:
#
#   1. how alike the molecules that share a packed block are (source vs shuffled
#      vs random pairs) -- the size of the shortcut the reshuffle removed
#   2. how much of MoleculeNet is already in the pretraining corpus -- OrganiX13
#      includes ZINC and ChEMBL, which several downstream sets also draw from
#
#   sbatch --export=ALL,VERIFY_OUT_ROOT=<results root>,CORPUS_PARQUET=<parquet>,\
#     EVAL_ROOT=<moleculenet root> workflows/compounds-corpus-audit.sbatch
#
# Nothing is trained. A GPU is requested only because it is the one partition here.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

# A git worktree holds only tracked files, so the interpreter is not inside it.
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

# Roots come from the environment and fail loudly: learning_source* is gitignored,
# so a default relative to the checkout would silently differ per worktree.
: "${CORPUS_PARQUET:?set CORPUS_PARQUET to the OrganiX13 parquet}"
: "${EVAL_ROOT:?set EVAL_ROOT to the moleculenet root holding <task>/raw.csv}"
OUT_ROOT="${VERIFY_OUT_ROOT:?set VERIFY_OUT_ROOT to a results root outside any worktree}"
[ -f "${CORPUS_PARQUET}" ] || { echo "CORPUS_PARQUET=${CORPUS_PARQUET} is not a file" >&2; exit 1; }
[ -d "${EVAL_ROOT}" ] || { echo "EVAL_ROOT=${EVAL_ROOT} is not a directory" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"

# Sampling sizes. Adjacency only needs enough pairs for the means to settle; the
# overlap scan reads the whole corpus unless CORPUS_LIMIT is set.
LIMIT="${ADJ_LIMIT:-20000}"
PAIRS="${ADJ_PAIRS:-5000}"
CORPUS_LIMIT_ARG=""
[ -n "${CORPUS_LIMIT:-}" ] && CORPUS_LIMIT_ARG="--corpus-limit ${CORPUS_LIMIT}"

echo "=== compounds corpus audit ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "corpus : ${CORPUS_PARQUET}"
echo "eval   : ${EVAL_ROOT}"
echo "outputs: ${OUT_ROOT}"

rc=0
echo; echo "--- 1/2 packed-block adjacency similarity ---"
"${PYTHON}" scripts/measure_pack_adjacency_similarity.py \
    --parquet "${CORPUS_PARQUET}" --limit "${LIMIT}" --pairs "${PAIRS}" \
    --output "${OUT_ROOT}/pack-adjacency-similarity.json" || rc=$?

echo; echo "--- 2/2 evaluation / pretraining overlap ---"
# Column names differ per task and are taken from each raw.csv's header.
"${PYTHON}" scripts/measure_eval_train_overlap.py \
    --parquet "${CORPUS_PARQUET}" --eval-root "${EVAL_ROOT}" ${CORPUS_LIMIT_ARG} \
    --tasks bace:mol bbbp:smiles esol:smiles tox21:smiles \
    --output "${OUT_ROOT}/eval-train-overlap.json" || rc=$?

echo; echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
