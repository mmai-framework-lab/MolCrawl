#!/bin/bash
#SBATCH --job-name=mc-cmp-corpus-structure
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Two things the ladder proposal needs before any GPU is spent (proposal items 2 and 4a):
#   1. CLM n-gram baselines, so the observed loss has a floor to be read against
#   2. which source each region of the corpus came from, so a loss curve can be told
#      apart from the source it was reading at the time
# Neither trains anything. A GPU is requested only because it is the one partition here.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python"; exit 1; }

: "${PACKED_DATASET:?set PACKED_DATASET to the packed GPT-2 dataset root (train/ valid/)}"
: "${CORPUS_PARQUET:?set CORPUS_PARQUET to the combined corpus parquet}"
: "${SOURCE_ROOT:?set SOURCE_ROOT to the directory tree holding the eight source parquets}"
OUT_ROOT="${VERIFY_OUT_ROOT:?set VERIFY_OUT_ROOT to a results root outside any worktree}"
mkdir -p "${OUT_ROOT}"

LLAMOL="${SOURCE_ROOT}/compounds/data/Fraunhofer-SCAI-llamol"
DATA="${SOURCE_ROOT}/compounds/data"

echo "=== compounds corpus structure ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code   : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "packed : ${PACKED_DATASET}"
echo "corpus : ${CORPUS_PARQUET}"
echo "outputs: ${OUT_ROOT}"

rc=0
echo; echo "--- 1/2 CLM n-gram baselines ---"
"${PYTHON}" scripts/measure_clm_ngram_baselines.py \
    --dataset-dir "${PACKED_DATASET}" --vocab "${VOCAB:-612}" \
    --output "${OUT_ROOT}/clm-ngram-baselines.json" || rc=$?

echo; echo "--- 2/2 corpus source ranges ---"
"${PYTHON}" scripts/map_corpus_source_ranges.py \
    --parquet "${CORPUS_PARQUET}" --samples "${RANGE_SAMPLES:-20000}" \
    --source "zinc_qm9=${LLAMOL}/qm9_zinc250_cep.parquet" \
    --source "opv=${SOURCE_ROOT}/compounds/processed/opv.parquet" \
    --source "pubchemqc_2017=${LLAMOL}/pubchemqc_energy.parquet" \
    --source "pubchemqc_2020=${LLAMOL}/pubchemqc2020_energy.parquet" \
    --source "zinc20=${DATA}/zinc20/zinc_processed.parquet" \
    --source "reddb=${LLAMOL}/RedDB_Full.parquet" \
    --source "pc9=${LLAMOL}/Full_PC9_GAP.parquet" \
    --source "chembl=${LLAMOL}/chembl_log_sascore.parquet" \
    --output "${OUT_ROOT}/corpus-source-ranges.json" || rc=$?

echo; echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
