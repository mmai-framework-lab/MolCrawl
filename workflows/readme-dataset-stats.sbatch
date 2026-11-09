#!/bin/bash
#SBATCH --job-name=mc-readme-stats
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Measure the figures the README "Dataset Statistics" section quotes, from the
# production builds the pretrain configs read, so the section can be rewritten from
# data rather than from the 2024 values it still carries.
#
# Three measurements, each by an existing script:
#   items    scripts/audit_corpus_item_counts.py  (parquet footers / arrow schema)
#   windows  scripts/audit_data_parameters.py     (rows per split x row length)
#   molecules in the packed compounds build  scripts/count_packed_molecules.py
# plus the on-disk size of each training-ready directory (du -sb).
#
# Nothing is written into the input trees. Roots come from the environment because
# learning_source* is gitignored and differs per checkout:
#
#   sbatch --export=ALL,REPORT_DIR=<dir>,GENOME_ROOT=<.../genome_sequence>,\
#   PROTEIN_ROOT=<.../protein_sequence>,RNA_ROOT=<.../rna>,\
#   COMPOUNDS_ROOT=<.../compounds/organix13>,MOLNL_ROOT=<.../molecule_nat_lang> \
#          workflows/readme-dataset-stats.sbatch
set -uo pipefail
: "${REPORT_DIR:?export REPORT_DIR to the directory the reports go to}"
: "${GENOME_ROOT:?export GENOME_ROOT to the genome_sequence root holding the 21 subsets}"
: "${PROTEIN_ROOT:?export PROTEIN_ROOT to the protein_sequence root}"
: "${RNA_ROOT:?export RNA_ROOT to the rna root}"
: "${COMPOUNDS_ROOT:?export COMPOUNDS_ROOT to compounds/organix13}"
: "${MOLNL_ROOT:?export MOLNL_ROOT to the molecule_nat_lang root}"
GENOME_SUBSETS="${GENOME_SUBSETS:-mammal_centered $(printf 'eukaryote_matched_random_seed%s ' 1 2 3 4 5 6 7 8 9 10)$(printf 'global_random_seed%s ' 1 2 3 4 5 6 7 8 9 10)}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
OUT="${REPORT_DIR%/}/readme-dataset-stats-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

echo "=== README dataset stats ${SLURM_JOB_ID:-} on ${SLURMD_NODENAME:-} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "out   : ${OUT}"
rc_all=0

# --- items -----------------------------------------------------------------
ITEM_ARGS=(
  --parquet-dir "protein_sequences=${PROTEIN_ROOT}/parquet_files"
  --parquet-dir "rna_cells=${RNA_ROOT}/parquet_files"
  --parquet-dir "compounds_organix13_molecules=${COMPOUNDS_ROOT}"
  --arrow "molnl_train=${MOLNL_ROOT}/arrow_splits/train.arrow"
  --arrow "molnl_valid=${MOLNL_ROOT}/arrow_splits/valid.arrow"
  --arrow "molnl_test=${MOLNL_ROOT}/arrow_splits/test.arrow"
)
for s in ${GENOME_SUBSETS}; do
  ITEM_ARGS+=(--parquet-dir "genome_${s}=${GENOME_ROOT}/${s}/parquet_gpt2")
done
echo "--- items"
"${PYTHON}" scripts/audit_corpus_item_counts.py "${ITEM_ARGS[@]}" \
  --report "${OUT}/items.json" || rc_all=1

# --- windows ---------------------------------------------------------------
WIN_ARGS=(
  --dataset "protein=${PROTEIN_ROOT}/training_ready_hf_dataset"
  --dataset "rna=${RNA_ROOT}/training_ready_hf_dataset"
  --dataset "compounds_gpt2=${COMPOUNDS_ROOT}/training_ready_hf_dataset_gpt2"
  --dataset "compounds_bert=${COMPOUNDS_ROOT}/training_ready_hf_dataset_bert"
  --dataset "molnl=${MOLNL_ROOT}/training_ready_hf_dataset_shuffled"
)
for s in ${GENOME_SUBSETS}; do
  WIN_ARGS+=(--dataset "genome_gpt2_${s}=${GENOME_ROOT}/${s}/training_ready_hf_dataset_gpt2")
  WIN_ARGS+=(--dataset "genome_bert_${s}=${GENOME_ROOT}/${s}/training_ready_hf_dataset_bert")
done
echo "--- windows"
"${PYTHON}" scripts/audit_data_parameters.py "${WIN_ARGS[@]}" \
  --report "${OUT}/windows.json" || rc_all=1

# --- molecules in the packed compounds build ([SEP] = 13) ------------------
echo "--- packed compounds molecules"
"${PYTHON}" scripts/count_packed_molecules.py \
  --dataset "${COMPOUNDS_ROOT}/training_ready_hf_dataset_gpt2" --sep-id 13 \
  --report "${OUT}/compounds-packed-molecules.json" || rc_all=1

# --- on-disk size ----------------------------------------------------------
echo "--- sizes"
{
  printf 'label\tbytes\tpath\n'
  for pair in "protein=${PROTEIN_ROOT}/training_ready_hf_dataset" \
              "rna=${RNA_ROOT}/training_ready_hf_dataset" \
              "compounds_gpt2=${COMPOUNDS_ROOT}/training_ready_hf_dataset_gpt2" \
              "molnl=${MOLNL_ROOT}/training_ready_hf_dataset_shuffled"; do
    p="${pair#*=}"; printf '%s\t%s\t%s\n' "${pair%%=*}" "$(du -sb "${p}" | cut -f1)" "${p}"
  done
  for s in ${GENOME_SUBSETS}; do
    p="${GENOME_ROOT}/${s}/training_ready_hf_dataset_gpt2"
    printf '%s\t%s\t%s\n' "genome_gpt2_${s}" "$(du -sb "${p}" | cut -f1)" "${p}"
  done
} > "${OUT}/sizes.tsv" || rc_all=1
cat "${OUT}/sizes.tsv"

echo "=== rc=${rc_all} @ $(date -Is) ==="
exit "${rc_all}"
