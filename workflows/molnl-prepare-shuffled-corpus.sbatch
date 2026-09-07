#!/bin/bash
#SBATCH --job-name=mc-molnl-prep-shuffled
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Rebuild the mol_nl packed corpus with the pre-packing shuffle (PR #137).
#
# The original was written in source order: JS divergence between the head and the
# middle of train measured 0.16449 against a 0.00101 sampling floor, and the rebuild
# brings it to 0.00096. Content is identical -- 325,752,832 tokens in 318,118 blocks
# of 1,024 -- and only the grouping into blocks differs.
#
# Decision 2026-08-21 chose option (b): write to a NEW name and leave the canonical
# path untouched, so the Mol-Instructions fine-tune path that shares this tree keeps
# reading exactly what it read before, and no symlink ever dangles. This script
# checks that before and after, and fails if the canonical tree moved.
#
#   output               <LEARNING_SOURCE_DIR>/molecule_nat_lang/training_ready_hf_dataset_shuffled
#   canonical, untouched <LEARNING_SOURCE_DIR>/molecule_nat_lang/training_ready_hf_dataset
#
# number_sample is absent from the YAML, so prepare_gpt2 uses None = every example.
# The old NO-GO ("never re-run prepare_gpt2, it subsamples to 50k") applied to the
# hard-coded number_sample=50000; that is config-driven now and defaults to None.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch workflows/molnl-prepare-shuffled-corpus.sbatch
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PREP_OUT_SUFFIX="${PREP_OUT_SUFFIX:-_shuffled}"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs

CANONICAL="${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset"
OUT="${CANONICAL}${PREP_OUT_SUFFIX}"

echo "=== mol_nl prep (shuffled) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "data : ${LEARNING_SOURCE_DIR}"
echo "outputs: ${OUT}   (PREP_OUT_SUFFIX=${PREP_OUT_SUFFIX})"
grep -n "PACK_ORDER_SEED = " molcrawl/data/molecule_nat_lang/dataset/prepare_gpt2.py
echo "--- canonical path BEFORE (must stay untouched) ---"
_before=$(du -sb "${CANONICAL}" | cut -f1); ls -ld "${CANONICAL}"; echo "  bytes ${_before}"

rc=0
"${PYTHON}" molcrawl/data/molecule_nat_lang/dataset/prepare_gpt2.py \
    assets/configs/molecule_nat_lang_config.yaml || rc=$?
echo "=== prep rc=${rc} @ $(date -Is) ==="
[ "${rc}" -ne 0 ] && exit "${rc}"

echo "--- new dataset ---"
du -sh "${OUT}"
"${PYTHON}" - "${OUT}" <<'PY'
import sys
from datasets import load_from_disk
d = load_from_disk(sys.argv[1])
print({k: len(v) for k, v in d.items()})
r = d["train"][0]["input_ids"]
print("block_len:", len(r), " EOS(50256) in row0:", sum(1 for x in r if x == 50256))
PY

echo "--- canonical path AFTER (must be unchanged) ---"
_after=$(du -sb "${CANONICAL}" | cut -f1); ls -ld "${CANONICAL}"; echo "  bytes ${_after}"
if [ "${_before}" != "${_after}" ]; then
  echo "FAILED: the canonical tree changed size (${_before} -> ${_after}). Option (b)" \
       "exists so the Mol-Instructions path keeps reading exactly what it read" \
       "before; it no longer does." >&2
  exit 1
fi
echo "=== rc=0 @ $(date -Is) ==="
