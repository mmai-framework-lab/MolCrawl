#!/bin/bash
#SBATCH --job-name=bert-prepare-tokenizer
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Write the custom tokenizer once, before a grid goes out.
#
# The rna and genome BERT configs build a tokenizer at import, save it to
# LEARNING_SOURCE_DIR/<domain>/custom_tokenizer_bert, and read it straight back.
# That path does not depend on the run, so runs launched together write over each
# other's copy mid-read. Three arms went out at once on 2026-09-16 and two died in
# AutoTokenizer.from_pretrained thirty seconds in (117320, 117321).
#
# Writing it in advance is not enough on its own, because every run writes it
# again. The three rna BERT base configs therefore save through
# molcrawl.core.tokenizer_io.save_pretrained_atomic, which moves complete files
# into place so a reader never sees a truncated one (48 concurrent executions
# failed 6 times before, 96 failed 0 after; job 121584 against 121585).
#
# What atomic replacement cannot cover is a directory that has never been
# populated: a run can arrive while only some of the files are there. This job
# closes that gap -- run it once on a new tree, and the grid follows with
# --dependency=afterok:<this job>.
#
# protein, compounds and molecule_nat_lang configs do not write a tokenizer at all
# (molecule_nat_lang reads GPT2_TOKENIZER_DIR), so grids for those do not need this.
#
#   jid=$(sbatch --parsable --export=ALL,LEARNING_SOURCE_DIR=<root>,\
# CONFIG=molcrawl/tasks/pretrain/configs/rna/bert_small.py \
#         workflows/bert-prepare-tokenizer.sbatch)
#   sbatch --dependency=afterok:${jid} ... workflows/rna-bert.sbatch
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root}"
: "${CONFIG:?set CONFIG to the BERT config whose tokenizer should be written}"
[ -f "${CONFIG}" ] || { echo "no config at ${CONFIG}" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export LEARNING_SOURCE_DIR

echo "=== prepare tokenizer on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "config : ${CONFIG}"
echo "root   : ${LEARNING_SOURCE_DIR}"

rc=0
"${PYTHON}" - "${CONFIG}" <<'PYTOK' || rc=$?
import runpy
import sys
from pathlib import Path

# Executing the config is what writes the tokenizer; there is no separate entry
# point for it, and reproducing the build here would be a second definition of
# the same thing.
g = runpy.run_path(sys.argv[1], run_name="__main__")

path = g.get("_custom_tokenizer_path")
if path is None:
    print("this config writes no custom tokenizer -- nothing to prepare, and a "
          "grid using it does not need this job")
    raise SystemExit(0)

p = Path(path)
files = sorted(f.name for f in p.iterdir()) if p.is_dir() else []
print(f"written: {p}")
for f in files:
    print(f"  {f}")

# Reading it back is the half that failed for the two arms, so it is the half
# worth checking before a grid depends on it.
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(p))
print(f"reads back: vocabulary {len(tok):,}, mask id {tok.mask_token_id}, "
      f"pad id {tok.pad_token_id}")
PYTOK
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
