#!/bin/bash
#SBATCH --job-name=mc-gen-1024prep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Rebuild the genome BERT dataset at a 1,024-token window.
#
# The shipped data is 512 ([CLS] + 510 + [SEP]) and that length is baked into
# both the parquet schema (fixed_size_list<int32>[512]) and the arrow features,
# so 1,024 is not a config flag on the training side -- the windows have to be
# cut again from the raw sequences.
#
# Only steps 3 and 4 run, and only for BERT:
#   step 3  raw_files/ -> parquet_bert/                 (bert_chunk_size 1022)
#   step 4  parquet_bert/ -> training_ready_hf_dataset_bert/
# Steps 1 and 2 (download, FASTA -> raw) are skipped by reusing the existing
# raw_files through a symlink. GPT-2 is untouched: models=("bert",), and its
# parquet/arrow are not even present in this tree.
#
# The source tree is read-only for us, which is what guarantees the 512 data
# cannot be damaged by this. Output goes to a separate root.
#
# Required: SUBSET. Optional: GENOME_1024_ROOT, CHUNK, WORKERS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${SUBSET:?set SUBSET, e.g. SUBSET=mammal_centered}"
ROOT="${GENOME_1024_ROOT:-/data1/rkp00024/matsubara/learning_source_genome_1024/genome_sequence}"
CHUNK="${CHUNK:-1022}"        # [CLS] + 1022 + [SEP] = 1024
WORKERS="${WORKERS:-32}"
BASE="${ROOT}/${SUBSET}"

if [ -x "./miniconda/bin/python" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda"; exit 1; }

[ -d "${BASE}/raw_files" ] || { echo "no raw_files under ${BASE} -- link it first"; exit 1; }

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome-1024"
export TOKENIZERS_PARALLELISM=false

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subset=${SUBSET} chunk=${CHUNK} ==="
echo "=== base_dir : ${BASE} ==="
echo "=== raw_files: $(ls "${BASE}/raw_files" | wc -l) files ==="
echo "=== python   : ${PY} ==="

t0=$(date +%s)
"${PY}" - "${BASE}" "${CHUNK}" "${WORKERS}" <<'PYEOF'
import sys
from molcrawl.data.genome_sequence.preparation import (
    process3_subset_raw_to_parquet,
    process4_subset_parquet_to_arrow,
)

base, chunk, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

print(f"########## step 3: raw -> parquet_bert (chunk={chunk}) ##########", flush=True)
ok = process3_subset_raw_to_parquet(
    base_dir=base,
    models=("bert",),
    bert_chunk_size=chunk,
    gpt2_chunk_size=1024,   # unused: models is bert only
    num_worker=workers,
    force=False,
)
if not ok:
    raise SystemExit("step 3 failed")

print("########## step 4: parquet_bert -> arrow ##########", flush=True)
# Split sizes stay at the values the 512 build used, so the two differ only in
# window length. The split is assigned per (accession, contig) with seed 42, so
# contigs stay disjoint across train/valid/test the same way.
ok = process4_subset_parquet_to_arrow(
    base_dir=base,
    models=("bert",),
    valid_size=50_000,
    test_size=50_000,
    valid_frac=0.005,
    test_frac=0.005,
    remove_parquet=False,
    force=False,
)
if not ok:
    raise SystemExit("step 4 failed")
print("done", flush=True)
PYEOF
rc=$?
echo "---- rc=${rc} after $(( $(date +%s) - t0 ))s ----"

if [ ${rc} -eq 0 ]; then
    "${PY}" - "${BASE}" <<'PYEOF'
import json, sys
from datasets import load_from_disk
d = f"{sys.argv[1]}/training_ready_hf_dataset_bert"
ds = load_from_disk(d)
info = json.load(open(f"{d}/train/dataset_info.json"))
print("########## result ##########")
print("  splits:", {k: len(v) for k, v in ds.items()})
print("  input_ids length:", info["features"]["input_ids"].get("length"))
print("  first row length:", len(ds["train"][0]["input_ids"]))
PYEOF
fi
echo "=== done @ $(date -Is) ==="
