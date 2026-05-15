#!/usr/bin/env bash
#SBATCH --job-name=eval-chebi20-mol-instr-retry
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_chebi20_mol_instr_retry_%j.log
#SBATCH --error=slurm_logs/eval_chebi20_mol_instr_retry_%j.log
#
# Retry chebi20 × molecule_nat_lang_mol_instructions × gpt2 for the
# three sizes whose HF checkpoint dirs bake a tokenizer.json:
# medium / large / ex-large. The gpt2 adapter was updated to honour
# ``--tokenizer-path`` when it points at an HF tokenizer dir, which
# unblocks the CUDA cublasSgemm error that the first sweep hit
# (caused by feeding mol_instructions ckpts through the master
# MoleculeNatLangTokenizer that has a different vocab mapping).
#
# Small is intentionally excluded — its checkpoint-N/ dirs don't
# carry tokenizer.json (training-side omission), so the fix doesn't
# apply.

set -uo pipefail
cd /lustre/home/matsubara/riken-dataset-fundational-model
# shellcheck disable=SC1091
source workflows/common_functions.sh

LSD="${LEARNING_SOURCE_DIR:-${PWD}/learning_source_20260316}"
export LEARNING_SOURCE_DIR="$LSD"

echo "[chebi20-retry] node: $(hostname)"
nvidia-smi 2>/dev/null | head -8 || true
echo

CHEBI20_DIR="${LSD}/eval/chebi20"
for size in medium large ex-large; do
    ckpt="${LSD}/molecule_nat_lang_mol_instructions/gpt2-output/molecule_nat_lang_mol_instructions-${size}/ckpt.pt"
    hf_tok="${LSD}/molecule_nat_lang_mol_instructions/gpt2-output/molecule_nat_lang_mol_instructions-${size}/checkpoint-5000"
    [[ -f "$ckpt" ]] || { echo "skip ${size}: ckpt.pt missing"; continue; }
    [[ -f "$hf_tok/tokenizer.json" ]] || { echo "skip ${size}: HF tokenizer.json missing"; continue; }

    slug="molecule_nat_lang-gpt2-${size}"
    outdir="${LSD%/}/experiment_data/eval/${slug}/finetune_mol_instructions_chebi20_gapfill"
    if find "$outdir" -name REPORT.md 2>/dev/null | grep -q .; then
        echo "SKIP done: ${size}"
        continue
    fi
    echo "RUN: ${size} -> ${outdir}"
    mkdir -p "$outdir"
    OUTPUT_DIR="$outdir" RUNTAG="finetune_mol_instructions_chebi20_gapfill" DEVICE=cuda \
        MODEL_PATH="$ckpt" TOKENIZER_PATH="$hf_tok" \
        CHEBI20_DIR="$CHEBI20_DIR" ARCH=gpt2 \
        bash workflows/eval-chebi20.sh \
        2>&1 | tail -3 || { echo "FAIL ${size}"; continue; }
done

echo
echo "[chebi20-retry] DONE."
