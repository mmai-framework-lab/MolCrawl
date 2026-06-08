#!/bin/bash
#SBATCH --job-name=protein-eval-old-vs-new
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/protein_sequence/logs/eval-old-vs-new-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/protein_sequence/logs/eval-old-vs-new-%j.out

# Cross-evaluate the legacy 50k-capped protein_sequence-small (checkpoint-5000
# from learning_source_20260316) and the bugfix 5M-cap retrain (checkpoint-25000
# from learning_source_20260520_uncapped) on the SAME held-out test set drawn
# from learning_source_20260520_uncapped's valid split (140,140 chunks).
#
# Both checkpoints share the same architecture (GPT-2 124M) and tokenizer
# (EsmSequenceTokenizer), so the perplexity / accuracy delta directly attributes
# to the training-data-amount difference (14M tokens vs 1.43B tokens, ~100x).

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

# test_checkpoint.py::load_gpt2_checkpoint expects the legacy nanoGPT-style
# single .pt file (dict with model/model_args/iter_num/best_val_loss keys), not
# the HF Trainer ``checkpoint-NNNN/`` directory layout. Both runs happen to also
# emit ckpt.pt next to the HF dirs (``keep_legacy_ckpt`` was True at the time
# the OLD model was trained, and the NEW small run has likewise produced one
# for the best-val state). Point at those instead.
OLD_CKPT="../learning_source_20260316/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt"
SHARED_TEST_DIR="../learning_source_20260520_uncapped/protein_sequence/training_ready_hf_dataset"
TEST_PARAMS="{\"dataset_dir\": \"${SHARED_TEST_DIR}\"}"

REPORT_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/logs/eval-old-vs-new-${SLURM_JOB_ID}"
mkdir -p "${REPORT_DIR}/old" "${REPORT_DIR}/new"

echo "==== eval-old-vs-new job ${SLURM_JOB_ID} starting at $(date) ===="
echo "Node: $(hostname)"
echo "OLD checkpoint: ${OLD_CKPT}"
echo "NEW checkpoint: ${NEW_CKPT}"
echo "Shared test set: ${SHARED_TEST_DIR}"
echo

echo "########## (1/2) OLD (50k cap, learning_source_20260316) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${OLD_CKPT}" \
    --output_dir "${REPORT_DIR}/old" \
    --domain protein_sequence \
    --test_dataset_params "${TEST_PARAMS}" \
    --max_test_samples 2000 \
    --device cuda 2>&1 | tee "${REPORT_DIR}/old/stdout.log"

echo
echo "########## (2/2) NEW (5M cap, learning_source_20260520_uncapped) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${NEW_CKPT}" \
    --output_dir "${REPORT_DIR}/new" \
    --domain protein_sequence \
    --test_dataset_params "${TEST_PARAMS}" \
    --max_test_samples 2000 \
    --device cuda 2>&1 | tee "${REPORT_DIR}/new/stdout.log"

echo
echo "########## summary (parse the two stdout logs) ##########"
"${PYTHON}" -c "
import re
for tag, log in [('OLD (50k cap, 14M tokens)', '${REPORT_DIR}/old/stdout.log'),
                 ('NEW (5M cap, 1.43B tokens)', '${REPORT_DIR}/new/stdout.log')]:
    with open(log) as f: txt = f.read()
    ppl = re.search(r'Perplexity[^\d]*([\d.]+)', txt)
    avgloss = re.search(r'Average loss[^\d]*([\d.]+)', txt)
    acc = re.search(r'Accuracy[^\d]*([\d.]+)', txt)
    top5 = re.search(r'Top-?5 accuracy[^\d]*([\d.]+)', txt)
    print(f'== {tag} ==')
    print(f'  Perplexity:  {ppl.group(1) if ppl else \"?\"}')
    print(f'  Avg loss:    {avgloss.group(1) if avgloss else \"?\"}')
    print(f'  Accuracy:    {acc.group(1) if acc else \"?\"}')
    print(f'  Top-5 acc:   {top5.group(1) if top5 else \"?\"}')
    print()
"

echo "==== finished at $(date) ===="
