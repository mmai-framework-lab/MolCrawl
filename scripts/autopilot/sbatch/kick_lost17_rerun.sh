#!/bin/bash
# LOST 17 subset rerun launcher — best-val ckpt 消失分の再学習。
#
# 用途: fix 実効 (best-val ckpt が save_total_limit 削除を免れているか)
# の実証確認後に 1 回だけ実行する。 実証前に走らせると再度消える可能性が
# ゼロではないので、 必ず verify 済後に kick すること。
#
# 対象 (2026-07-24 audit):
#   BERT 完走 20 中 11 LOST + GPT-2 完走 8 中 6 LOST = 17 subset
#
# 出力 dir 命名 (boss reply-0724-rerun-lost17-go.md §3 承認):
#   BERT: ...-retry-bestckpt/  (via BERT_LR_TAG=retry-bestckpt)
#   GPT-2: ...-lr1e-4v2/       (via GPT2_LR_TAG=lr1e-4v2)
#   → 旧 dir と現行 lr1e-4v1 は保存、 audit 追跡可能

set -euo pipefail

cd /lustre/home/matsubara/riken-dataset-fundational-model

LSD=/lustre/home/matsubara/learning_source_20260710_genome_v2
SBATCH_DIR=tmp/scripts/autopilot/sbatch

# BERT LOST 11 subset
BERT_LOST=(
    eukaryote_matched_random_seed1
    eukaryote_matched_random_seed3
    eukaryote_matched_random_seed6
    eukaryote_matched_random_seed8
    global_random_seed2
    global_random_seed3
    global_random_seed4
    global_random_seed6      # early best@59500、 rerun で再現するか観察 (boss §)
    global_random_seed8
    global_random_seed9
    global_random_seed10
)

# GPT-2 LOST 6 subset
GPT2_LOST=(
    eukaryote_matched_random_seed1
    eukaryote_matched_random_seed2
    eukaryote_matched_random_seed4
    eukaryote_matched_random_seed6
    eukaryote_matched_random_seed7
    eukaryote_matched_random_seed8
)

echo "=== LOST 17 rerun kick ==="
echo "date: $(date -Iseconds)"

echo ""
echo "--- BERT ${#BERT_LOST[@]} subset ---"
for subset in "${BERT_LOST[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="sub-bert-${subset}-retry-bestckpt" \
        --export=ALL,WORKFLOW=03c-genome_sequence-train-bert-small-subset.sh,GENOME_SUBSET=${subset},NUM_GPUS=4,LEARNING_SOURCE_DIR=${LSD},BERT_LR_TAG=retry-bestckpt \
        ${SBATCH_DIR}/subset_train.sbatch)
    echo "  bert-${subset} → JOBID $JID"
done

echo ""
echo "--- GPT-2 ${#GPT2_LOST[@]} subset (LR 1e-4 v2, wd 0.01) ---"
for subset in "${GPT2_LOST[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="sub-gpt2-${subset}-lr1e-4v2" \
        --export=ALL,WORKFLOW=03a-genome_sequence-train-gpt2-small-subset.sh,GENOME_SUBSET=${subset},NUM_GPUS=4,LEARNING_SOURCE_DIR=${LSD},SUBSET_GPT2_LR=0.0001,SUBSET_GPT2_WD=0.01,GPT2_LR_TAG=lr1e-4v2,SUBSET_GPT2_MAX_CKPT=5 \
        ${SBATCH_DIR}/subset_train.sbatch)
    echo "  gpt2-${subset} → JOBID $JID"
done

echo ""
echo "=== 全 17 本投入完了 ==="
sleep 3
squeue -u matsubara -o "%.10i %.55j %.2t %.10M %R" | head -25
