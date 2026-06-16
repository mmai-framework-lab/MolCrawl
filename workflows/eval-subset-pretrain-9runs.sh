#!/bin/bash
# Submit 21 subset × N model SLURM jobs for the Evo2 subset pretrain campaign.
#
# History:
#   - Initial scope (post_phase6_instructions, 2026-06-08): 9 subset (first batch,
#     mammal_centered + eukaryote 1-5 + global 1-3). 18 ckpts completed by 6/13.
#   - Extended scope (2026-06-15): expanded to all 21 Phase-6-prepared subsets
#     (eukaryote 1-10 + global 1-10 + mammal_centered), 24 jobs added.
#
# H200x8 → up to 8 concurrent; the rest queue (PD).
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   bash workflows/eval-subset-pretrain-9runs.sh
#       [--dry-run]   # show sbatch commands without submitting
#
# Env knobs:
#   MODELS       — space-separated list, default "bert gpt2".
#                   e.g. MODELS=bert bash ... → BERT jobs only.
#   SBATCH_TIME  — --time value, default 2-00:00:00 (48 h).
#   SKIP_EXISTING — if "1", skip subsets that already have a checkpoint dir
#                   (avoids re-submitting completed jobs).
#
# Note: filename retains "9runs" for backward-compat; functionally covers all 21
# subsets now. A rename + new symlink may follow.
set -e

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=${MODELS:-bert gpt2}
SKIP_EXISTING=${SKIP_EXISTING:-0}

SUBSETS=(
    mammal_centered
    eukaryote_matched_random_seed1
    eukaryote_matched_random_seed2
    eukaryote_matched_random_seed3
    eukaryote_matched_random_seed4
    eukaryote_matched_random_seed5
    eukaryote_matched_random_seed6
    eukaryote_matched_random_seed7
    eukaryote_matched_random_seed8
    eukaryote_matched_random_seed9
    eukaryote_matched_random_seed10
    global_random_seed1
    global_random_seed2
    global_random_seed3
    global_random_seed4
    global_random_seed5
    global_random_seed6
    global_random_seed7
    global_random_seed8
    global_random_seed9
    global_random_seed10
)

for subset in "${SUBSETS[@]}"; do
    for model in $MODELS; do
        case "$model" in
            bert)
                wrapper="${SCRIPT_DIR}/03c-genome_sequence-train-bert-small-subset.sh"
                out_subdir="bert-output"
                ;;
            gpt2)
                wrapper="${SCRIPT_DIR}/03a-genome_sequence-train-gpt2-small-subset.sh"
                out_subdir="gpt2-output"
                ;;
        esac

        # Optional: skip subsets that already have a checkpoint dir.
        if [ "$SKIP_EXISTING" = "1" ]; then
            existing_dir="${LEARNING_SOURCE_DIR}/genome_sequence/${out_subdir}/genome_sequence-small-${subset}"
            if [ -d "$existing_dir" ]; then
                echo "SKIP (existing dir): ${subset}/${model}"
                continue
            fi
        fi

        # --time=2-00:00:00 (48 h) overrides the h200-long DefaultTime of
        # 24 h; the 60 000-step pretrain at ~1.44 s/it lands at ~24 h and
        # was clipped by the default for job 19018. Tune via SBATCH_TIME.
        cmd="sbatch --partition=h200-long --gres=gpu:1 --time=${SBATCH_TIME:-2-00:00:00} \
              --job-name=${subset}-${model}-small \
              --export=ALL,GENOME_SUBSET=${subset} ${wrapper}"
        if [ "$DRY_RUN" = 1 ]; then
            echo "DRY: $cmd"
        else
            echo "SUBMIT: $cmd"
            eval "$cmd"
        fi
    done
done
