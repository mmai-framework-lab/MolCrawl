#!/bin/bash
# Submit 9 subset × N model SLURM jobs for the Evo2 subset pretrain campaign.
#
# Per Step E plan: mammal_centered (1), eukaryote_matched_random_seed{1..5} (5),
# global_random_seed{1..3} (3). H200x8 → up to 8 concurrent; the rest queue (PD).
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   bash workflows/eval-subset-pretrain-9runs.sh
#       [--dry-run]   # show sbatch commands without submitting
#
# Env knobs:
#   MODELS       — space-separated list, default "bert gpt2".
#                   e.g. MODELS=bert bash ... → 9 BERT jobs only.
#   SBATCH_TIME  — --time value, default 2-00:00:00 (48 h).
set -e

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=${MODELS:-bert gpt2}

SUBSETS=(
    mammal_centered
    eukaryote_matched_random_seed1
    eukaryote_matched_random_seed2
    eukaryote_matched_random_seed3
    eukaryote_matched_random_seed4
    eukaryote_matched_random_seed5
    global_random_seed1
    global_random_seed2
    global_random_seed3
)

for subset in "${SUBSETS[@]}"; do
    for model in $MODELS; do
        case "$model" in
            bert) wrapper="${SCRIPT_DIR}/03c-genome_sequence-train-bert-small-subset.sh" ;;
            gpt2) wrapper="${SCRIPT_DIR}/03a-genome_sequence-train-gpt2-small-subset.sh" ;;
        esac
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
