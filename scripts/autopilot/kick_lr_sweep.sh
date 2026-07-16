#!/bin/bash
# Kick the mini LR sweep — charter 2026-07-14 reply §「mini LR sweep」.
# BERT {1e-4, 5e-5, 1e-5} × GPT-2 {6e-4, 3e-4, 6e-5}
# × subsets {mammal_centered, global_random_seed1}
# = 12 short runs (each SMOKE_MAX_STEPS=800, ~ 30-60 min per run × max 4
# concurrent → ~2 h wall clock, ~25 GPU-h total).
#
# global_random_seed6 は charter で「外れ値」 なので選ばない。 mammal_centered =
# mammal 寄り 1、 global_random_seed1 = global 系 1 という代表。

set -euo pipefail

SBATCH_DIR=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/sbatch
SBATCH=${SBATCH_DIR}/subset_lr_sweep.sbatch
STATE_DIR=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/state
MAX_STEPS="${SMOKE_MAX_STEPS:-800}"

declare -A BERT_LRS=(
    ["1e-4"]="0.0001"
    ["5e-5"]="0.00005"
    ["1e-5"]="0.00001"
)
declare -A GPT2_LRS=(
    ["6e-4"]="0.0006"
    ["3e-4"]="0.0003"
    ["6e-5"]="0.00006"
)
SUBSETS=(mammal_centered global_random_seed1)

echo "# subset LR sweep matrix — kicked $(date -Iseconds)" > ${STATE_DIR}/lr_sweep_jobids.txt
echo "# max_steps=${MAX_STEPS}" >> ${STATE_DIR}/lr_sweep_jobids.txt
echo "# jobid | arch | subset | lr_tag | lr_value" >> ${STATE_DIR}/lr_sweep_jobids.txt

for subset in "${SUBSETS[@]}"; do
    for tag in "${!BERT_LRS[@]}"; do
        lr=${BERT_LRS[$tag]}
        name="sweep-bert-${subset}-${tag}"
        jobid=$(sbatch --parsable --job-name="${name}" \
            --export=ALL,WORKFLOW=03c-genome_sequence-train-bert-small-subset.sh,\
GENOME_SUBSET=${subset},SUBSET_BERT_LR=${lr},BERT_LR_TAG=${tag},\
SMOKE_MAX_STEPS=${MAX_STEPS},NUM_GPUS=4 \
            "${SBATCH}")
        echo "  submit: ${name} → jobid ${jobid}"
        echo "${jobid} | bert | ${subset} | ${tag} | ${lr}" >> ${STATE_DIR}/lr_sweep_jobids.txt
    done
    for tag in "${!GPT2_LRS[@]}"; do
        lr=${GPT2_LRS[$tag]}
        name="sweep-gpt2-${subset}-${tag}"
        jobid=$(sbatch --parsable --job-name="${name}" \
            --export=ALL,WORKFLOW=03a-genome_sequence-train-gpt2-small-subset.sh,\
GENOME_SUBSET=${subset},SUBSET_GPT2_LR=${lr},GPT2_LR_TAG=${tag},\
SMOKE_MAX_STEPS=${MAX_STEPS},NUM_GPUS=4 \
            "${SBATCH}")
        echo "  submit: ${name} → jobid ${jobid}"
        echo "${jobid} | gpt2 | ${subset} | ${tag} | ${lr}" >> ${STATE_DIR}/lr_sweep_jobids.txt
    done
done

echo ""
echo "=== 12 jobs submitted. Roster: ${STATE_DIR}/lr_sweep_jobids.txt ==="
cat ${STATE_DIR}/lr_sweep_jobids.txt
