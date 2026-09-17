#!/bin/bash
# Submit the 30 BERT grid arms and the 2 fp32/bf16 check runs of all-bert-order-2026-09-17,
# with every segment queued up front and chained by --dependency=afterany.
#
# Nothing is submitted unless SUBMIT=1. Without it this prints each sbatch command it would
# run, with placeholder ids for the dependencies.
#
#   MAIN=<main checkout> bash workflows/submit-bert-grids.sh             # print only
#   MAIN=<main checkout> SUBMIT=1 bash workflows/submit-bert-grids.sh    # submit
#
# Run it from the checkout whose code the runs should use; SLURM_SUBMIT_DIR is where each
# job reads its code, for every segment, for as long as the grid takes.
#
# Segments (§5.2): the measured bf16 s/step of small, times the size's compute ratio
# (small 1.00, medium 3.39, large 6.35), times max_steps, divided by 96 hours and rounded
# up, plus one. A segment beyond the end of a run is refused by the launcher in seconds
# (scripts/check_bert_grid_segments.sh), so an extra one costs a few seconds of allocation.
#
# afterany, not afterok (§5.1): a segment cut by the time limit ends TIMEOUT, and afterok
# would never start the next one. The launchers continue only from TIMEOUT, so a segment
# that crashed stops the chain instead of resuming into the same failure.
set -euo pipefail
: "${MAIN:?MAIN=<main checkout> (learning_source trees, miniconda)}"
SUBMIT="${SUBMIT:-0}"
RECORD="${RECORD:-${MAIN}/learning_source_bert_grid_submissions/submitted-$(date +%Y%m%dT%H%M%S).tsv}"
C=molcrawl/tasks/pretrain/configs

declare -A SMALL_SPS=([rna]=3.812 [protein_sequence]=4.600 [molecule_nat_lang]=3.868 [compounds]=2.248)
declare -A STEPS=([rna]=120960 [protein_sequence]=33531 [molecule_nat_lang]=12000 [compounds]=15000)
declare -A RATIO=([small]=1.00 [medium]=3.39 [large]=6.35)
declare -A LSD=([rna]="${MAIN}/learning_source_20260723_b200prep" [protein_sequence]="${MAIN}/learning_source_20260730_protein_uniref50" \
                [compounds]="${MAIN}/learning_source_20260805_compounds_packed" [molecule_nat_lang]="${MAIN}/learning_source")
declare -A OUTROOT=([rna]="${MAIN}/learning_source_rna_runs" [protein_sequence]="${MAIN}/learning_source_protein_runs" \
                    [compounds]="${MAIN}/learning_source_compounds_runs")

segments() {  # modality size
  awk -v s="${SMALL_SPS[$1]}" -v r="${RATIO[$2]}" -v n="${STEPS[$1]}" \
      'BEGIN { h = s * r * n / 3600; k = int(h / 96); if (k * 96 < h) k++; print k + 1 }'
}

sb() {  # print or run sbatch; echoes the job id (in print-only mode, a placeholder named for the job)
  if [ "${SUBMIT}" = "1" ]; then sbatch --parsable "$@"
  else
    local a name=""
    for a in "$@"; do case "${a}" in --job-name=*) name="${a#--job-name=}";; esac; done
    echo "DRYRUN sbatch $*" >&2; echo "<${name}>"
  fi
}

[ "${SUBMIT}" = "1" ] && mkdir -p "$(dirname "${RECORD}")" && printf 'modality\tconfig\tsegment\tof\tjob\tdepends_on\n' > "${RECORD}"
record() { [ "${SUBMIT}" = "1" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RECORD}"; return 0; }

grid() {  # modality size lrtags...
  local m="$1" size="$2"; shift 2
  local k seg tag cfg name prev job dep script env
  k=$(segments "${m}" "${size}")
  for tag in "$@"; do
    cfg="${C}/${m}/bert_${size}_lr${tag}.py"; prev=""
    for seg in $(seq 1 "${k}"); do
      name="mc-bert-${m%%_*}-${size}-lr${tag}-s${seg}"
      dep=(); [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
      if [ "${m}" = "molecule_nat_lang" ]; then
        script=workflows/molnl-bert-ladder.sbatch
        env="ALL,LEARNING_SOURCE_DIR=${LSD[$m]},CONFIG=${cfg},SEGMENT=${seg}"
      else
        script=workflows/bert-grid.sbatch
        env="ALL,MODALITY=${m},LEARNING_SOURCE_DIR=${LSD[$m]},MODEL_OUTPUT_ROOT=${OUTROOT[$m]},CONFIG=${cfg},SEGMENT=${seg}"
      fi
      job=$(sb --job-name="${name}" "${dep[@]}" --export="${env}" "${script}")
      record "${m}" "${cfg}" "${seg}" "${k}" "${job}" "${prev:--}"
      echo "${m} ${size} lr${tag} segment ${seg}/${k} -> ${job}${prev:+ (after ${prev})}"
      prev="${job}"
    done
  done
}

LRS="1e4 3e4 1e3"
# §7: longest first.
grid protein_sequence large ${LRS}
grid rna small ${LRS}; grid rna medium ${LRS}; grid rna large ${LRS}
grid protein_sequence medium ${LRS}
grid molecule_nat_lang large ${LRS}
grid protein_sequence small ${LRS}
grid molecule_nat_lang medium ${LRS}; grid molecule_nat_lang small ${LRS}
grid compounds small 5e4 1e3 2e3
for p in fp32 bf16; do
  job=$(sb --job-name="mc-bert-rna-precision-${p}" \
        --export="ALL,PRECISION=${p},LEARNING_SOURCE_DIR=${LSD[rna]},MODEL_OUTPUT_ROOT=${OUTROOT[rna]}" \
        workflows/bert-precision-check.sbatch)
  record rna "${C}/rna/bert_small.py [${p}, stop 8000]" 1 1 "${job}" -
  echo "rna small precision check ${p} -> ${job}"
done
[ "${SUBMIT}" = "1" ] && echo "record: ${RECORD}"
exit 0
