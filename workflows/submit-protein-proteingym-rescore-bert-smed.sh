#!/usr/bin/env bash
# Rescore ProteinGym for BERT small and medium on their ADOPTED checkpoint
# (protein-judgment-2026-11-10 §3.2). The earlier …-best slugs scored the final
# checkpoint-33531, which carries no eval record and is not the adopted weight; the
# adopted checkpoint is the argmin of eval_loss_mask, which trainer_state.json records
# as best_model_checkpoint:
#
#   BERT small  (lr 3e-4) -> protein_sequence-small-lr3e4/checkpoint-32000   (2.5072)
#   BERT medium (lr 1e-4) -> protein_sequence-medium-lr1e4/checkpoint-32900  (2.4938)
#
# Same fix already applied to BERT large and the GPT-2 four: one model's reached loss and
# its downstream score now come from the same run and the same checkpoint. Settings are
# unchanged (200 examples / stratified / seed 42 / bootstrap 100 / context 1024 / all 217
# assays); the sweep sbatch carries those defaults and nothing here overrides them.
#
# Each writes a NEW eval slug (…-s42, matching bert-large-s42) so the …-best outputs are
# left in place; the tables step (§5) switches MODELS over to these slugs deliberately.
#
# Resumable: a segment cut by the 24h wall resumes on the next assay. A segment is a chain
# of afterany dependencies (afterany, not afterok, because a wall TIMEOUT is the ordinary
# end of a segment -- see check_job_outcomes.py, which treats TIMEOUT as benign).
# Completeness is confirmed after the chains drain with
# scripts/check_proteingym_sweep_complete.py (content-based, not REPORT.md existence).
#
# Run from the main checkout (the data and the conda env live there):
#   SUBMIT=1 workflows/submit-protein-proteingym-rescore-bert-smed.sh
# Without SUBMIT=1 it prints what it would submit and exits (dry run).
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO}"

LSD="${LEARNING_SOURCE_DIR:-${REPO}/learning_source_20260730_protein_uniref50}"
RUNS="${REPO}/learning_source_protein_runs/protein_sequence"
SWEEP="workflows/eval-gpu-proteingym-sweep.sbatch"
RECORD="${SUBMIT_RECORD:-workflows/submit-records/protein-proteingym-rescore-bert-smed-$(date +%Y%m%d).tsv}"

# The data and the conda env only exist in the main checkout; fail clearly otherwise.
for d in "${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions" "${RUNS}"; do
  [ -d "${d}" ] || { echo "ERROR: ${d} not found -- run this from the main checkout" >&2; exit 1; }
done
[ -f "${SWEEP}" ] || { echo "ERROR: ${SWEEP} not found" >&2; exit 1; }

DRY=1; [ "${SUBMIT:-0}" = "1" ] && DRY=0
if [ "${DRY}" = "0" ]; then mkdir -p "$(dirname "${RECORD}")"; fi

common_env() {  # model_path slug
  printf 'ALL,LEARNING_SOURCE_DIR=%s,MODEL_PATH=%s,MODEL_SLUG=%s,ARCH=bert,MODALITY=protein_sequence' \
    "${LSD}" "$1" "$2"
}

# size  checkpoint_path  slug  segments
#   small is ~1 wall of compute; medium is heavier -- generous segment counts just no-op
#   once every assay is done (a segment that finds the sweep complete exits quickly).
JOBS=(
  "small   ${RUNS}/bert-output/protein_sequence-small-lr3e4/checkpoint-32000    protein_sequence-bert-small-s42    4"
  "medium  ${RUNS}/bert-output/protein_sequence-medium-lr1e4/checkpoint-32900   protein_sequence-bert-medium-s42   10"
)

echo "=== protein ProteinGym rescore: BERT small/medium on adopted checkpoint (§3.2) ==="
echo "record: ${RECORD}"
echo

for spec in "${JOBS[@]}"; do
  read -r size ckpt slug segs <<<"${spec}"
  [ -d "${ckpt}" ] || { echo "ERROR: missing checkpoint ${ckpt}" >&2; exit 1; }
  env="$(common_env "${ckpt}" "${slug}")"
  echo "bert ${size} -> slug ${slug} (${segs} segments, afterany chain)"
  echo "  checkpoint: ${ckpt}"
  prev=""
  for seg in $(seq 1 "${segs}"); do
    name="pg-rescore-bert-${size}-seg${seg}"
    if [ "${DRY}" != "0" ]; then
      echo "  DRYRUN segment ${seg}/${segs} sbatch --job-name=${name}${prev:+ --dependency=afterany:${prev}} --export=<env> ${SWEEP}"
      prev="<jid${seg}>"
      continue
    fi
    if [ -z "${prev}" ]; then
      # First segment: through sbatch-watched.sh so a failure recorder is chained on and
      # the id is recorded even if this session goes away.
      jid=$(SUBMIT_RECORD="${RECORD}" workflows/sbatch-watched.sh "${name}" -- \
            --job-name="${name}" --export="${env}" "${SWEEP}")
    else
      jid=$(sbatch --parsable --job-name="${name}" --dependency="afterany:${prev}" \
            --export="${env}" "${SWEEP}")
      printf '%s\t%s\t%s\n' "${jid}" "${name}" "$(date -Is)" >> "${RECORD}"
    fi
    echo "  segment ${seg}/${segs} -> ${jid}${prev:+ (after ${prev})}"
    prev="${jid}"
  done
  echo
done

if [ "${DRY}" = "0" ]; then
  echo "DONE. Check outcomes later with:"
  echo "  python scripts/check_job_outcomes.py --record ${RECORD}"
  echo "Then confirm completeness:"
  echo "  python scripts/check_proteingym_sweep_complete.py protein_sequence-bert-small-s42 protein_sequence-bert-medium-s42"
else
  echo "DRY RUN -- set SUBMIT=1 to submit."
fi
