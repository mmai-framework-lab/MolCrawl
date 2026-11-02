#!/usr/bin/env bash
# Rescore ProteinGym on the seed-42 confirmed protein weights (protein-order-2026-11-02
# §3, decision B). One model per eval slug, each a 200-example / stratified / seed-42 /
# bootstrap-100 / context-1024 sweep over the full 217-assay set (the current settings;
# nothing here changes them -- the sweep sbatch carries the defaults).
#
# What changes versus the previous scoring: the previous ProteinGym numbers for these
# models came from seed-1001 weights (the GPT-2 four) or from a different seed-42 run
# (BERT large was scored on the `production` checkpoint, not the confirmed `-rerun`).
# Each model here is scored on its confirmed series instead:
#
#   GPT-2 small/medium/large/xl -> protein_sequence-{small,medium,large,ex-large}-rerun/ckpt.pt
#   BERT large                  -> protein_sequence-large-lr3p5e5-rerun/checkpoint-32000
#                                  (the rerun's best eval_loss_mask checkpoint)
#
# Each writes a NEW eval slug (…-s42) so the existing scored outputs are left in place;
# the tables step (§5) switches MODELS over to these slugs deliberately, it is not a
# silent overwrite.
#
# Resumable: the sweep skips any assay that already has a REPORT.md, so a segment cut by
# the 24h wall simply resumes on the next one. BERT large is ~221 GPU-h, so it is a chain
# of afterany segments (afterany, not afterok, because a wall TIMEOUT is the ordinary end
# of a segment here -- see check_job_outcomes.py, which treats TIMEOUT as benign).
#
# Run from the main checkout (the data and the conda env live there):
#   SUBMIT=1 workflows/submit-protein-proteingym-rescore.sh
# Without SUBMIT=1 it prints what it would submit and exits (dry run).
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO}"

LSD="${LEARNING_SOURCE_DIR:-${REPO}/learning_source_20260730_protein_uniref50}"
RUNS="${REPO}/learning_source_protein_runs/protein_sequence"
SWEEP="workflows/eval-gpu-proteingym-sweep.sbatch"
BERT_SEGMENTS="${BERT_SEGMENTS:-12}"
RECORD="${SUBMIT_RECORD:-workflows/submit-records/protein-proteingym-rescore-$(date +%Y%m%d).tsv}"

# The data and the conda env only exist in the main checkout; fail clearly otherwise.
for d in "${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions" "${RUNS}"; do
  [ -d "${d}" ] || { echo "ERROR: ${d} not found -- run this from the main checkout" >&2; exit 1; }
done
[ -f "${SWEEP}" ] || { echo "ERROR: ${SWEEP} not found" >&2; exit 1; }

DRY=1; [ "${SUBMIT:-0}" = "1" ] && DRY=0
if [ "${DRY}" = "0" ]; then mkdir -p "$(dirname "${RECORD}")"; fi
record() { [ "${DRY}" = "0" ] && printf '%s\t%s\t%s\n' "$1" "$2" "$(date -Is)" >> "${RECORD}"; }

# arch  size  model_path  slug
JOBS=(
  "gpt2 small    ${RUNS}/gpt2-output/protein_sequence-small-rerun/ckpt.pt       protein_sequence-gpt2-small-s42"
  "gpt2 medium   ${RUNS}/gpt2-output/protein_sequence-medium-rerun/ckpt.pt      protein_sequence-gpt2-medium-s42"
  "gpt2 large    ${RUNS}/gpt2-output/protein_sequence-large-rerun/ckpt.pt       protein_sequence-gpt2-large-s42"
  "gpt2 xl       ${RUNS}/gpt2-output/protein_sequence-ex-large-rerun/ckpt.pt    protein_sequence-gpt2-xl-s42"
)
BERT_PATH="${RUNS}/bert-output/protein_sequence-large-lr3p5e5-rerun/checkpoint-32000"
BERT_SLUG="protein_sequence-bert-large-s42"

common_env() {  # arch model_path slug
  printf 'ALL,LEARNING_SOURCE_DIR=%s,MODEL_PATH=%s,MODEL_SLUG=%s,ARCH=%s,MODALITY=protein_sequence' \
    "${LSD}" "$2" "$3" "$1"
}

echo "=== protein ProteinGym rescore (seed 42, decision B) ==="
echo "record: ${RECORD}"
echo

# --- GPT-2: four independent jobs, each completes well within one 24h wall ---
for spec in "${JOBS[@]}"; do
  read -r arch size mpath slug <<<"${spec}"
  [ -e "${mpath}" ] || { echo "ERROR: missing checkpoint ${mpath}" >&2; exit 1; }
  name="pg-rescore-${arch}-${size}"
  env="$(common_env "${arch}" "${mpath}" "${slug}")"
  echo "${arch} ${size} -> slug ${slug}"
  if [ "${DRY}" = "0" ]; then
    jid=$(SUBMIT_RECORD="${RECORD}" workflows/sbatch-watched.sh "${name}" -- \
          --job-name="${name}" --export="${env}" "${SWEEP}")
    echo "  submitted ${jid}"
  else
    echo "  DRYRUN sbatch --job-name=${name} --export=${env} ${SWEEP}"
  fi
done

# --- BERT large: afterany chain of resumable segments (~221 GPU-h) ---
[ -d "${BERT_PATH}" ] || { echo "ERROR: missing checkpoint ${BERT_PATH}" >&2; exit 1; }
env="$(common_env bert "${BERT_PATH}" "${BERT_SLUG}")"
echo
echo "bert large -> slug ${BERT_SLUG} (${BERT_SEGMENTS} segments, afterany chain)"
prev=""
for seg in $(seq 1 "${BERT_SEGMENTS}"); do
  name="pg-rescore-bert-large-seg${seg}"
  dep=(); [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
  if [ "${DRY}" = "0" ]; then
    jid=$(sbatch --parsable --job-name="${name}" "${dep[@]}" --export="${env}" "${SWEEP}")
    record "${jid}" "${name}"
    echo "  segment ${seg}/${BERT_SEGMENTS} -> ${jid}${prev:+ (after ${prev})}"
    prev="${jid}"
  else
    echo "  DRYRUN segment ${seg}/${BERT_SEGMENTS} sbatch --job-name=${name} ${dep[*]} --export=${env} ${SWEEP}"
  fi
done

echo
if [ "${DRY}" = "0" ]; then
  echo "DONE. Check outcomes later with:"
  echo "  python scripts/check_job_outcomes.py ${RECORD}"
else
  echo "DRY RUN -- set SUBMIT=1 to submit."
fi
