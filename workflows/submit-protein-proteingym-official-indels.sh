#!/usr/bin/env bash
# Full-assay ProteinGym *indel* zero-shot scoring for the OFFICIAL aggregation
# (Issue #388, M3). Scores every variant of all 66 DMS indel assays against the
# official release data with the causal GPT-2 whole-sequence scoring (pll_delta
# = LL(mutant) - LL(wildtype)).
#
# GPT-2 ONLY. Indels change sequence length, so the masked-marginals protocol
# (which diffs equal-length reference/variant token streams) does not apply; the
# only MLM option is whole-sequence pseudo-log-likelihood, whose cost
# (~ O(n_variants x seq_len)) is infeasible on the large indel assays
# (CAPSD_AAV2S designed_indels alone is ~226k variants). The official ProteinGym
# indel benchmark is likewise carried by autoregressive / alignment models, not
# masked LMs. BERT indels are therefore out of scope here (team decision,
# 2026-11-20).
#
# Adopted checkpoints = the seed-42 table weights (nanoGPT best-val ckpt.pt):
#   GPT-2 {small,medium,large,xl} -> gpt2-output/protein_sequence-{small,medium,large,ex-large}-rerun/ckpt.pt
#
# New eval slugs (…-official-indels-s42) keep the substitution outputs untouched.
# The wildtype for each assay is the reference target_seq, looked up by the sweep
# from PG_REFERENCE and passed to the evaluator (indel CSVs carry no wildtype).
#
# Run (dry by default; SUBMIT=1 submits via sbatch-watched):
#   SUBMIT=1 workflows/submit-protein-proteingym-official-indels.sh
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO}"

MAIN="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
[ -n "${MAIN}" ] || MAIN="${REPO}"
LSD="${LEARNING_SOURCE_DIR:-${MAIN}/learning_source_20260730_protein_uniref50}"
RUNS="${PROTEIN_RUNS:-${MAIN}/learning_source_protein_runs/protein_sequence}"
MC_PYTHON="${MOLCRAWL_PYTHON:-${MAIN}/miniconda/bin/python}"
SWEEP="workflows/eval-gpu-proteingym-sweep.sbatch"
PG_DIR="${LSD}/eval/proteingym/official/data/DMS_ProteinGym_indels"
PG_REFERENCE="${LSD}/eval/proteingym/official/refs/DMS_indels.csv"
RECORD="${SUBMIT_RECORD:-workflows/submit-records/protein-proteingym-official-indels-$(date +%Y%m%d).tsv}"

for d in "${PG_DIR}" "${RUNS}"; do
  [ -d "${d}" ] || { echo "ERROR: ${d} not found (MAIN=${MAIN})" >&2; exit 1; }
done
[ -f "${PG_REFERENCE}" ] || { echo "ERROR: reference ${PG_REFERENCE} not found" >&2; exit 1; }
[ -f "${SWEEP}" ] || { echo "ERROR: ${SWEEP} not found" >&2; exit 1; }
[ -x "${MC_PYTHON}" ] || { echo "ERROR: interpreter ${MC_PYTHON} not executable -- set MOLCRAWL_PYTHON" >&2; exit 1; }
mkdir -p workflows/slurm-logs
echo "[official-indels] MAIN checkout: ${MAIN}"
echo "[official-indels] code (import): ${REPO}"
echo "[official-indels] interpreter:   ${MC_PYTHON}"

DRY=1; [ "${SUBMIT:-0}" = "1" ] && DRY=0
if [ "${DRY}" = "0" ]; then mkdir -p "$(dirname "${RECORD}")"; fi

ONLY="${ONLY:-}"

# arch  size  model_path  slug  scoring_method
JOBS=(
  "gpt2 small  ${RUNS}/gpt2-output/protein_sequence-small-rerun/ckpt.pt     protein_sequence-gpt2-small-official-indels-s42   pll_delta"
  "gpt2 medium ${RUNS}/gpt2-output/protein_sequence-medium-rerun/ckpt.pt    protein_sequence-gpt2-medium-official-indels-s42  pll_delta"
  "gpt2 large  ${RUNS}/gpt2-output/protein_sequence-large-rerun/ckpt.pt     protein_sequence-gpt2-large-official-indels-s42   pll_delta"
  "gpt2 xl     ${RUNS}/gpt2-output/protein_sequence-ex-large-rerun/ckpt.pt  protein_sequence-gpt2-xl-official-indels-s42      pll_delta"
)

common_env() {  # arch model_path slug scoring_method
  printf 'ALL,LEARNING_SOURCE_DIR=%s,MODEL_PATH=%s,MODEL_SLUG=%s,ARCH=%s,MODALITY=protein_sequence,PG_DIR=%s,PG_REFERENCE=%s,SCORING_METHOD=%s,MOLCRAWL_PYTHON=%s,MAX_EXAMPLES=' \
    "${LSD}" "$2" "$3" "$1" "${PG_DIR}" "${PG_REFERENCE}" "$4" "${MC_PYTHON}"
}

echo "=== protein ProteinGym OFFICIAL full-assay INDEL scoring (Issue #388, M3) ==="
echo "assays:  $(ls "${PG_DIR}"/*.csv 2>/dev/null | wc -l) (official indels)"
echo "record:  ${RECORD}"
echo

for spec in "${JOBS[@]}"; do
  read -r arch size mpath slug method <<<"${spec}"
  if [ -n "${ONLY}" ] && [ "${ONLY}" != "${arch}" ] && [ "${ONLY}" != "${arch}-${size}" ]; then
    continue
  fi
  [ -e "${mpath}" ] || { echo "ERROR: missing checkpoint ${mpath}" >&2; exit 1; }
  name="pg-official-indels-${arch}-${size}"
  env="$(common_env "${arch}" "${mpath}" "${slug}" "${method}")"
  echo "${arch} ${size} -> ${slug}  [${method}]"
  if [ "${DRY}" = "0" ]; then
    jid=$(SUBMIT_RECORD="${RECORD}" workflows/sbatch-watched.sh "${name}" -- \
          --job-name="${name}" --export="${env}" "${SWEEP}")
    echo "  submitted ${jid}"
  else
    echo "  DRYRUN sbatch --job-name=${name} --export=${env} ${SWEEP}"
  fi
done

echo
if [ "${DRY}" = "0" ]; then
  echo "DONE. When all jobs drain, aggregate with MUTATION_TYPE=indels:"
else
  echo "DRY RUN -- set SUBMIT=1 to submit. After scoring, aggregate with:"
fi
echo "  sbatch workflows/run-proteingym-official-aggregate.sbatch  # MUTATION_TYPE=indels variant: see that script"
