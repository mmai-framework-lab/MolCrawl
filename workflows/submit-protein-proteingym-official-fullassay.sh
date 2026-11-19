#!/usr/bin/env bash
# Full-assay ProteinGym substitution scoring for the OFFICIAL aggregation
# (Issue #388, M2). Scores every variant of all 217 DMS substitution assays
# (no 200-variant subsample) against the official release data, with the
# official scoring methods:
#
#   GPT-2 small/medium/large/xl -> causal mean log-likelihood (pll_delta),
#                                  now batched by sequence length.
#   BERT  small/medium/large    -> masked marginals (official BERT protocol),
#                                  cost ~ O(seq_len) per assay.
#
# Adopted checkpoints (argmin eval_loss_mask = best_model_checkpoint for BERT;
# nanoGPT best-val ckpt.pt for GPT-2), the same weights the seed-42 tables use:
#   GPT-2 {small,medium,large,xl} -> gpt2-output/protein_sequence-{small,medium,large,ex-large}-rerun/ckpt.pt
#   BERT  small  -> bert-output/protein_sequence-small-lr3e4/checkpoint-32000
#   BERT  medium -> bert-output/protein_sequence-medium-lr1e4/checkpoint-32900
#   BERT  large  -> bert-output/protein_sequence-large-lr3p5e5-rerun/checkpoint-32000
#
# New eval slugs (…-official-s42) so the 200-example subsample outputs are left
# untouched. One job per model; each resumes on the next assay if a wall cut
# occurs (the sweep skips assays whose metrics.json is already complete).
# Full-assay cost is small with the efficient scoring (BERT is variant-count
# independent; GPT-2 batches same-length variants), so no afterany chains.
#
# Run from the main checkout (data + conda env live there):
#   SUBMIT=1 workflows/submit-protein-proteingym-official-fullassay.sh
# Without SUBMIT=1 it prints what it would submit and exits (dry run).
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO}"

# Data, checkpoints and the interpreter live in the MAIN checkout even when this
# runs from a worktree (learning_source* and ./miniconda are not in a worktree).
# Resolve the main checkout through git's common dir (portable-run-paths rule 3).
MAIN="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
[ -n "${MAIN}" ] || MAIN="${REPO}"
LSD="${LEARNING_SOURCE_DIR:-${MAIN}/learning_source_20260730_protein_uniref50}"
RUNS="${PROTEIN_RUNS:-${MAIN}/learning_source_protein_runs/protein_sequence}"
# The job imports molcrawl from THIS checkout (its cwd shadows the editable
# install), so a worktree run uses the worktree's code; the interpreter is the
# main checkout's miniconda, passed to common_functions via MOLCRAWL_PYTHON.
MC_PYTHON="${MOLCRAWL_PYTHON:-${MAIN}/miniconda/bin/python}"
SWEEP="workflows/eval-gpu-proteingym-sweep.sbatch"
PG_DIR="${LSD}/eval/proteingym/official/data/DMS_ProteinGym_substitutions"
RECORD="${SUBMIT_RECORD:-workflows/submit-records/protein-proteingym-official-fullassay-$(date +%Y%m%d).tsv}"

for d in "${PG_DIR}" "${RUNS}"; do
  [ -d "${d}" ] || { echo "ERROR: ${d} not found (MAIN=${MAIN})" >&2; exit 1; }
done
[ -f "${SWEEP}" ] || { echo "ERROR: ${SWEEP} not found" >&2; exit 1; }
[ -x "${MC_PYTHON}" ] || { echo "ERROR: interpreter ${MC_PYTHON} not executable -- set MOLCRAWL_PYTHON" >&2; exit 1; }
mkdir -p workflows/slurm-logs
echo "[official-fullassay] MAIN checkout: ${MAIN}"
echo "[official-fullassay] code (import): ${REPO}"
echo "[official-fullassay] interpreter:   ${MC_PYTHON}"

DRY=1; [ "${SUBMIT:-0}" = "1" ] && DRY=0
if [ "${DRY}" = "0" ]; then mkdir -p "$(dirname "${RECORD}")"; fi

# arch  size  model_path  slug  scoring_method
JOBS=(
  "gpt2 small  ${RUNS}/gpt2-output/protein_sequence-small-rerun/ckpt.pt            protein_sequence-gpt2-small-official-s42   pll_delta"
  "gpt2 medium ${RUNS}/gpt2-output/protein_sequence-medium-rerun/ckpt.pt           protein_sequence-gpt2-medium-official-s42  pll_delta"
  "gpt2 large  ${RUNS}/gpt2-output/protein_sequence-large-rerun/ckpt.pt            protein_sequence-gpt2-large-official-s42   pll_delta"
  "gpt2 xl     ${RUNS}/gpt2-output/protein_sequence-ex-large-rerun/ckpt.pt         protein_sequence-gpt2-xl-official-s42      pll_delta"
  "bert small  ${RUNS}/bert-output/protein_sequence-small-lr3e4/checkpoint-32000   protein_sequence-bert-small-official-s42   masked_marginals"
  "bert medium ${RUNS}/bert-output/protein_sequence-medium-lr1e4/checkpoint-32900  protein_sequence-bert-medium-official-s42  masked_marginals"
  "bert large  ${RUNS}/bert-output/protein_sequence-large-lr3p5e5-rerun/checkpoint-32000 protein_sequence-bert-large-official-s42 masked_marginals"
)

common_env() {  # arch model_path slug scoring_method
  # MAX_EXAMPLES= (trailing empty) scores the FULL assay. No value contains a
  # comma, so the comma-separated --export list is safe. MOLCRAWL_PYTHON points
  # common_functions at the main checkout's interpreter for a worktree run.
  printf 'ALL,LEARNING_SOURCE_DIR=%s,MODEL_PATH=%s,MODEL_SLUG=%s,ARCH=%s,MODALITY=protein_sequence,PG_DIR=%s,SCORING_METHOD=%s,MOLCRAWL_PYTHON=%s,MAX_EXAMPLES=' \
    "${LSD}" "$2" "$3" "$1" "${PG_DIR}" "$4" "${MC_PYTHON}"
}

echo "=== protein ProteinGym OFFICIAL full-assay scoring (Issue #388, M2) ==="
echo "assays:  $(ls "${PG_DIR}"/*.csv 2>/dev/null | wc -l) (official substitutions)"
echo "record:  ${RECORD}"
echo

# ONLY restricts which jobs are (re)submitted, e.g. ONLY=bert to resubmit only
# the three BERT sizes, or ONLY='bert-large' for one. Matches against
# "<arch>" and "<arch>-<size>". Empty = all seven. The sweep skips any assay
# whose metrics.json is already complete, so a resubmit only fills the gaps
# (e.g. the long-protein assays a prior bug left unscored) and never touches a
# model whose job is still running or already finished.
ONLY="${ONLY:-}"

for spec in "${JOBS[@]}"; do
  read -r arch size mpath slug method <<<"${spec}"
  if [ -n "${ONLY}" ] && [ "${ONLY}" != "${arch}" ] && [ "${ONLY}" != "${arch}-${size}" ]; then
    continue
  fi
  [ -e "${mpath}" ] || { echo "ERROR: missing checkpoint ${mpath}" >&2; exit 1; }
  name="pg-official-${arch}-${size}"
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
  echo "DONE. When all jobs drain, confirm completeness then aggregate:"
else
  echo "DRY RUN -- set SUBMIT=1 to submit. After scoring completes, aggregate with:"
fi
cat <<'NEXT'
  # 1. confirm every assay scored (content-based, not REPORT.md existence):
  for s in protein_sequence-gpt2-small-official-s42 protein_sequence-gpt2-medium-official-s42 \
           protein_sequence-gpt2-large-official-s42 protein_sequence-gpt2-xl-official-s42 \
           protein_sequence-bert-small-official-s42 protein_sequence-bert-medium-official-s42 \
           protein_sequence-bert-large-official-s42; do
    ./miniconda/bin/python scripts/check_proteingym_sweep_complete.py --slug "$s"
  done
  # 2. emit official per-assay CSVs under <base>/scores/zero_shot/substitutions/<MODEL>/:
  #    (one --model-name per registered model; see scripts/proteingym_emit_official.py)
  # 3. register the 7 models and run the official aggregation:
  #    scripts/proteingym_prepare_config.py --models MolCrawl_GPT2_small ... --replace
  #    workflows/eval-proteingym-official-aggregate.sh   # merge.py + performance_DMS_benchmarks.py
NEXT
