#!/usr/bin/env bash
#SBATCH --job-name=pg-official-aggregate
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Official ProteinGym DMS zero-shot *substitution* aggregation for the seven
# MolCrawl protein models (Issue #388, M2 -- the emit -> register -> aggregate
# chain that turns the per-assay sweep outputs into the official five metrics).
# CPU only: the scoring is already done; this just reshapes and runs the
# official merge.py + performance_DMS_benchmarks.py. No GPU is requested, but
# compute still belongs on a node, not the login host.
#
# Submit from the checkout whose code you want to run (a worktree is fine; its
# scripts import correctly because cwd shadows the editable install):
#   sbatch workflows/run-proteingym-official-aggregate.sbatch
#
# The scored outputs, vendored ProteinGym and the interpreter all live in the
# MAIN checkout, so resolve it through git-common-dir (portable-run-paths).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"
mkdir -p workflows/slurm-logs

if [ -x "${REPO}/miniconda/bin/python" ]; then
  MAIN="${REPO}"
else
  MAIN="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${MAIN}/miniconda/bin/python}"
LSD="${LEARNING_SOURCE_DIR:-${MAIN}/learning_source_20260730_protein_uniref50}"
BASE="${LSD%/}/eval/proteingym/official"
SCORES="${BASE}/scores/zero_shot/substitutions"
CATEGORY="model_list_zero_shot_substitutions_DMS"

[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
[ -d "${BASE}" ] || { echo "official tree not found: ${BASE}" >&2; exit 1; }

echo "=== pg official aggregate on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "code: ${REPO}  $(git rev-parse --short HEAD 2>/dev/null)"
echo "LSD:  ${LSD}"

# Registered model name  <-  MolCrawl eval slug. One entry per model; each is a
# distinct registered model so the official table reports it on its own row.
MODELS=(
  "MolCrawl_GPT2_small  protein_sequence-gpt2-small-official-s42"
  "MolCrawl_GPT2_medium protein_sequence-gpt2-medium-official-s42"
  "MolCrawl_GPT2_large  protein_sequence-gpt2-large-official-s42"
  "MolCrawl_GPT2_xl     protein_sequence-gpt2-xl-official-s42"
  "MolCrawl_BERT_small  protein_sequence-bert-small-official-s42"
  "MolCrawl_BERT_medium protein_sequence-bert-medium-official-s42"
  "MolCrawl_BERT_large  protein_sequence-bert-large-official-s42"
)

echo "--- 1. emit per-assay official CSVs (${#MODELS[@]} models) ---"
names=()
for spec in "${MODELS[@]}"; do
  read -r name slug <<<"${spec}"
  names+=("${name}")
  LEARNING_SOURCE_DIR="${LSD}" "${PYTHON}" scripts/proteingym_emit_official.py \
    --slug "${slug}" --model-name "${name}" \
    --mutation-type substitutions --out "${SCORES}"
done

echo "--- 2. register the ${#names[@]} models in a config clone (references blank) ---"
"${PYTHON}" scripts/proteingym_prepare_config.py \
  --base "${BASE}" --category "${CATEGORY}" \
  --models "${names[@]}" --key mutant --replace

echo "--- 3. official merge.py + performance_DMS_benchmarks.py ---"
LEARNING_SOURCE_DIR="${LSD}" MUTATION_TYPE=substitutions SCORES="${SCORES}" PYTHON="${PYTHON}" \
  bash workflows/eval-proteingym-official-aggregate.sh

echo "=== DONE @ $(date -Is); results under ${BASE}/results/zero_shot/substitutions ==="
