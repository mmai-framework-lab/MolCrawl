#!/bin/bash
#SBATCH --job-name=mc-gen-clinvar-results
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The campaign's two remaining analyses, in one job.
#
#   1. One table over 21 subsets, three folds, both architectures and both
#      methods, carrying the model-free control and the untrained floor as rows
#      of the same table rather than as numbers in another file.
#
#   2. Three paired comparisons: BERT zero-shot against BERT probe, GPT-2
#      zero-shot against BERT zero-shot, GPT-2 probe against BERT probe. Both
#      sides score the same variants, so the difference is taken variant by
#      variant and bootstrapped over variants -- at these sample sizes two
#      AUROCs can differ by less than the interval width of either one.
#
# Folds are never pooled: chr21 is 21% pathogenic, chr22 20%, chrX 48%, and
# AUROC moves with class balance.
#
# No GPU work happens here. It runs on the gpu partition because that is where
# compute nodes are, and nothing in this repo runs on the login node.
#
# Required: RESULTS_ROOT. Optional: OUT_DIR, ROUNDS, SEED.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/results}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== results: ${RESULTS_ROOT} ==="
echo "=== out:     ${OUT_DIR} ==="

# The fold analyses used for the table are the ones with the matched-range
# second correlation (analyses/corr2), not the earlier per-directory ones: the
# AUROCs are identical between them, and taking both from one place keeps the
# table and the correlation describing the same run of the same analysis.
CORR2="${RESULTS_ROOT}/analyses/corr2"
rc=0

echo ""; echo "################ 1. 結果表 ################"
"${PY}" scripts/clinvar_results_table.py \
    --zeroshot-bert "${CORR2}/zeroshot-bert.json" \
    --zeroshot-gpt2 "${CORR2}/zeroshot-gpt2.json" \
    --probe-bert    "${CORR2}/probe-bert.json" \
    --probe-gpt2    "${CORR2}/probe-gpt2.json" \
    --probe-json-bert "${RESULTS_ROOT}/clinvar-probe-bert-w1026-v2/probe.json" \
    --probe-json-gpt2 "${RESULTS_ROOT}/clinvar-probe-gpt2/probe.json" \
    --untrained-bert  "${RESULTS_ROOT}/clinvar-probe-bert-untrained-v2/probe.json" \
    --untrained-gpt2  "${RESULTS_ROOT}/clinvar-probe-gpt2-untrained/probe.json" \
    --out-tsv "${OUT_DIR}/clinvar-results.tsv" \
    --out-md  "${OUT_DIR}/clinvar-results.md" || rc=1

echo ""; echo "################ 2. アーキテクチャ間の差 ################"
pair() {   # name a_dir b_dir a_name b_name
    echo ""; echo "---- $1 ----"
    "${PY}" scripts/clinvar_arch_pairs.py \
        --a-dir "${RESULTS_ROOT}/$2" --b-dir "${RESULTS_ROOT}/$3" \
        --a-name "$4" --b-name "$5" \
        --rounds "${ROUNDS:-2000}" --seed "${SEED:-1026}" \
        --out "${OUT_DIR}/pair-$1.json" || rc=1
}
pair bert-probe-vs-bert-zeroshot \
     clinvar-probe-bert-w1026-v2-runs-only clinvar-zeroshot-bert-w1026 \
     "BERT probe" "BERT zeroshot"
pair gpt2-zeroshot-vs-bert-zeroshot \
     clinvar-zeroshot-gpt2-w1024 clinvar-zeroshot-bert-w1026 \
     "GPT-2 zeroshot" "BERT zeroshot"
pair gpt2-probe-vs-bert-probe \
     clinvar-probe-gpt2-runs-only clinvar-probe-bert-w1026-v2-runs-only \
     "GPT-2 probe" "BERT probe"

echo ""; echo "---- rc=${rc} ----"
ls -la "${OUT_DIR}" | sed 's/^/  /'
echo "=== done @ $(date -Is) ==="
exit ${rc}
