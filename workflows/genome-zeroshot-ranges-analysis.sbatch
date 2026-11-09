#!/bin/bash
#SBATCH --job-name=mc-gen-zs-ranges
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# §2, §4 and §6 of the instruction of 2026-11-02, over the per-variant scores
# the 21 bert-w1026 runs now carry.
#
# Five steps, in one job because they all read the same rows:
#
#   1. The rows are what the instruction asks for: the columns are there, the
#      count is 21,350 per run, and there are 21 runs. Checked before anything
#      is computed from them, so a missing column is not found halfway through.
#
#   2. The new pass is held against the one of 2026-10-30, value by value. The
#      reported fold table is built from the earlier aggregates; if the new
#      whole-window numbers differ from them, that is a finding and not a
#      detail, so this step fails the job rather than printing a warning.
#
#   3. The AUROC table over 4 ranges x 3 chromosomes with the count-weighted
#      mean -- the first of the two attachments §6 asks for.
#
#   4. The interval comparison, run once per (control, range): the 5-mer Markov
#      model and the probe's model-free 14-feature control, each over all four
#      ranges. 8 runs of the same bootstrap, 2,000 rounds, seed 1026.
#
#   5. Those 8 outputs gathered into the second attachment, plus the summary
#      table the judgment of 2026-11-09 asks for: the number of runs that
#      cleared each control on all three chromosomes beside that control's own
#      per-chromosome AUROCs (§4.1), and the group test ranked on the
#      count-weighted mean rather than a pooled AUROC (§4.2).
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: SCORES_DIR, OLD_DIR, OUT_DIR, ROUNDS, SEED.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SCORES_DIR="${SCORES_DIR:-${RESULTS_ROOT}/analyses/ep9-ranges}"
OLD_DIR="${OLD_DIR:-${RESULTS_ROOT}/analyses/ep9-window-predictions}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/ep9-ranges}"
ROUNDS="${ROUNDS:-2000}"
SEED="${SEED:-1026}"
MARKOV="${RESULTS_ROOT}/clinvar-1024/zeroshot-baseline-scores.jsonl"
FREE="${RESULTS_ROOT}/clinvar-probe-bert-untrained-v2/model_free/predictions.jsonl"
RANGES="window flank128 flank32 point"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== scores: ${SCORES_DIR} ==="
echo "=== out:    ${OUT_DIR} ==="
echo "=== rounds=${ROUNDS} seed=${SEED} ==="
for f in "${MARKOV}" "${FREE}"; do
    [ -s "${f}" ] || { echo "no control at ${f}" >&2; exit 1; }
done
rc=0

echo ""; echo "################ 1. 行そのものの点検 ################"
"${PY}" scripts/check_per_variant_outputs.py \
    --dir "${SCORES_DIR}" --pattern 'window-*-predictions.jsonl' \
    --require vcv_id,chrom,pos,ref_allele,alt_allele,clinical_significance \
    --require label_pathogenic,score_window,score_flank128,score_flank32,score_point \
    --expect-rows 21350 --expect-files 21 || rc=1
[ "${rc}" -eq 0 ] || { echo "行が条件を満たさない。ここで止める" >&2; exit 1; }

echo ""; echo "################ 2. 2026-10-30 の採点との突き合わせ ################"
"${PY}" scripts/compare_zeroshot_passes.py \
    --new "${SCORES_DIR}" --old "${OLD_DIR}" --tolerance 0 || rc=1

echo ""; echo "################ 3. 範囲 x 染色体の AUROC 表 ################"
"${PY}" scripts/clinvar_zeroshot_range_table.py \
    --scores-dir "${SCORES_DIR}" \
    --out "${OUT_DIR}/zeroshot-bert-w1026-ranges.tsv" || rc=1

echo ""; echo "################ 4. 対照との区間比較 ################"
SPECS=()
for r in ${RANGES}; do
    for ctl in markov free; do
        case "${ctl}" in
            markov) SRC="${MARKOV}"; KEY=markov;  NAME="5-mer Markov" ;;
            free)   SRC="${FREE}";   KEY=score;   NAME="モデル不使用（14 特徴）" ;;
        esac
        OUT="${OUT_DIR}/interval-${ctl}-${r}.json"
        echo ""; echo "---- ${ctl} / ${r} ----"
        "${PY}" scripts/clinvar_final_analysis.py \
            --scores-dir "${SCORES_DIR}" \
            --scores-pattern 'window-*-predictions.jsonl' \
            --score-key "score_${r}" \
            --baseline-scores "${SRC}" --baseline-key "${KEY}" \
            --baseline-name "${NAME}" \
            --rounds "${ROUNDS}" --seed "${SEED}" --out "${OUT}" || rc=1
        [ -s "${OUT}" ] && SPECS+=(--json "${r}:${OUT}")
    done
done

echo ""; echo "################ 5. 区間の表 ################"
if [ "${#SPECS[@]}" -gt 0 ]; then
    "${PY}" scripts/clinvar_range_interval_table.py "${SPECS[@]}" \
        --out "${OUT_DIR}/zeroshot-bert-w1026-intervals.tsv" \
        --summary-out "${OUT_DIR}/zeroshot-bert-w1026-controls-groups.tsv" || rc=1
else
    echo "区間の出力が 1 つも無い" >&2; rc=1
fi

echo ""; echo "---- rc=${rc} ----"
ls -la "${OUT_DIR}"/*.tsv "${OUT_DIR}"/interval-*.json 2>/dev/null | sed 's/^/  /'
echo "=== done @ $(date -Is) ==="
exit ${rc}
