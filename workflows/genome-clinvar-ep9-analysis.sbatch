#!/bin/bash
#SBATCH --job-name=mc-gen-clinvar-ep9
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The 9-epoch series' ClinVar analysis, and the comparisons it makes possible.
#
# Four things, in one job because they read the same predictions:
#
#   1. Fold-wise analysis of the 9-epoch zero-shot and probe, against the same
#      controls the 3-epoch series was measured against: the order-4 Markov
#      model for zero-shot, the probe's own 14-feature control for the probe.
#
#   2. The same for the 3-epoch zero-shot at window 1024, which has no
#      fold-analysis.json yet. Without it the epoch comparison has nothing on
#      the other side.
#
#   3. Architecture differences with the epochs matched. Until the 9-epoch GPT-2
#      finished, BERT's 9 epochs could only be set against GPT-2's 3.
#
#   4. The epoch difference itself, paired variant by variant.
#
# WHY WINDOW 1024 AND NOT clinvar-zeroshot-gpt2. That directory scored at
# context_length 512 with flank 64; the 9-epoch run and clinvar-zeroshot-gpt2-w1024
# both scored at 1024 with flank 512. Comparing across the two measures the
# scoring window, not the epochs: it turns 7 of 21 improving into 21 of 21.
#
# WHY THE PROBE NEEDS A runs-only VIEW. The probe writes its control beside the
# runs as model_free/, and clinvar_final_analysis.py reads every subdirectory as
# a run -- the control would be compared against itself. The existing views are
# symlink farms made by hand; this builds one per job under its own temp dir so
# nothing permanent has to be kept in step with the runs.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
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

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/ep9}"
TMP="${TMPDIR:-/tmp}/clinvar-ep9-${SLURM_JOB_ID:-0}"
ROUNDS="${ROUNDS:-2000}"
SEED="${SEED:-1026}"
MARKOV="${RESULTS_ROOT}/clinvar-1024/zeroshot-baseline-scores.jsonl"
mkdir -p "${OUT_DIR}" "${TMP}"
trap 'rm -rf "${TMP}"' EXIT

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== results: ${RESULTS_ROOT} ==="
echo "=== out:     ${OUT_DIR} ==="
echo "=== rounds=${ROUNDS} seed=${SEED} ==="
rc=0

# A view of a probe directory holding only its runs.
runs_only() {   # src_dir -> prints the view's path
    local src="$1" view="${TMP}/$(basename "$1")-runs-only"
    mkdir -p "${view}"
    for d in "${src}"/*/; do
        case "$(basename "${d}")" in
            model_free|untrained*) continue ;;
        esac
        ln -sfn "${d}" "${view}/$(basename "${d}")"
    done
    echo "${view}"
}

fold() {   # name scores_dir baseline_file baseline_key baseline_name
    echo ""; echo "################ fold 別: $1 ################"
    echo "  scores=$2 ($(ls -d "$2"/*/ 2>/dev/null | wc -l) runs) baseline=$4"
    "${PY}" scripts/clinvar_final_analysis.py \
        --scores-dir "$2" --baseline-scores "$3" --baseline-key "$4" \
        --baseline-name "$5" --rounds "${ROUNDS}" --seed "${SEED}" \
        --out "${OUT_DIR}/$1.json" || rc=1
}

pair() {   # name a_dir b_dir a_name b_name
    echo ""; echo "################ 対応のある差: $1 ################"
    "${PY}" scripts/clinvar_arch_pairs.py \
        --a-dir "$2" --b-dir "$3" --a-name "$4" --b-name "$5" \
        --rounds "${ROUNDS}" --seed "${SEED}" \
        --out "${OUT_DIR}/pair-$1.json" || rc=1
}

PROBE_EP9="$(runs_only "${RESULTS_ROOT}/clinvar-probe-gpt2-ep9")"
PROBE_EP3="${RESULTS_ROOT}/clinvar-probe-gpt2-runs-only"
PROBE_BERT="${RESULTS_ROOT}/clinvar-probe-bert-w1026-v2-runs-only"
ZS_EP9="${RESULTS_ROOT}/clinvar-zeroshot-gpt2-ep9"
ZS_EP3="${RESULTS_ROOT}/clinvar-zeroshot-gpt2-w1024"
ZS_BERT="${RESULTS_ROOT}/clinvar-zeroshot-bert-w1026"

########## 1-2. fold 別 ##########
fold zeroshot-gpt2-ep9 "${ZS_EP9}" "${MARKOV}" markov "5-mer Markov"
fold probe-gpt2-ep9 "${PROBE_EP9}" \
     "${RESULTS_ROOT}/clinvar-probe-gpt2-ep9/model_free/predictions.jsonl" \
     score "モデル不使用（14 特徴）"

########## 3. エポックを揃えたアーキテクチャ差 ##########
pair ep9-gpt2-zeroshot-vs-bert-zeroshot "${ZS_EP9}" "${ZS_BERT}" \
     "GPT-2 zeroshot 9ep" "BERT zeroshot 9ep"
pair ep9-gpt2-probe-vs-bert-probe "${PROBE_EP9}" "${PROBE_BERT}" \
     "GPT-2 probe 9ep" "BERT probe 9ep"

########## 4. エポックの差そのもの（WITH_EP3=1 のときだけ） ##########
# 2026-10-30 の判断書により、3 エポックの系列は図表にも本文にも出さない。
# 出力は消していないので、WITH_EP3=1 で元どおり出る。
if [ -n "${WITH_EP3:-}" ]; then
    fold zeroshot-gpt2-ep3-w1024 "${ZS_EP3}" "${MARKOV}" markov "5-mer Markov"
    pair gpt2-zeroshot-ep9-vs-ep3 "${ZS_EP9}" "${ZS_EP3}" \
         "GPT-2 zeroshot 9ep" "GPT-2 zeroshot 3ep"
    pair gpt2-probe-ep9-vs-ep3 "${PROBE_EP9}" "${PROBE_EP3}" \
         "GPT-2 probe 9ep" "GPT-2 probe 3ep"
fi

echo ""; echo "---- rc=${rc} ----"
ls -la "${OUT_DIR}" | sed 's/^/  /'
echo "=== done @ $(date -Is) ==="
exit ${rc}
