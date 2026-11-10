#!/bin/bash
#SBATCH --job-name=mc-molnl-retrieval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Retrieval between molecules and their descriptions, both directions, for one model.
#
# The checkpoint is resolved from the run directory rather than passed in, so that what
# was evaluated is the run and not a path somebody typed: nanoGPT runs are read from
# ckpt.pt, HF runs from the best checkpoint their trainer_state names, falling back to
# the newest one.
#
#   sbatch --export=ALL,RUN_DIR=<run>,ARCH=gpt2|bert,SIZE=<size>,LABEL=<name>,\
#          PAIRS_ROOT=<Mol-Instructions dir>,OUT_DIR=<dir> workflows/molnl-retrieval.sbatch
#
# PAIRS_FILE=<jsonl> replaces PAIRS_ROOT and is what the SMolInstruct test split is
# scored from: the pairs are frozen into one file so the scoring and the overlap check
# cannot drift apart, and the molecules are taken in the corpus's own spelling.
#
# PAIRS_FILE is the current path. The outputs of the PAIRS_ROOT round were deleted on
# 2026-11-10 under the judgment's 3.3, so a run with PAIRS_ROOT measures something this
# project no longer holds results for. The Mol-Instructions distribution itself stays --
# the judgment keeps it -- which is why the argument is still here.
#
# Optional: WRAP (plain|inst -- inst writes the corpus's own record form), NOTATION
# (selfies|smiles -- the corpus is SMILES, the evaluation pairs are SELFIES),
# CANDIDATES (default 100), QUERIES (0 = every test pair), QUERIES_T2M (same for
# the description->molecule direction alone, which costs BERT one masked forward per
# molecule token), DIRECTIONS (molecule_to_text, text_to_molecule, or both -- one at a
# time keeps a BERT measurement inside the wall clock limit, and LABEL has to differ or
# the first direction's own output is overwritten), CONTEXT (1024), BERT_MLM_BATCH,
# TOKENIZER_PATH, GPT2_TOKENIZER_DIR, PYTHON.
set -eo pipefail

: "${RUN_DIR:?set RUN_DIR to the run directory to evaluate}"
: "${ARCH:?set ARCH to gpt2 or bert}"
if [ -z "${PAIRS_FILE:-}" ]; then
  : "${PAIRS_ROOT:?set PAIRS_ROOT to the Mol-Instructions dataset directory, or PAIRS_FILE to a frozen pair file}"
fi
: "${OUT_DIR:?set OUT_DIR to where the json and tsv are written}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# A molnl checkpoint carries no tokenizer of its own, so the adapter falls back to the
# corpus's GPT-2 BPE -- which it cannot download on a compute node. Three jobs died 28
# seconds in for want of this line. Same default the corpus scan uses.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
cd "${REPO}"; mkdir -p workflows/slurm-logs "${OUT_DIR}"

if [ "${ARCH}" = "gpt2" ]; then
  MODEL="${RUN_DIR}/ckpt.pt"
  [ -f "${MODEL}" ] || { echo "no ckpt.pt under ${RUN_DIR}" >&2; exit 1; }
else
  MODEL="$("${PYTHON}" - "${RUN_DIR}" <<'PY'
import json, os, sys
run = sys.argv[1]
cks = {int(d.split("-")[1]): os.path.join(run, d)
       for d in os.listdir(run)
       if d.startswith("checkpoint-") and d.split("-")[1].isdigit()
       and os.path.exists(os.path.join(run, d, "trainer_state.json"))}
if not cks:
    sys.exit("no checkpoint with a trainer_state under " + run)
newest = cks[max(cks)]
best = json.load(open(os.path.join(newest, "trainer_state.json"))).get("best_model_checkpoint")
# The named best can have been pruned by the retention callback; fall back to the newest.
print(best if best and os.path.isdir(best) else newest)
PY
)"
fi
echo "=== retrieval ${LABEL:-${ARCH}_${SIZE:-model}} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    run   ${RUN_DIR}"
echo "    model ${MODEL}"

rc=0
"${PYTHON}" scripts/molnl_retrieval_eval.py \
  --model-path "${MODEL}" --arch "${ARCH}" ${SIZE:+--size "${SIZE}"} \
  ${LABEL:+--label "${LABEL}"} ${TOKENIZER_PATH:+--tokenizer-path "${TOKENIZER_PATH}"} \
  ${PAIRS_FILE:+--pairs-file "${PAIRS_FILE}"} \
  ${PAIRS_ROOT:+--pairs-root "${PAIRS_ROOT}"} --out-dir "${OUT_DIR}" \
  --candidates "${CANDIDATES:-100}" --queries "${QUERIES:-0}" \
  --queries-text-to-molecule "${QUERIES_T2M:-0}" \
  ${DIRECTIONS:+--directions "${DIRECTIONS}"} \
  --wrap "${WRAP:-plain}" --molecule-notation "${NOTATION:-selfies}" \
  ${BERT_MLM_BATCH:+--bert-mlm-batch "${BERT_MLM_BATCH}"} \
  --context "${CONTEXT:-1024}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
