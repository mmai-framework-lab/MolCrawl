#!/bin/bash
#SBATCH --job-name=mc-molnl-retrieval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
# Optional: CANDIDATES (default 100), QUERIES (0 = every test pair), CONTEXT (1024),
# TOKENIZER_PATH, GPT2_TOKENIZER_DIR, PYTHON.
set -eo pipefail

: "${RUN_DIR:?set RUN_DIR to the run directory to evaluate}"
: "${ARCH:?set ARCH to gpt2 or bert}"
: "${PAIRS_ROOT:?set PAIRS_ROOT to the Mol-Instructions dataset directory}"
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
  --pairs-root "${PAIRS_ROOT}" --out-dir "${OUT_DIR}" \
  --candidates "${CANDIDATES:-100}" --queries "${QUERIES:-0}" \
  --context "${CONTEXT:-1024}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
