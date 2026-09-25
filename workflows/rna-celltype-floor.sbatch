#!/bin/bash
# The floor of the cell-type probe: the same probe on untrained trunks.
#
# 4 sizes x 3 seeds. The spread across seeds is what says whether the dips in
# f1_macro are real and whether 0.0008 between large and xl is a difference at
# all; the level says how much of the accuracy comes from shape, vocabulary and
# tokenisation rather than from pretraining.
#
# The untrained weights are written as checkpoints and fed to the same probe the
# thirteen trained points went through, so the floor and the runs differ in the
# weights and in nothing else.
#
# Submit one job per size (FLOOR_SIZES=xl, and so on). The trained points took 17
# minutes at small and 2h48 at xl, so all twelve in one job would run past any
# sensible wall; per size the longest is under nine hours. A job that is cut off
# can simply be submitted again: a seed whose metrics.json already exists is
# skipped, so nothing is measured twice.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=rna-celltype-floor
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
# The shape comes out of a grid run directory. GPT-2 and BERT hang off
# MODEL_OUTPUT_ROOT at different depths, so the GPT-2 root is named rather than
# built from the other one.
: "${GPT2_RUNS_ROOT:?export GPT2_RUNS_ROOT to the directory holding the rna-gpt2-* run directories}"
[ -d "${GPT2_RUNS_ROOT}" ] || { echo "GPT2_RUNS_ROOT=${GPT2_RUNS_ROOT} is not a directory" >&2; exit 1; }
: "${EVAL_JSONL:?export EVAL_JSONL to the evaluation JSONL}"
: "${FLOOR_OUT_DIR:?export FLOOR_OUT_DIR to where the floor results go, outside every input tree}"
: "${TOKENIZER_DIR:?export TOKENIZER_DIR to the tokenizer directory}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
CKPT_DIR="${FLOOR_OUT_DIR}/_untrained"
mkdir -p workflows/slurm-logs "${FLOOR_OUT_DIR}" "${CKPT_DIR}"

echo "=== rna-celltype-floor @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "=== outputs: ${FLOOR_OUT_DIR} ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# The rate is irrelevant to an untrained trunk; 6e-4 is named only because its
# run is the one whose checkpoint carries the shape for that size.
for SIZE in ${FLOOR_SIZES:-small medium large xl}; do
    SHAPE="${GPT2_RUNS_ROOT}/rna-gpt2-${SIZE}-lr6e4/checkpoint-40320/training_state.bin"
    [ -f "${SHAPE}" ] || { echo "no checkpoint to take the ${SIZE} shape from: ${SHAPE}" >&2; exit 1; }
    for S in ${FLOOR_SEEDS:-1 2 3}; do
            CK="${CKPT_DIR}/${SIZE}-seed${S}.pt"
            OUT="${FLOOR_OUT_DIR}/${SIZE}-seed${S}"
            if [ -f "${OUT}/metrics.json" ]; then
                echo "--- ${SIZE} seed ${S}: already measured, skipping"
                continue
            fi
            echo "--- ${SIZE} seed ${S} @ $(date -Is)"
            "${PY}" scripts/rna_make_untrained_ckpt.py \
                --from-checkpoint "${SHAPE}" --out "${CK}" --seed "${S}"
            "${PY}" -m molcrawl.tasks.evaluation.tabula_sapiens \
                --model-path "${CK}" \
                --jsonl-path "${EVAL_JSONL}" \
                --output-dir "${OUT}" \
                --tokenizer-path "${TOKENIZER_DIR}" \
                --arch gpt2 --modality rna \
                ${MAX_CELLS:+--max-cells "${MAX_CELLS}"}
            rm -f "${CK}"   # the weights are a function of the seed; keep the result, not the file
    done
done
echo "=== done @ $(date -Is) ==="
