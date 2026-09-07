#!/bin/bash
#SBATCH --job-name=mc-molnl-bert
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# mol_nl BERT ladder on the shuffled corpus.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,SIZE=medium,STEPS=12000,SEGMENT=1 \
#     workflows/molnl-bert-ladder.sbatch
#
# 4 GPUs are REQUIRED, not an optimisation. Under HF Trainer the effective global
# batch is batch_size * grad_accum * world_size, so 8 * 80 * 4 = 2,560; a different
# GPU count trains a different model. The configs declare expected_global_batch and
# main.py refuses to start when the product differs, so this fails rather than
# quietly producing something incomparable.
#
# The config is UNMODIFIED. Only operational values are passed here:
#   --max_steps      the step budget is a research value pending the cross-modality
#                    decision, so it is passed rather than edited into the config.
#                    The config's 373 (3 epochs) is known to be far short.
#   --log_interval / --save_steps
#                    main.py sets eval_steps = log_interval and forces
#                    load_best_model_at_end, and HF requires save_steps % eval_steps
#                    == 0. 250/250 over 12,000 steps gives 48 eval points and keeps
#                    every one of them checkpointable.
#
# SPLIT RUNS. medium and large exceed the partition's 4-day limit and are run in
# segments. main.py resolves the checkpoint itself -- it lists model_path, sorts by
# step and resumes from the highest -- so "resume from the newest" cannot be got
# wrong by hand.
#
# The other half can be got wrong, and silently. Retention rebuilds its ranking from
# state.log_history, which HF restores from the checkpoint being resumed; point a
# later segment at a different directory and main.py finds nothing, prints
# "Starting training from scratch...", and spends four days re-running step 0 while
# the earlier checkpoints sit where nothing is looking for them. On large that is one
# mistake out of four resumes.
#
# So OUT is derived here rather than accepted, and SEGMENT states which one this is:
#   SEGMENT=1        first segment; refuses if checkpoints already exist
#   SEGMENT=2,3,...  a continuation; refuses if the directory holds none
# Both directions fail before a GPU is touched.
#
# STEPS must not change between segments either. HF decays the learning rate over
# max_steps and has no separate decay length, so a segment given a smaller STEPS does
# not stop early -- it runs a compressed schedule and trains a different model, at the
# same output directory, resuming cleanly. Segmentation happens by hitting the time
# limit, not by shortening the run. The newest checkpoint's trainer_state.json carries
# the max_steps it was trained under, and a continuation is refused unless STEPS
# matches it.
set -eo pipefail
: "${SIZE:?SIZE=small|medium|large}"
: "${SEGMENT:?SEGMENT=1 for the first, 2,3,... for each continuation}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
STEPS="${STEPS:-12000}"
EVERY="${EVERY:-250}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_${SIZE}.py"

# A git worktree holds only tracked files, so miniconda and assets/ are not inside
# it. Resolve them through git's common dir when running from one.
if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || {
  echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun" >&2; exit 1; }
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

# Outputs must not land in a worktree: learning_source* is gitignored, so each
# checkout has its own, and a worktree is meant to be disposable. RUNS_ROOT is the
# one place to override that, and the log says which root was used.
RUNS_ROOT="${RUNS_ROOT:-${LEARNING_SOURCE_DIR}/molecule_nat_lang}"
OUT="${RUNS_ROOT}/bert-output-shuffled/molecule_nat_lang-${SIZE}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs

# Segment guard. Counted before anything is loaded, so a wrong path costs seconds.
_n_ckpt=$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)
_last=$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ "${SEGMENT}" = "1" ] && [ "${_n_ckpt}" -ne 0 ]; then
  echo "REFUSING: SEGMENT=1 asks to start from scratch, but ${OUT} already holds ${_n_ckpt}" \
       "checkpoint(s) (newest step ${_last}). main.py would resume from the newest," \
       "so this would not be the fresh start it was asked for. Use SEGMENT=2 to" \
       "continue, or move the directory aside." >&2
  exit 1
fi
if [ "${SEGMENT}" != "1" ] && [ "${_n_ckpt}" -eq 0 ]; then
  echo "REFUSING: SEGMENT=${SEGMENT} is a continuation, but ${OUT} holds no checkpoint." \
       "main.py would print \"Starting training from scratch\" and spend the whole" \
       "time limit re-running step 0. Check that SIZE=${SIZE}, LEARNING_SOURCE_DIR" \
       "and RUNS_ROOT match the earlier segment -- OUT is derived from all three." >&2
  exit 1
fi

# The schedule must be the same run continued, not a new one written over it.
if [ "${SEGMENT}" != "1" ]; then
  _state="${OUT}/checkpoint-${_last}/trainer_state.json"
  [ -f "${_state}" ] || {
    echo "REFUSING: ${_state} is missing, so the schedule this checkpoint was trained" \
         "under cannot be read and STEPS=${STEPS} cannot be checked against it." >&2
    exit 1; }
  read -r _prev_steps _prev_done <<EOF
$("${PYTHON}" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["max_steps"], int(s["global_step"]))' "${_state}")
EOF
  if [ "${_prev_steps}" != "${STEPS}" ]; then
    echo "REFUSING: the checkpoint at step ${_last} was trained with max_steps=${_prev_steps}," \
         "and this segment passes STEPS=${STEPS}. HF decays the learning rate over" \
         "max_steps with no separate decay length, so resuming under a different value" \
         "compresses or stretches the schedule and continues a different run into the" \
         "same directory. Pass STEPS=${_prev_steps}, or start a new run elsewhere." >&2
    exit 1
  fi
  if [ "${_prev_done}" -ge "${STEPS}" ]; then
    echo "REFUSING: the run already reached step ${_prev_done} of ${STEPS}, so there is" \
         "nothing for this segment to do." >&2
    exit 1
  fi
  echo "schedule check: max_steps ${_prev_steps} matches, ${_prev_done}/${STEPS} steps done"
fi

echo "=== mol_nl bert-${SIZE} (shuffled) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified)"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}"
echo "steps=${STEPS}  eval/save every ${EVERY}  global batch 8x80x4 = 2560 (4-GPU launch required)"
echo "segment=${SEGMENT}  ${_n_ckpt} checkpoint(s) present  $([ "${SEGMENT}" = 1 ] && echo 'starting from scratch' || echo "resuming from step ${_last}")"
echo "--- config values under test ---"
grep -hE "^max_steps|^learning_rate|^weight_decay|^batch_size|^gradient_accumulation_steps|^max_length|^document_masking|^dataset_dir|^seed|^expected_global_batch|^degenerate" "${CFG}" | sed 's/^/    /'
echo "baselines (mol_nl, job 22503): unigram 3.8638 / degenerate 3.4774 / left 2.2533 / both 1.2804"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CFG}" \
  --model_path="${OUT}" \
  --max_steps="${STEPS}" --log_interval="${EVERY}" --save_steps="${EVERY}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
