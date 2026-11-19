#!/bin/bash
#SBATCH --job-name=mc-molnl-bert
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# mol_nl BERT ladder on the shuffled corpus.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch \
#     --export=ALL,CONFIG=molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_medium_lr3e4.py,SEGMENT=1 \
#     workflows/molnl-bert-ladder.sbatch
#
# 4 GPUs are REQUIRED, not an optimisation. Under HF Trainer the effective global
# batch is batch_size * grad_accum * world_size, so 8 * 80 * 4 = 2,560; a different
# GPU count trains a different model. The configs declare expected_global_batch and
# main.py refuses to start when the product differs, so this fails rather than
# quietly producing something incomparable.
#
# The config is UNMODIFIED. Only operational values are passed here:
# Only --model_path is passed. max_steps, the evaluation and save intervals, warmup, the
# learning rate and the seed all live in the config, so what a run was is answerable from
# the file it names rather than from the submit command that started it -- and the nine
# arms of the learning-rate grid cannot drift apart in anything but the learning rate.
#
# SPLIT RUNS. medium and large exceed the partition's 4-day limit and are run in
# segments, so the time limit here is the partition maximum: a segment should be cut
# by the limit, not finish under it with time left unused. main.py resolves the checkpoint itself -- it lists model_path, sorts by
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
# max_steps must not change between segments either. HF decays the learning rate over
# max_steps and has no separate decay length, so a segment given a smaller STEPS does
# not stop early -- it runs a compressed schedule and trains a different model, at the
# same output directory, resuming cleanly. Segmentation happens by hitting the time
# limit, not by shortening the run. The newest checkpoint's trainer_state.json carries
# the max_steps it was trained under, and a continuation is refused unless the config's
# max_steps matches it.
#
#   EXTEND=1       continue a FINISHED run to a larger max_steps
# An extension exists to run a finished schedule on to where the validation loss stops
# falling. It raises max_steps and follows a COMPLETED segment, which are the two things
# the rules above refuse, so it is gated on the only thing that makes it safe: a constant
# learning rate, which has no decay length to recompute. The gate also checks that
# main.py reads lr_scheduler_type at all -- a config file is exec'd, so declaring the key
# proves nothing about whether the trainer honours it. Downwards is still refused, and so
# is any previous state other than COMPLETED. With EXTEND unset every path behaves
# exactly as before. This is workflows/bert-grid.sbatch's gate, which the compounds
# extensions of 2026-10 ran under, with its wording kept so the two files stay readable
# against each other.
set -eo pipefail
: "${CONFIG:?CONFIG=path to the config to run, e.g. molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_small_lr3e4.py}"
: "${SEGMENT:?SEGMENT=1 for the first, 2,3,... for each continuation}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="${CONFIG}"

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
# Inputs and outputs are separate trees. core/output_guard.py refuses a model_path
# that lands inside LEARNING_SOURCE_DIR, and mol_nl used to point both at the same
# place: LEARNING_SOURCE_DIR named learning_source_20260803_molnl_train, which holds
# the outputs and two symlinks to the corpus. So the default here was a path the
# guard is guaranteed to reject, and the failure came two minutes into a 4-GPU job.
#
# LEARNING_SOURCE_DIR now names the corpus itself (learning_source), which is what
# the symlinks pointed at, so the data read is the same bytes. MOLNL_ARTIFACT_ROOT
# names where mol_nl's own artifacts already live, and nothing moves.
ARTIFACT_ROOT="${MOLNL_ARTIFACT_ROOT:-${PYROOT}/learning_source_20260803_molnl_train}"
RUNS_ROOT="${RUNS_ROOT:-${ARTIFACT_ROOT}/molecule_nat_lang}"
# Named after the config, not the size: the grid runs three learning rates per size, and
# a directory named for the size alone would have all three arms writing to one place --
# where the first would be resumed by the second instead of it starting.
CFG_NAME="$(basename "${CFG}" .py)"
OUT="${RUNS_ROOT}/bert-output-shuffled/${CFG_NAME}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${OUT}" "${HF_DATASETS_CACHE}" workflows/slurm-logs
[ -f "${CFG}" ] || { echo "no config at ${CFG} (relative to ${REPO})" >&2; exit 1; }

# The grid configs import their base rather than restating it, so the values cannot be
# read out of the file with grep. Run it the way main.py will and read what it resolves to.
_cfg_dump=$("${PYTHON}" - "${CFG}" <<'PYEOF' 2>/dev/null
import runpy, sys
g = runpy.run_path(sys.argv[1])
keys = ("max_steps", "warmup_steps", "learning_rate", "seed", "batch_size",
        "gradient_accumulation_steps", "per_device_eval_batch_size", "expected_global_batch",
        "log_interval", "save_steps", "max_length", "document_masking", "save_on_improve",
        "early_stopping", "early_stopping_patience", "lr_scheduler_type", "dataset_dir")
for k in keys:
    print(f"{k}={g.get(k)}")
PYEOF
) || { echo "could not read ${CFG}" >&2; exit 1; }
_cfg_steps=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^max_steps=//p')
[ -n "${_cfg_steps}" ] || { echo "no max_steps in ${CFG}" >&2; exit 1; }

# Ask the guard itself rather than re-deriving its rule here, so the two cannot drift
# and so nanoGPT -- which does not call it anywhere -- is covered too. Seconds, and
# before the allocation is put to work.
"${PYTHON}" -c 'import sys; from molcrawl.core.output_guard import assert_output_dir; assert_output_dir(sys.argv[1], what=sys.argv[2])' "${OUT}" "model_path" || exit 1

# --- extension gate ---
# Two of the guards below exist to stop a continuation from becoming a different run:
# max_steps must match the checkpoint's, and the previous segment must have ended
# TIMEOUT. An extension -- continuing a finished run to a larger max_steps to see
# whether the loss flattens -- trips both on purpose, so it has to say so explicitly
# and prove the one thing that makes it safe.
#
# EXTEND=1 opts in. Everything below is refused unless all of it holds, and with
# EXTEND unset not one byte of the original behaviour changes.
_extend=0
if [ "${EXTEND:-0}" = "1" ]; then
  _cfg_sched=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^lr_scheduler_type=//p')
  case "${_cfg_sched}" in
    constant|constant_with_warmup) ;;
    *) echo "REFUSING: EXTEND=1 but ${CFG} has lr_scheduler_type=${_cfg_sched:-None}." \
            "HF decays the rate over max_steps with no separate decay length, so" \
            "extending under a decaying schedule recomputes the whole curve and" \
            "continues a different run. Set lr_scheduler_type to constant or" \
            "constant_with_warmup." >&2
       exit 1;;
  esac
  # The config can declare the key whether or not the trainer reads it: a config file
  # is exec'd, so configurator.py never sees an unknown name and sets an unused global
  # instead. Confirm main.py honours it, because a silent fallback to "linear" here is
  # exactly the corruption this gate is meant to prevent.
  _main="molcrawl/models/bert/main.py"
  grep -qE '^[[:space:]]*lr_scheduler_type[[:space:]]*=' "${_main}" || {
    echo "REFUSING: ${_main} does not declare lr_scheduler_type, so it would ignore" \
         "the config's ${_cfg_sched} and decay the rate anyway. The extension needs" \
         "the schedule passthrough in main.py first." >&2
    exit 1; }
  grep -q 'globals().get("lr_scheduler_type")' "${_main}" || {
    echo "REFUSING: ${_main} declares lr_scheduler_type but does not pass it to" \
         "TrainingArguments, so the config's ${_cfg_sched} would be ignored." >&2
    exit 1; }
  # HF restores the scheduler from the checkpoint, and LambdaLR.load_state_dict does
  # self.__dict__.update(state_dict): that brings back base_lrs and last_epoch, not
  # just the step counter. get_lr then returns base_lr * lambda(last_epoch). Read off
  # a real checkpoint-15000/scheduler.pt: base_lrs [0.001, 0.001], last_epoch 15000.
  # So an extension that inherits it runs at the ORIGINAL peak with the warmup already
  # behind it -- the config's learning_rate and warmup_steps are both ignored, and the
  # rate jumps to ten times the intended one. Refuse rather than let that happen
  # silently. Extend from a copy of the checkpoint with scheduler.pt removed; the
  # optimizer moments in optimizer.pt are kept, which is what a continuation needs.
  _n_ckpt_pre=$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)
  if [ "${_n_ckpt_pre}" -ne 0 ]; then
    _last_pre=$(find "${OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
                | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -f "${OUT}/checkpoint-${_last_pre}/scheduler.pt" ]; then
      # Two different resumes reach this line and only one of them is dangerous.
      #
      # A NEW extension resumes from a checkpoint the previous, shorter schedule wrote.
      # Its scheduler.pt carries that run's peak, so restoring it runs this config at
      # ten times the intended rate -- the case this guard exists for.
      #
      # A LATER SEGMENT of an extension already under way resumes from a checkpoint
      # THIS config wrote. Its scheduler.pt carries this config's own constant rate, so
      # restoring it is not merely safe but correct, and refusing it would mean every
      # segment boundary needed the file removed by hand -- which is how four segments
      # of the compounds 45,000-step extension came back FAILED on 2026-10-29/30.
      #
      # Tell them apart by what the checkpoint was trained under, not by the file's
      # presence: its own max_steps, and the rate it was last logged at. The danger is
      # always a rate ABOVE this config's, so the rate check is one-sided and a
      # checkpoint taken inside the warmup ramp still passes.
      _cfg_lr=$(printf '%s\n' "${_cfg_dump}" | sed -n 's/^learning_rate=//p')
      read -r _ck_steps _ck_lr <<EOF
$("${PYTHON}" -c 'import json, sys
s = json.load(open(sys.argv[1]))
rates = [h["learning_rate"] for h in s.get("log_history", []) if "learning_rate" in h]
print(s.get("max_steps"), repr(rates[-1]) if rates else "none")' \
        "${OUT}/checkpoint-${_last_pre}/trainer_state.json" 2>/dev/null)
EOF
      _same_schedule=0
      if [ "${_ck_steps}" = "${_cfg_steps}" ] && [ -n "${_ck_lr}" ]; then
        "${PYTHON}" -c 'import sys
ck, cfg = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if ck <= cfg * (1.0 + 1e-9) else 1)' "${_ck_lr}" "${_cfg_lr}" 2>/dev/null \
          && _same_schedule=1
      fi
      # A STAGED REWRITE is the third case, and it is the one that keeps the moments.
      #
      # "Remove scheduler.pt, keep optimizer.pt" does not keep the moments.
      # transformers 4.45.1 guards BOTH loads with one condition on scheduler.pt's
      # existence (Trainer._load_optimizer_and_scheduler), so removing the file leaves
      # Adam unloaded as well -- silently. The molnl extensions of 2026-11-16 ran that
      # way: their checkpoints carry an Adam step counted from the resume (100 at step
      # 24,100) against 24,000 in the checkpoint they resumed from. Another session
      # measured the three hand-overs on 2026-11-18 (job 363888): keep gives the
      # moments and the original peak, remove gives the asked-for rate and no moments,
      # a re-pointed file gives both.
      #
      # So prepare the copy instead of amputating it: base_lrs to this config's
      # learning_rate and last_epoch to 0, which
      # scripts/extension_scheduler_rewrite.py writes. Restoring that applies this
      # config's own schedule from its first step, warmup included.
      #
      # The check is on the file's contents, so an unprepared copy still falls through
      # to the refusal below.
      if [ "${_same_schedule}" != "1" ]; then
        _rewritten=$("${PYTHON}" -c 'import sys, torch
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
lr = float(sys.argv[2])
base = st.get("base_lrs") or []
ok = (st.get("last_epoch") == 0 and base
      and all(abs(float(b) - lr) <= abs(lr) * 1e-9 for b in base))
print("1" if ok else "0")' \
          "${OUT}/checkpoint-${_last_pre}/scheduler.pt" "${_cfg_lr}" 2>/dev/null)
        if [ "${_rewritten}" = "1" ]; then
          echo "extension: checkpoint-${_last_pre}/scheduler.pt was re-pointed at this" \
               "config (base_lrs ${_cfg_lr}, last_epoch 0), so restoring it applies this" \
               "schedule from its first step and optimizer.pt loads with it"
          _same_schedule=1
        fi
      fi
      if [ "${_same_schedule}" = "1" ]; then
        echo "extension: checkpoint-${_last_pre} was written by this same schedule" \
             "(its max_steps ${_ck_steps} = the config's, last rate ${_ck_lr} <=" \
             "${_cfg_lr}), so its scheduler.pt holds this config's own rate and is" \
             "kept -- a segment continuation, not a new extension"
      else
        echo "REFUSING: EXTEND=1 but ${OUT}/checkpoint-${_last_pre}/scheduler.pt exists" \
             "and that checkpoint was not written by this schedule (its max_steps" \
             "${_ck_steps:-unreadable} against the config's ${_cfg_steps}, its last rate" \
             "${_ck_lr:-unreadable} against the config's ${_cfg_lr})." \
             "HF restores base_lrs and last_epoch from it, so ${CFG}'s learning_rate and" \
             "warmup_steps would both be ignored and the run would continue at the" \
             "original peak with no warmup. Re-point scheduler.pt at this config" \
             "(base_lrs = its learning_rate, last_epoch = 0) with" \
             "scripts/extension_scheduler_rewrite.py, or equivalent. Removing the file" \
             "also fixes the rate, but leaves the Adam moments unloaded: transformers" \
             "guards both loads with one condition on that file." >&2
        exit 1
      fi
    fi
  fi
  _extend=1
  echo "extension: EXTEND=1, schedule ${_cfg_sched}, target max_steps ${_cfg_steps}"
fi
# --- end extension gate ---

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
       "time limit re-running step 0. Check that CONFIG=${CFG}, LEARNING_SOURCE_DIR" \
       "and RUNS_ROOT match the earlier segment -- OUT is derived from all three." >&2
  exit 1
fi

# The schedule must be the same run continued, not a new one written over it.
if [ "${SEGMENT}" != "1" ]; then
  _state="${OUT}/checkpoint-${_last}/trainer_state.json"
  [ -f "${_state}" ] || {
    echo "REFUSING: ${_state} is missing, so the schedule this checkpoint was trained" \
         "under cannot be read and the config's max_steps=${_cfg_steps} cannot be checked" \
         "against it." >&2
    exit 1; }
  read -r _prev_steps _prev_done <<EOF
$("${PYTHON}" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["max_steps"], int(s["global_step"]))' "${_state}")
EOF
  if [ "${_prev_steps}" != "${_cfg_steps}" ]; then
    # An extension raises max_steps on purpose. Only upwards: a smaller value would
    # still compress the schedule, and under a constant rate it would also mean the
    # run is already past its own target.
    if [ "${_extend}" = "1" ] && [ "${_cfg_steps}" -gt "${_prev_steps}" ]; then
      echo "extension: max_steps ${_prev_steps} -> ${_cfg_steps} (constant rate, no decay to recompute)"
    else
      echo "REFUSING: the checkpoint at step ${_last} was trained with max_steps=${_prev_steps}," \
           "and ${CFG} says max_steps=${_cfg_steps}. HF decays the learning rate over" \
           "max_steps with no separate decay length, so resuming under a different value" \
           "compresses or stretches the schedule and continues a different run into the" \
           "same directory. Run a config whose max_steps is ${_prev_steps}, or start a new" \
           "run in another directory." \
           "$([ "${_extend}" = "1" ] && echo "EXTEND=1 raises max_steps, it does not lower it.")" >&2
      exit 1
    fi
  fi
  # A chained segment that turns out not to be needed exits 0 rather than failing.
  # Two jobs per run are queued up front because nobody knows whether 24,000 steps fit
  # in one time limit; when the first segment finishes the schedule, the second has
  # nothing to do and that is the expected outcome, not an error to investigate.
  # Anything else that stopped the run short still goes through the guard below.
  if [ "${_prev_done}" -ge "${_cfg_steps}" ]; then
    echo "nothing to do: the run reached step ${_prev_done} of ${_cfg_steps} in an earlier" \
         "segment. Exiting without training."
    exit 0
  fi
  if [ "${_extend}" != "1" ]; then
    echo "schedule check: max_steps ${_prev_steps} matches the config, ${_prev_done}/${_cfg_steps} steps done"
  else
    echo "schedule check: ${_prev_done}/${_cfg_steps} steps done, extending from ${_prev_steps}"
  fi
fi

# --- continuation guard ---
# A continuation follows only a segment that ran out of time. It is queued with a
# dependency (afterany / afternotok), and both fire on TIMEOUT and on a crash alike;
# none of the guards above can tell those apart either, because after a crash the
# checkpoint is still there and max_steps and the batch still match. So ask Slurm how the
# previous segment ended, and continue only from TIMEOUT. COMPLETED is refused too:
# a run that finished has nothing to continue.
#
# Every segment appends its job id to ${OUT}/segments.log, so the next one finds its
# predecessor without being told. PREV_JOB overrides that, and is required for a
# directory whose earlier segments predate the log.
_segments_log="${OUT}/segments.log"
_prev_segment_state() {  # $1 = job id. Waits briefly while accounting catches up.
  local s="" i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    s=$(sacct -X -n -j "$1" -o State%30 2>/dev/null | head -1 | awk '{print $1}' || true)
    case "${s}" in ""|RUNNING|PENDING|COMPLETING|REQUEUED|RESIZING|SUSPENDED) sleep 30;; *) break;; esac
  done
  echo "${s:-UNKNOWN}"
}
if [ "${SEGMENT}" != "1" ]; then
  _prev_seg=$(( SEGMENT - 1 ))
  _prev_job="${PREV_JOB:-}"
  if [ -z "${_prev_job}" ] && [ -f "${_segments_log}" ]; then
    _prev_job=$(awk -v s="${_prev_seg}" '$1==s {j=$2} END {print j}' "${_segments_log}" || true)
  fi
  if [ -z "${_prev_job}" ]; then
    echo "REFUSING: cannot tell which job ran segment ${_prev_seg} -- ${_segments_log} has no" \
         "entry for it. Pass PREV_JOB=<job id> so its end state can be checked." >&2
    exit 1
  fi
  _prev_state=$(_prev_segment_state "${_prev_job}")
  # An extension follows a run that FINISHED, which is the one state this guard was
  # written to refuse. COMPLETED is accepted under EXTEND=1 and nothing else is.
  if [ "${_extend}" = "1" ] && [ "${_prev_state}" = "COMPLETED" ]; then
    # COMPLETED covers two endings and only one of them may be extended. A schedule that
    # ran to its end has more steps to give; a run that early-stopped exited 0 at the
    # step where the validation loss stopped falling, and resuming it would walk straight
    # past the answer the stop was there to give. Tell them apart by the checkpoint's own
    # numbers, not by the exit status.
    if [ "${_prev_done}" -lt "${_prev_steps}" ]; then
      echo "REFUSING: segment ${_prev_seg} (job ${_prev_job}) ended COMPLETED at step" \
           "${_prev_done} of its own max_steps ${_prev_steps}, which is where early" \
           "stopping leaves a run. An extension continues a schedule that finished, not" \
           "one that stopped because the metric stopped improving. Read its log, and" \
           "raise the patience in the config if the stop was premature." >&2
      exit 1
    fi
    echo "previous segment: ${_prev_seg} (job ${_prev_job}) ended COMPLETED at its own" \
         "max_steps ${_prev_steps}, extending it"
  elif [ "${_prev_state}" != "TIMEOUT" ]; then
    if [ "${_prev_state}" = "COMPLETED" ]; then
      _why="it finished the run, so there is nothing to continue; EXTEND=1 is how a finished run is extended"
    else
      _why="a segment that did not run out of time may have stopped for a reason a resume would repeat"
    fi
    echo "REFUSING: segment ${_prev_seg} (job ${_prev_job}) ended ${_prev_state}, not TIMEOUT:" \
         "${_why}. Read its log before continuing." >&2
    exit 1
  else
    echo "previous segment: ${_prev_seg} (job ${_prev_job}) ended TIMEOUT"
  fi
fi
echo "${SEGMENT} ${SLURM_JOB_ID:-local} $(date -Is)" >> "${_segments_log}"
# --- end continuation guard ---

echo "=== mol_nl ${CFG_NAME} (shuffled) ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified)"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}"
echo "4-GPU launch required: under HF the effective batch is per_device x grad_accum x world_size"
echo "segment=${SEGMENT}  ${_n_ckpt} checkpoint(s) present  $([ "${SEGMENT}" = 1 ] && echo 'starting from scratch' || echo "resuming from step ${_last}")"
echo "--- config values as resolved (not grepped: the grid configs import their base) ---"
printf '%s\n' "${_cfg_dump}" | sed 's/^/    /'
# What this run's eval_loss_mask is read against. It is measured on the VALID split,
# which is the split the metric is computed on: unigram 4.6449 and the blended
# degenerate level 4.1780 (report/mlm-baselines-20261102/full/molnl.json, 1,631 rows,
# mask rate 0.2, fit on 200,000 train rows).
#
# The four numbers this line used to print -- unigram 3.8638 / degenerate 3.4774 /
# left 2.2533 / both 1.2804, job 22503 -- are the same measurement on the TRAIN split.
# The floor differs by split, so printing them next to a validation metric invited the
# comparison that 2026-11-10 corrected everywhere else.
echo "baselines (mol_nl, valid split, report/mlm-baselines-20261102): unigram 4.6449 / blended degenerate 4.1780   [train split, job 22503: unigram 3.8638 / degenerate 3.4774]"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/bert/main.py "${CFG}" \
  --model_path="${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
