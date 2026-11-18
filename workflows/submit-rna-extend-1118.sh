#!/bin/bash
# Queue the rna 2026-11-18 extension: eight points from 120,960 steps to 241,920.
#
# rna-order-2026-11-18b §1-§2; rna-judgment-2026-11-18b 確認事項 1-5 and §7.
# scripts/prepare_rna_extension.py must have run first: every point resumes from a
# COPY in its own output directory, and the BERT copies need their scheduler.pt
# re-pointed at the extension's rate. This script refuses a point whose copy is
# missing rather than starting a fresh run under a name that claims otherwise.
#
# Usage:
#   LEARNING_SOURCE_DIR=<input root> BERT_ROOT=<runs root> GPT2_ROOT=<runs root>/gpt2-grid \
#     SLURM_ACCOUNT=<account> ./workflows/submit-rna-extend-1118.sh
#
#   SIDE=bert | gpt2 | both   which half to queue (default both)
#   BERT_LINKS, GPT2_LINKS    chained segments per point (default 8 each)
#   DRY_RUN=1                 print the plan, submit nothing
#   PLAIN_SBATCH=1            bypass sbatch-watched.sh (not the default: a run of
#                             this length needs a watcher outside the session)
#   SUBMIT_RECORD=<file>      where sbatch-watched.sh appends the job ids. Defaults
#                             to <runs root>/_submissions/rna-extend-1118.tsv --
#                             beside the outputs rather than in this worktree, which
#                             is disposable, and outside the session, which is the
#                             point: four compounds runs once died on a Friday and
#                             the only record of what had been submitted was in the
#                             session that submitted them.
#   EXPECT_COMMIT=<sha>       pin the checkout each link runs. Optional, and a
#                             rewritten history kills every queued link, so pass
#                             the full hash of a commit that will not move.
#
# Requeuing a chain that was withdrawn (Slurm keeps the batch script it was given
# at submission, so a fix to bert-grid.sbatch only reaches links submitted after
# it):
#   BERT_SEGMENT_START=<n>    the segment number link 1 carries (default 2)
#   BERT_PREV="tag=jobid;..." override the predecessor link 1 is told to follow,
#                             per arm, by the tag used in the job names
#   AFTER_CANCELLED=1         link 1 also carries RESUME_AFTER_CANCELLED for that
#                             predecessor. bert-grid.sbatch continues from a
#                             CANCELLED segment only when named this way, and it
#                             authorises exactly one predecessor, so a later link
#                             of the same chain cannot be waved through by it.
#
# Eight links at the 4-day limit covers xl BERT, which held 120,960 steps in
# 30.96 days over nine segments; the smaller sizes finish sooner.
#
# Links after the first are chained afternotok, not afterany, so a link that
# COMPLETES ends the chain. Both ways a link can complete mean "do not continue":
# it reached max_steps, or the collapse detector set should_training_stop, which
# also exits 0 and so lands as COMPLETED. Under afterany the next link would
# instead resume the collapsed model, run its own patience out and stop again --
# bounded, but about 500 steps per remaining link, which on xl is three hours
# each. A TIMEOUT or a crash is "notok" and still releases the successor, where
# bert-grid.sbatch checks the end state and refuses the ones a resume would
# repeat.
#
# The first BERT link has no predecessor in its own segments.log, because the copy
# is fresh, so PREV_JOB carries the nine-epoch run's last job and EXTEND=1 is what
# lets a COMPLETED predecessor be continued.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
SIDE="${SIDE:-both}"
BERT_LINKS="${BERT_LINKS:-8}"
GPT2_LINKS="${GPT2_LINKS:-8}"
CFG_DIR=molcrawl/tasks/pretrain/configs/rna

# tag : config : nine-epoch run's last job (its state is checked by the guard)
BERT_POINTS="
small-lr3e4:bert_small_lr3e4_ext242k:122417
medium-lr3e4:bert_medium_lr3e4_ext242k:122429
large-lr1p7e4:bert_large_lr1p7e4_ext242k:199879
xl-lr2p5e5:bert_xl_lr2p5e5_ext242k:199909
"
GPT2_POINTS="
small-lr1p2e3:gpt2_small_lr1p2e3_9ep_ext242k
medium-lr1p2e3:gpt2_medium_lr1p2e3_9ep_ext242k
large-lr6e4:gpt2_large_lr6e4_9ep_ext242k
xl-lr6e4:gpt2_xl_lr6e4_9ep_ext242k
"

_submit() {  # $1 = watcher name, rest = sbatch args and the script
    local name="$1"; shift
    if [ -n "${DRY_RUN:-}" ]; then
        echo "<dry-run:${name}>"
    elif [ -n "${PLAIN_SBATCH:-}" ]; then
        sbatch --parsable "$@"
    else
        workflows/sbatch-watched.sh "${name}" -- "$@"
    fi
}

# The config names its own output directory, so ask it rather than rebuilding the path.
_out_of() {  # $1 = config module name, $2 = the key to read, $3 = MODEL_OUTPUT_ROOT
    MODEL_OUTPUT_ROOT="$3" PYTHONPATH="$PWD" "${PYTHON}" -c \
        "import importlib,sys; m=importlib.import_module('molcrawl.tasks.pretrain.configs.rna.'+sys.argv[1]); print(getattr(m,sys.argv[2]))" \
        "$1" "$2"
}

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
  PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
[ -x "${PYTHON}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

# sbatch-watched.sh refuses without this, so default it rather than fail eight chains
# in at the first link.
if [ -z "${SUBMIT_RECORD:-}" ]; then
  _rec_root="${BERT_ROOT:-${GPT2_ROOT:-}}"
  [ -n "${_rec_root}" ] || { echo "set BERT_ROOT or GPT2_ROOT (or SUBMIT_RECORD)" >&2; exit 1; }
  SUBMIT_RECORD="${_rec_root}/_submissions/rna-extend-1118.tsv"
fi
export SUBMIT_RECORD

total=0
echo "=== rna extension to 241,920 steps ==="
echo "  submission record: ${SUBMIT_RECORD}"
echo "  side=${SIDE}  links: bert ${BERT_LINKS}, gpt2 ${GPT2_LINKS}"
echo "  submit via: $([ -n "${DRY_RUN:-}" ] && echo 'dry run' || { [ -n "${PLAIN_SBATCH:-}" ] && echo 'plain sbatch' || echo 'sbatch-watched.sh'; })"

if [ "${SIDE}" = "bert" ] || [ "${SIDE}" = "both" ]; then
  : "${BERT_ROOT:?set BERT_ROOT to the runs root the BERT configs resolve against}"
  echo
  echo "--- BERT, under ${BERT_ROOT}"
  for entry in ${BERT_POINTS}; do
    IFS=: read -r tag cfgname prev <<< "${entry}"
    cfg="${CFG_DIR}/${cfgname}.py"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    out=$(_out_of "${cfgname}" model_path "${BERT_ROOT}")
    [ -d "${out}/checkpoint-120960" ] || {
        echo "refusing ${tag}: ${out}/checkpoint-120960 is not staged --" \
             "run scripts/prepare_rna_extension.py --scheduler rewrite first" >&2
        exit 1; }
    if [ -n "${BERT_PREV:-}" ]; then
      _override=$(printf '%s' "${BERT_PREV}" | tr ';' '\n' \
                  | awk -F= -v t="${tag}" '$1==t {print $2}')
      [ -n "${_override}" ] && prev="${_override}"
    fi
    echo "  ${tag}: ${out}"
    echo "    follows job ${prev}${AFTER_CANCELLED:+ (as a cancelled predecessor)}"
    dep=""
    for i in $(seq 1 "${BERT_LINKS}"); do
      seg=$(( i + ${BERT_SEGMENT_START:-2} - 1 ))
      exp="ALL,MODALITY=rna,LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
      exp="${exp},MODEL_OUTPUT_ROOT=${BERT_ROOT},CONFIG=${cfg},SEGMENT=${seg},EXTEND=1"
      [ -n "${EXPECT_COMMIT:-}" ] && exp="${exp},EXPECT_COMMIT=${EXPECT_COMMIT}"
      # Only the first link needs to be told what it follows; later ones read
      # segments.log in the output directory.
      if [ "${i}" = "1" ]; then
        exp="${exp},PREV_JOB=${prev}"
        [ -n "${AFTER_CANCELLED:-}" ] && exp="${exp},RESUME_AFTER_CANCELLED=${prev}"
      fi
      args=(--job-name="rna-bert-${tag}-ext-S${seg}" --account="${SLURM_ACCOUNT}" --export="${exp}")
      [ -n "${dep}" ] && args+=(--dependency=afternotok:"${dep}")
      jid=$(_submit "bert-ext-${tag}-S${seg}" "${args[@]}" workflows/bert-grid.sbatch)
      echo "    link ${i}/${BERT_LINKS}: segment ${seg}, job ${jid}${dep:+ (after ${dep})}"
      dep="${jid}"; total=$(( total + 1 ))
    done
  done
fi

if [ "${SIDE}" = "gpt2" ] || [ "${SIDE}" = "both" ]; then
  : "${GPT2_ROOT:?set GPT2_ROOT to the runs root the GPT-2 configs resolve against}"
  echo
  echo "--- GPT-2, under ${GPT2_ROOT}"
  for entry in ${GPT2_POINTS}; do
    IFS=: read -r tag cfgname <<< "${entry}"
    cfg="${CFG_DIR}/${cfgname}.py"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    out=$(_out_of "${cfgname}" out_dir "${GPT2_ROOT}")
    [ -f "${out}/ckpt.pt" ] || {
        echo "refusing ${tag}: ${out}/ckpt.pt is not staged -- run" \
             "scripts/prepare_rna_extension.py --framework gpt2 first." \
             "init_from=resume with an empty out_dir would start from scratch." >&2
        exit 1; }
    echo "  ${tag}: ${out}"
    dep=""
    for i in $(seq 1 "${GPT2_LINKS}"); do
      exp="ALL,LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR},MODEL_OUTPUT_ROOT=${GPT2_ROOT}"
      exp="${exp},GRID_CONFIG=${cfg}"
      [ -n "${EXPECT_COMMIT:-}" ] && exp="${exp},EXPECT_COMMIT=${EXPECT_COMMIT}"
      args=(--job-name="rna-gpt2-${tag}-ext-C${i}" --account="${SLURM_ACCOUNT}" --export="${exp}")
      # GPT-2 has no collapse detector, so a COMPLETED link here means max_iters was
      # reached. Ending the chain on it is right for the same reason.
      [ -n "${dep}" ] && args+=(--dependency=afternotok:"${dep}")
      jid=$(_submit "gpt2-ext-${tag}-C${i}" "${args[@]}" workflows/rna-gpt2-grid.sbatch)
      echo "    link ${i}/${GPT2_LINKS}: job ${jid}${dep:+ (after ${dep})}"
      dep="${jid}"; total=$(( total + 1 ))
    done
  done
fi

echo
echo "=== ${total} link(s) ${DRY_RUN:+would be }queued ==="
