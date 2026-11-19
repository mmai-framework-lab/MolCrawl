#!/bin/bash
#SBATCH --job-name=mc-molnl-extend-prepare
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Prepare the seven mol_nl extension runs of the 2026-11-16 order: one directory per run,
# holding a copy of the finished run's final checkpoint.
#
#   sbatch --export=ALL[,DRY_RUN=1][,SUFFIX=_extm][,SCHEDULER=rewrite|remove]\
#          [,ONLY_ARCH=bert|gpt2] workflows/molnl-extend-prepare.sbatch
#
# SUFFIX names the extension directories, so a second extension of the same arms stages
# beside the first instead of over it. The extension config has to carry the same suffix:
# the launcher derives the output directory from the config name, and the rate written
# into scheduler.pt is read from that config rather than passed here, so the two cannot
# disagree.
#
# --gpus=0 asks for no accelerator. It does not keep the job off one: the gpu partition
# is the default and bills by GPU, and a job submitted here was allocated gres/gpu=1 all
# the same (job 305167). Nothing below uses a GPU; this is file copying.
#
# Why a copy rather than extending in place. Both trainers keep the best N checkpoints by
# score and prune the rest, so an extension writing into the finished run's directory
# would delete the checkpoints the reported numbers refer to -- the adopted ones of
# 2026-11-12 among them. Separate directories also mean the original run stays answerable
# after the extension changes the answer.
#
# What is copied, and what is deliberately not:
#
#   BERT   checkpoint-24000 with scheduler.pt RE-POINTED at the extension's schedule --
#          base_lrs to its learning_rate, last_epoch to 0 -- which
#          scripts/extension_scheduler_rewrite.py writes and the launcher's gate checks
#          by content. HF's LambdaLR.load_state_dict does
#          self.__dict__.update(state_dict), so an untouched file would run the
#          extension at the ORIGINAL peak with the warmup already behind it.
#
#          SCHEDULER=remove deletes the file instead. That also fixes the rate, and it is
#          what the 2026-11-16 extensions did -- at the cost of the Adam moments, because
#          transformers 4.45.1 guards the optimizer load and the scheduler load with one
#          condition on scheduler.pt's existence. Those runs' checkpoints carry an Adam
#          step counted from the resume (100 at step 24,100) against 24,000 in the
#          checkpoint they resumed from. rewrite is the default for that reason.
#
#   GPT-2  checkpoint-1500 and the run's logging_*.csv. The CSV is not decoration:
#          train.py reads the evaluations back from it (eval_history) and without it the
#          copied checkpoint ranks as unscored and is prunable. run_manifest.json is NOT
#          copied -- the extension writes its own and the name is fixed.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
ARTIFACT_ROOT="${MOLNL_ARTIFACT_ROOT:-${PYROOT}/learning_source_20260803_molnl_train}"
RUNS_ROOT="${RUNS_ROOT:-${ARTIFACT_ROOT}/molecule_nat_lang}"
BERT_ROOT="${RUNS_ROOT}/bert-output-shuffled"
GPT2_ROOT="${RUNS_ROOT}/gpt2-output-grid"
DRY_RUN="${DRY_RUN:-0}"
SUFFIX="${SUFFIX:-_ext}"
SCHEDULER="${SCHEDULER:-rewrite}"
ONLY_ARCH="${ONLY_ARCH:-}"
case "${SCHEDULER}" in rewrite|remove) ;; *) echo "SCHEDULER must be rewrite or remove" >&2; exit 1;; esac
CFG_DIR="molcrawl/tasks/pretrain/configs/molecule_nat_lang"
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-${PYROOT}/learning_source}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
# Relative paths below -- the config tree and scripts/ -- are the checkout's, not
# whatever directory the job happened to start in.
cd "${REPO}"

echo "=== mol_nl extension prepare ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "runs root: ${RUNS_ROOT}"
echo "dry run  : ${DRY_RUN}"
echo "suffix   : ${SUFFIX}"
echo "scheduler: ${SCHEDULER}"
echo "only arch: ${ONLY_ARCH:-(both)}"

# <source run> <final step> <arch>
RUNS="
bert_small_24k_lr1e3 24000 bert
bert_medium_24k_lr3e4 24000 bert
bert_large_24k_lr1p7e4 24000 bert
gpt2_small_1500_lr1p2e3 1500 gpt2
gpt2_medium_1500_lr6e4 1500 gpt2
gpt2_large_1500_lr3e4 1500 gpt2
gpt2_xl_1500_lr3e4 1500 gpt2
"

_sizes() {  # every regular file under $1 as "<relative path> <bytes>", sorted
  (cd "$1" && find . -type f -printf '%P %s\n' | sort)
}

rc=0
printf '%s' "${RUNS}" | while read -r run step arch; do
  [ -n "${run}" ] || continue
  [ -z "${ONLY_ARCH}" ] || [ "${arch}" = "${ONLY_ARCH}" ] || continue
  case "${arch}" in
    bert) root="${BERT_ROOT}";;
    gpt2) root="${GPT2_ROOT}";;
    *) echo "unknown arch ${arch}" >&2; exit 1;;
  esac
  src="${root}/${run}"
  dst="${root}/${run}${SUFFIX}"
  echo "--- ${run} -> $(basename "${dst}")"
  [ -d "${src}/checkpoint-${step}" ] || { echo "REFUSING: no ${src}/checkpoint-${step}" >&2; exit 1; }
  if [ -n "$(find "${dst}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
    echo "    already holds a checkpoint; leaving it alone (this script never overwrites" \
         "a prepared or running extension)"
    continue
  fi
  echo "    source: ${src}/checkpoint-${step}  ($(du -sh "${src}/checkpoint-${step}" | cut -f1))"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "    dry run: would copy the checkpoint$([ "${arch}" = gpt2 ] && echo " and logging_*.csv")$([ "${arch}" = bert ] && echo " and remove scheduler.pt")"
    continue
  fi
  mkdir -p "${dst}"
  cp -r "${src}/checkpoint-${step}" "${dst}/checkpoint-${step}"
  if [ "${arch}" = "bert" ]; then
    if [ ! -f "${dst}/checkpoint-${step}/scheduler.pt" ]; then
      echo "    no scheduler.pt in the source checkpoint"
    elif [ "${SCHEDULER}" = "remove" ]; then
      rm "${dst}/checkpoint-${step}/scheduler.pt"
      echo "    removed scheduler.pt (fixes the rate; the Adam moments will NOT be restored)"
    else
      # The rate comes from the extension config, not from here, so the staged file and
      # the run cannot disagree about it.
      _ext_cfg="${CFG_DIR}/${run}${SUFFIX}.py"
      [ -f "${_ext_cfg}" ] || { echo "REFUSING: no ${_ext_cfg} to read the rate from" >&2; exit 1; }
      _ext_lr=$("${PYTHON}" -c 'import runpy, sys; print(runpy.run_path(sys.argv[1]).get("learning_rate"))' "${_ext_cfg}" 2>/dev/null)
      [ -n "${_ext_lr}" ] || { echo "REFUSING: could not read learning_rate from ${_ext_cfg}" >&2; exit 1; }
      echo "    re-pointing scheduler.pt at ${_ext_cfg} (learning_rate ${_ext_lr})"
      "${PYTHON}" scripts/extension_scheduler_rewrite.py \
        "${dst}/checkpoint-${step}/scheduler.pt" "${_ext_lr}" | sed 's/^/      /'
    fi
    for f in optimizer.pt trainer_state.json; do
      [ -f "${dst}/checkpoint-${step}/${f}" ] || { echo "REFUSING: the copy has no ${f}" >&2; exit 1; }
    done
  else
    cp "${src}"/logging_*.csv "${dst}/"
    echo "    copied $(ls "${dst}"/logging_*.csv | wc -l) logging CSV(s)"
    [ -f "${dst}/checkpoint-${step}/training_state.bin" ] || {
      echo "REFUSING: the copy has no training_state.bin, which is what train.py resumes from" >&2; exit 1; }
  fi
  # Every file the copy still carries must match the source byte count. cp exits non-zero
  # on failure, but a short write on a full filesystem is the case worth naming.
  diff <(_sizes "${src}/checkpoint-${step}" | grep -v '^scheduler.pt ') \
       <(_sizes "${dst}/checkpoint-${step}" | grep -v '^scheduler.pt ') \
       > /tmp/molnl-ext-${run}.diff 2>&1 \
    && echo "    verified: $(_sizes "${dst}/checkpoint-${step}" | wc -l) file(s), every size equal to the source" \
    || { echo "REFUSING: the copy differs from the source:" >&2; sed 's/^/      /' /tmp/molnl-ext-${run}.diff >&2; exit 1; }
  rm -f /tmp/molnl-ext-${run}.diff
  echo "    ready: ${dst}"
done || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
