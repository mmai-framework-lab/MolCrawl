#!/bin/bash
#SBATCH --job-name=mc-molnl-gpt2
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# mol_nl GPT-2 ladder on the shuffled corpus.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<dir> sbatch --export=ALL,SIZE=small,STAGE=s2,ITERS=3000 \
#     workflows/molnl-gpt2-ladder.sbatch
#
# The config is UNMODIFIED. Only operational values are passed here:
#   --init_from=scratch       the config defaults to "resume"; these are new runs
#   --log_interval=10 --eval_interval
#                             the config's 200 gives two points over a short run, so
#                             the val curve the launch decision asks to watch cannot
#                             be seen
#   --always_save_checkpoint=False --save_checkpoint_steps=100
#                             the config saves at every eval; at eval_interval=25
#                             that is ~1.5 GB written per eval. The best-val save is
#                             an independent OR branch in train.py, so it stays.
#
# Unlike BERT, nanoGPT's effective batch is GPU-count-independent: train.py divides
# the configured accumulation by the world size (train.py:277-278), so the product is
# the same on any GPU count. The 4 here is for throughput, not for the batch.
#
# Divergence watch: the genome stability sweep found the peak LR level itself decides
# divergence, and that "only a few steps at high LR" does not hold -- base/C/D all
# diverged at peak 6e-4, and only 3e-4 and 1e-4 were stable. mol_nl small runs at peak
# 6e-4, so its val curve is watched across the whole run. GPT-2 has no automatic
# collapse detector; that mechanism is BERT-only.
#
# STAGE keeps each pass in its own tree. The config's out_dir points at gpt2-output/,
# which still holds the four finished source-order runs the 2026-08-20 verdict says
# to keep.
set -eo pipefail
: "${SIZE:?SIZE=small|medium|large|xl}"
: "${STAGE:?STAGE=s1|s2|... -- keeps each pass in its own tree}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the learning_source root holding molecule_nat_lang/}"
[ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang" ] || {
  echo "LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR} has no molecule_nat_lang/ under it" >&2; exit 1; }
ITERS="${ITERS:-373}"
EVERY="${EVERY:-25}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_${SIZE}.py"

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

# Outputs must not land in a worktree: learning_source* is gitignored, so each
# checkout has its own, and a worktree is meant to be disposable.
RUNS_ROOT="${RUNS_ROOT:-${LEARNING_SOURCE_DIR}/molecule_nat_lang}"
OUT="${RUNS_ROOT}/gpt2-output-shuffled-${STAGE}/molecule_nat_lang-${SIZE}"

export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-${PYROOT}/assets/tokenizers/gpt2}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PYROOT}/tmp/hf-cache-molnl}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
cd "${REPO}"; mkdir -p "${HF_DATASETS_CACHE}" workflows/slurm-logs

echo "=== mol_nl gpt2-${SIZE} stage ${STAGE} ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config: ${CFG} (unmodified)"
echo "outputs: ${OUT}   (RUNS_ROOT=${RUNS_ROOT})"
echo "data : ${LEARNING_SOURCE_DIR}  (config resolves dataset_dir; see \"Loading dataset\" below)"
echo "iters=${ITERS} (max_iters = lr_decay_iters, length-normalized)  eval every ${EVERY}"
echo "--- config values under test ---"
grep -hE "^learning_rate|^min_lr|^max_iters|^lr_decay_iters|^warmup_iters|^batch_size|^gradient_accumulation_steps|^weight_decay|^seed" "${CFG}" | sed 's/^/    /'
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

rc=0
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  molcrawl/models/gpt2/train.py "${CFG}" \
  --init_from=scratch --out_dir="${OUT}" --tensorboard_dir="${OUT}" \
  --max_iters="${ITERS}" --lr_decay_iters="${ITERS}" \
  --log_interval=10 --eval_interval="${EVERY}" \
  --always_save_checkpoint=False --save_checkpoint_steps=100 || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
