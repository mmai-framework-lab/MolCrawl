#!/bin/bash
#SBATCH --job-name=mc-signal-probe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:04:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Which process receives Slurm's --signal, when the batch script starts torchrun directly
# and does not use srun -- the form every trainer here is launched in.
#
# The question decides whether a run can save on its way out of the time limit instead of
# losing the steps since its last checkpoint. Measured 2026-09-11, Slurm 25.11.5:
#
#   --signal=USR1@150    VARIANT=plain          no process received it
#   --signal=USR1@150    VARIANT=plain-ignored  no process received it
#   --signal=B:USR1@150  VARIANT=forward        the batch shell received it, and the
#                                               workers received the forwarded signal
#
# man sbatch: without "B:" the signal goes to all job steps "but not the batch shell
# itself" -- and a batch script that never calls srun has no job steps, so nothing is
# signalled. Only "B:" is usable here, and the shell has to forward it: torchrun's agent
# installs handlers for TERM, INT, HUP and QUIT only, so a USR1 delivered to torchrun
# itself would kill it by default action. Forwarding with pkill -P reaches the workers,
# which are torchrun's direct children, without touching the agent.
#
# The signal may arrive up to 60 s earlier than asked (man sbatch); it did here.
#
# Submit from the repository root, pairing --signal with VARIANT:
#   sbatch --signal=B:USR1@150 --export=ALL,VARIANT=forward       workflows/signal-delivery-probe.sbatch
#   sbatch --signal=USR1@150   --export=ALL,VARIANT=plain         workflows/signal-delivery-probe.sbatch
#   sbatch --signal=USR1@150   --export=ALL,VARIANT=plain-ignored workflows/signal-delivery-probe.sbatch
#
# No GPU, no training: two workers that do nothing but report the signals they receive.
set -eo pipefail
VARIANT="${VARIANT:-forward}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -x "${REPO}/miniconda/bin/torchrun" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
TORCHRUN="${TORCHRUN:-${PYROOT}/miniconda/bin/torchrun}"
[ -x "${TORCHRUN}" ] || {
  echo "no torchrun at ${TORCHRUN}; export TORCHRUN=<main checkout>/miniconda/bin/torchrun" >&2; exit 1; }

export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
cat > "${WORK}/worker.py" <<'PY'
"""Stands in for a training worker: reports the signals it receives, and nothing else."""
import os
import signal
import time

RANK = os.environ.get("RANK", "?")


def report(sig, _frame):
    print(f"[{time.strftime('%H:%M:%S')}] worker rank={RANK} pid={os.getpid()} "
          f"got {signal.Signals(sig).name}", flush=True)


for s in (signal.SIGUSR1, signal.SIGTERM):
    signal.signal(s, report)
print(f"[{time.strftime('%H:%M:%S')}] worker rank={RANK} pid={os.getpid()} "
      f"ppid={os.getppid()} up", flush=True)
while True:
    time.sleep(5)
PY

echo "=== signal delivery probe: VARIANT=${VARIANT} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "batch shell pid=$$"
case "${VARIANT}" in
  plain)
    "${TORCHRUN}" --standalone --nproc_per_node=2 "${WORK}/worker.py" || true
    ;;
  plain-ignored)
    trap '' USR1
    "${TORCHRUN}" --standalone --nproc_per_node=2 "${WORK}/worker.py" || true
    ;;
  forward)
    "${TORCHRUN}" --standalone --nproc_per_node=2 "${WORK}/worker.py" & TR=$!
    trap 'echo "[$(date +%H:%M:%S)] batch shell got USR1 -> forwarding to the workers"; pkill -USR1 -P "${TR}" || true' USR1
    while kill -0 "${TR}" 2>/dev/null; do wait "${TR}" || true; done
    ;;
  *)
    echo "VARIANT must be one of: plain | plain-ignored | forward" >&2; exit 1
    ;;
esac
echo "=== done @ $(date -Is) ==="
