#!/bin/bash
# slurm_submit.sh — wrapper that submits an arbitrary workflow script as a SLURM job.
#
# Usage:
#   LEARNING_SOURCE_DIR=<dir> workflows/slurm_submit.sh <workflow_script.sh> [sbatch options...]
#
# Examples:
#   LEARNING_SOURCE_DIR=learning_source_20260302 workflows/slurm_submit.sh workflows/03a-molecule_nat_lang-train-small.sh
#   LEARNING_SOURCE_DIR=learning_source_20260302 workflows/slurm_submit.sh workflows/03a-molecule_nat_lang-train-small.sh --partition=h200-long
#
# Environment variables:
#   LEARNING_SOURCE_DIR  (required) Learning source directory
#   SLURM_PARTITION      (optional) Partition name (default: h200)
#   SLURM_GPUS           (optional) Number of GPUs per node (default: 1)
#   SLURM_TIME           (optional) Maximum run time (default: 8:00:00)
#   SLURM_JOB_NAME       (optional) Job name (default: workflow script basename)

set -e

# --- Argument check ---
if [ -z "$1" ]; then
    echo "Usage: LEARNING_SOURCE_DIR=<dir> $0 <workflow_script.sh> [sbatch options...]"
    exit 1
fi

WORKFLOW_SCRIPT="$(realpath "$1")"
shift  # remaining args are forwarded to sbatch

if [ ! -f "$WORKFLOW_SCRIPT" ]; then
    echo "ERROR: Workflow script not found: $WORKFLOW_SCRIPT"
    exit 1
fi

# --- LEARNING_SOURCE_DIR check ---
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "ERROR: LEARNING_SOURCE_DIR is not set."
    echo "  Example: LEARNING_SOURCE_DIR=learning_source_20260302 $0 $WORKFLOW_SCRIPT"
    exit 1
fi

# --- SLURM parameter setup ---
PARTITION="${SLURM_PARTITION:-h200}"
GPUS="${SLURM_GPUS:-1}"
TIME_LIMIT="${SLURM_TIME:-8:00:00}"
CPUS_PER_GPU="${SLURM_CPUS_PER_GPU:-8}"
SCRIPT_BASENAME="$(basename "$WORKFLOW_SCRIPT" .sh)"
JOB_NAME="${SLURM_JOB_NAME:-${SCRIPT_BASENAME}}"
WORKDIR="$(cd "$(dirname "$WORKFLOW_SCRIPT")/.." && pwd)"
TOTAL_CPUS=$((CPUS_PER_GPU * GPUS))

# Log directory
LOG_DIR="${WORKDIR}/slurm_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_DIR}/${JOB_NAME}-${TIMESTAMP}.log"

# --- Build the temporary sbatch script ---
TMP_SCRIPT="$(mktemp /tmp/slurm_job_XXXXXX.sh)"
trap "rm -f $TMP_SCRIPT" EXIT

cat > "$TMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${TOTAL_CPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --export=ALL

echo "=== SLURM Job Info ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "GPUs: \$SLURM_JOB_GPUS / CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "Started: \$(date)"
echo "======================"

# Move to the project working directory
cd "${WORKDIR}"

# Export LEARNING_SOURCE_DIR for the workflow script
export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}"

# Use the GPUs allocated by SLURM (skip auto_select_gpu)
if [ "${GPUS}" -gt 1 ]; then
    # Multi-GPU: use every GPU allocated by SLURM.
    # SLURM normally sets CUDA_VISIBLE_DEVICES itself; if not, fall back to 0..N-1.
    if [ -z "\$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$((${GPUS} - 1)))
    fi
    echo "Multi-GPU mode: ${GPUS} GPUs (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES)"
else
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
fi

# Export NUM_GPUS so the workflow script can read it
export NUM_GPUS=${GPUS}

# Run the workflow script. SLURM already manages the job lifecycle so nohup/&
# inside the workflow is unnecessary, but we source the script so any backgrounded
# child processes remain direct children of this shell and `wait` catches them.
. "${WORKFLOW_SCRIPT}"

# Wait for backgrounded processes (including those launched via nohup &).
wait

echo "=== Job Finished: \$(date) ==="
EOF

# --- Submit the job ---
echo "Submitting SLURM job..."
echo "  Script:    $WORKFLOW_SCRIPT"
echo "  Partition: $PARTITION"
echo "  GPUs:      $GPUS"
echo "  CPUs:      $TOTAL_CPUS"
echo "  Time:      $TIME_LIMIT"
echo "  Log:       $LOG_FILE"
echo "  Workdir:   $WORKDIR"
if [ "$GPUS" -gt 1 ]; then
    echo "  Mode:      Multi-GPU (DDP via torchrun)"
fi
echo ""

JOB_RESULT=$(sbatch "$@" "$TMP_SCRIPT")
echo "$JOB_RESULT"

JOB_ID=$(echo "$JOB_RESULT" | awk '{print $NF}')
echo ""
echo "Check job status:"
echo "  squeue -j $JOB_ID"
echo "Tail the log in real time:"
echo "  tail -f $LOG_FILE"
