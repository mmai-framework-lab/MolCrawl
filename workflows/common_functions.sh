#!/bin/bash
# Common functions for bootstrap scripts

# ---------------------------------------------------------------------------
# Python executable — select ROCm env on AMD GPU nodes (gpu04), molcrawl elsewhere
# ---------------------------------------------------------------------------
_SCRIPT_DIR_CF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_LOCAL_MINICONDA_PYTHON="${_SCRIPT_DIR_CF}/../miniconda/bin/python"

_NODE_NAME="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null)}"
_IS_AMD_NODE="false"
if [ "$_NODE_NAME" = "gpu04" ]; then
    _IS_AMD_NODE="true"
fi

if [ "$_IS_AMD_NODE" = "true" ]; then
    # Override with MOLCRAWL_ROCM_PYTHON if set; otherwise look for the
    # standard conda layout under the user's home directory.
    _ROCM_PYTHON="${MOLCRAWL_ROCM_PYTHON:-$HOME/miniforge3/envs/molcrawl_rocm/bin/python}"
    if [ -f "$_ROCM_PYTHON" ]; then
        PYTHON="$_ROCM_PYTHON"
        echo "Using molcrawl_rocm env for ROCm node: $PYTHON"
    else
        echo "ERROR: Running on AMD GPU node $_NODE_NAME but molcrawl_rocm env not found." >&2
        PYTHON="$(which python3 || which python)"
    fi
elif [ -f "$_LOCAL_MINICONDA_PYTHON" ]; then
    PYTHON="$(realpath "$_LOCAL_MINICONDA_PYTHON")"
else
    _MOLCRAWL_PYTHON="$(conda run -n molcrawl which python 2>/dev/null || true)"
    if [ -n "$_MOLCRAWL_PYTHON" ] && [ -f "$_MOLCRAWL_PYTHON" ]; then
        PYTHON="$_MOLCRAWL_PYTHON"
    else
        echo "WARNING: local miniconda and conda env 'molcrawl' not found. Falling back to system python." >&2
        PYTHON="$(which python3 || which python)"
    fi
fi
export PYTHON
export PYTHONUNBUFFERED=1
unset _LOCAL_MINICONDA_PYTHON _MOLCRAWL_PYTHON _ROCM_PYTHON _NODE_NAME _IS_AMD_NODE _SCRIPT_DIR_CF

# Check if LEARNING_SOURCE_DIR environment variable is set
# Usage: check_learning_source_dir
check_learning_source_dir() {
    if [ -z "$LEARNING_SOURCE_DIR" ]; then
        echo "ERROR: LEARNING_SOURCE_DIR environment variable is not set."
        echo "Please set it before running this script:"
        echo "  export LEARNING_SOURCE_DIR='...'"
        exit 1
    fi
    echo "DatabaseDir: $LEARNING_SOURCE_DIR"
}

# Select the GPU with the most free memory
# Usage: select_best_gpu
# Returns: GPU ID with most free memory
select_best_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi command not found. Cannot detect GPU."
        return 1
    fi

    # Get GPU with most free memory (in MB)
    # Format: GPU_ID FREE_MEMORY_MB
    local best_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{print $1, $2}' | \
        sort -k2 -rn | \
        head -1 | \
        awk '{print $1}')

    if [ -z "$best_gpu" ]; then
        echo "ERROR: Could not determine best GPU."
        return 1
    fi

    echo "$best_gpu"
}

# Check if GPU has enough free memory
# Usage: check_gpu_memory GPU_ID MIN_MEMORY_GB
# Returns: 0 if enough memory, 1 otherwise
check_gpu_memory() {
    local gpu_id=$1
    local min_memory_gb=$2

    if [ -z "$gpu_id" ] || [ -z "$min_memory_gb" ]; then
        echo "ERROR: GPU ID and minimum memory (GB) are required."
        return 1
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi command not found. Skipping memory check."
        return 0
    fi

    # Get free memory in MB for specified GPU
    local free_memory_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id")
    local free_memory_gb=$(echo "scale=2; $free_memory_mb / 1024" | bc)

    echo "GPU $gpu_id: ${free_memory_gb}GB free memory available"

    # Compare with minimum required
    local has_enough=$(echo "$free_memory_gb >= $min_memory_gb" | bc)

    if [ "$has_enough" -eq 0 ]; then
        echo "WARNING: GPU $gpu_id has only ${free_memory_gb}GB free, but ${min_memory_gb}GB required."
        echo "Training may fail with CUDA Out of Memory error."
        return 1
    fi

    return 0
}

# Auto-select GPU if CUDA_VISIBLE_DEVICES is not set
# Usage: auto_select_gpu [MIN_MEMORY_GB]
# Sets CUDA_VISIBLE_DEVICES to the best available GPU
auto_select_gpu() {
    local min_memory_gb=${1:-10}  # Default: 10GB minimum

    # On AMD GPU nodes, SLURM sets ROCR_VISIBLE_DEVICES; mirror it to CUDA_VISIBLE_DEVICES
    # so downstream PyTorch code (which uses "cuda" device names via HIP) works transparently.
    if [ "$(hostname -s 2>/dev/null)" = "gpu04" ] || [ "${SLURMD_NODENAME:-}" = "gpu04" ]; then
        if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
            export CUDA_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
            echo "ROCm node detected: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from ROCR_VISIBLE_DEVICES)"
        else
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            echo "ROCm node detected: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (fallback)"
        fi
        return 0
    fi

    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is already set to: $CUDA_VISIBLE_DEVICES"
        # Still check if the selected GPU has enough memory
        check_gpu_memory "$CUDA_VISIBLE_DEVICES" "$min_memory_gb" || {
            echo "WARNING: Proceeding anyway with user-specified GPU."
        }
        return 0
    fi

    echo "Auto-selecting GPU with most free memory..."
    local best_gpu=$(select_best_gpu)

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to select GPU automatically."
        return 1
    fi

    export CUDA_VISIBLE_DEVICES=$best_gpu
    echo "Selected GPU $best_gpu"

    # Check if it has enough memory
    check_gpu_memory "$best_gpu" "$min_memory_gb" || {
        echo "WARNING: Proceeding anyway. Monitor for OOM errors."
    }

    return 0
}

# ---------------------------------------------------------------------------
# Multi-GPU support — select multiple GPUs and launch training via torchrun
# ---------------------------------------------------------------------------

# Select top-N GPUs by free memory
# Usage: select_multi_gpu [NUM_GPUS] [MIN_MEMORY_GB]
# Sets CUDA_VISIBLE_DEVICES to comma-separated GPU IDs
select_multi_gpu() {
    local num_gpus=${1:-1}
    local min_memory_gb=${2:-10}

    # On AMD GPU nodes, defer to ROCR_VISIBLE_DEVICES
    if [ "$(hostname -s 2>/dev/null)" = "gpu04" ] || [ "${SLURMD_NODENAME:-}" = "gpu04" ]; then
        if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
            export CUDA_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
            echo "ROCm node detected: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from ROCR_VISIBLE_DEVICES)"
        else
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            echo "ROCm node detected: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (fallback)"
        fi
        return 0
    fi

    # If CUDA_VISIBLE_DEVICES is already set by the user or SLURM, respect it
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is already set to: $CUDA_VISIBLE_DEVICES"
        return 0
    fi

    if [ "$num_gpus" -le 1 ]; then
        auto_select_gpu "$min_memory_gb"
        return $?
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi command not found. Cannot detect GPUs."
        return 1
    fi

    echo "Auto-selecting top ${num_gpus} GPUs by free memory..."
    local selected=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{print $1, $2}' | \
        sort -k2 -rn | \
        head -"$num_gpus" | \
        sort -k1 -n | \
        awk '{print $1}' | \
        paste -sd,)

    if [ -z "$selected" ]; then
        echo "ERROR: Could not select GPUs."
        return 1
    fi

    local actual_count=$(echo "$selected" | tr ',' '\n' | wc -l)
    if [ "$actual_count" -lt "$num_gpus" ]; then
        echo "WARNING: Requested ${num_gpus} GPUs but only ${actual_count} available."
    fi

    export CUDA_VISIBLE_DEVICES="$selected"
    echo "Selected GPUs: $CUDA_VISIBLE_DEVICES"

    # Check memory on each selected GPU
    for gpu_id in $(echo "$selected" | tr ',' ' '); do
        check_gpu_memory "$gpu_id" "$min_memory_gb" || {
            echo "WARNING: GPU $gpu_id may not have enough memory. Proceeding anyway."
        }
    done

    return 0
}

# Count the number of GPUs in CUDA_VISIBLE_DEVICES
# Usage: count_visible_gpus
count_visible_gpus() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "1"
        return
    fi
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' '
}

# Build torchrun command prefix for multi-GPU training
# Usage: build_torchrun_cmd
# Prints the torchrun prefix if multi-GPU, empty string if single-GPU
build_torchrun_cmd() {
    local ngpus
    ngpus=$(count_visible_gpus)

    if [ "$ngpus" -le 1 ]; then
        echo ""
        return
    fi

    # Resolve torchrun from the same directory as $PYTHON
    local python_dir
    python_dir="$(dirname "$PYTHON")"
    local torchrun_bin="${python_dir}/torchrun"

    if [ -x "$torchrun_bin" ]; then
        echo "${torchrun_bin} --standalone --nproc_per_node=${ngpus}"
    elif command -v torchrun &> /dev/null; then
        echo "torchrun --standalone --nproc_per_node=${ngpus}"
    else
        # Fallback: use python -m torch.distributed.run
        echo "${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${ngpus}"
    fi
}

# Run a training script with automatic single/multi-GPU handling
# Usage: run_training <python_script> [script_args...]
#
# Behavior:
#   - If CUDA_VISIBLE_DEVICES lists multiple GPUs: launches via torchrun for DDP
#   - Otherwise: launches directly with $PYTHON
#
# Examples:
#   run_training molcrawl/models/gpt2/train.py gpt2/configs/compounds/train_gpt2_small_config.py
#   run_training molcrawl/models/bert/main.py bert/configs/compounds.py
run_training() {
    local script="$1"
    shift
    local args="$@"

    local ngpus
    ngpus=$(count_visible_gpus)
    local torchrun_cmd
    torchrun_cmd=$(build_torchrun_cmd)

    if [ -n "$torchrun_cmd" ]; then
        echo "Multi-GPU training: ${ngpus} GPUs via torchrun"
        echo "Command: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${torchrun_cmd} ${script} ${args}"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $torchrun_cmd $script $args
    else
        echo "Single-GPU training: GPU ${CUDA_VISIBLE_DEVICES}"
        echo "Command: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON} ${script} ${args}"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $PYTHON $script $args
    fi
}

# Run a training script in background with automatic single/multi-GPU handling
# Usage: run_training_background <log_file> <python_script> [script_args...]
#
# When running inside a SLURM job (SLURM_JOB_ID is set), runs in foreground
# with output redirected to the log file. This prevents SLURM from killing
# the training process when the job script exits.
#
# When running outside SLURM, runs in background with nohup as before.
run_training_background() {
    local log_file="$1"
    shift
    local script="$1"
    shift
    local args="$@"

    local ngpus
    ngpus=$(count_visible_gpus)
    local torchrun_cmd
    torchrun_cmd=$(build_torchrun_cmd)

    # Inside SLURM: run foreground (SLURM manages the process lifecycle)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        if [ -n "$torchrun_cmd" ]; then
            echo "Multi-GPU training: ${ngpus} GPUs via torchrun (SLURM foreground)"
            echo "Log: ${log_file}"
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
                $torchrun_cmd $script $args > "$log_file" 2>&1
        else
            echo "Single-GPU training: GPU ${CUDA_VISIBLE_DEVICES} (SLURM foreground)"
            echo "Log: ${log_file}"
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
                $PYTHON $script $args > "$log_file" 2>&1
        fi
        return
    fi

    # Outside SLURM: run in background with nohup
    if [ -n "$torchrun_cmd" ]; then
        echo "Multi-GPU training: ${ngpus} GPUs via torchrun (background)"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
            nohup $torchrun_cmd $script $args > "$log_file" 2>&1 &
    else
        echo "Single-GPU training: GPU ${CUDA_VISIBLE_DEVICES} (background)"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
            nohup $PYTHON $script $args > "$log_file" 2>&1 &
    fi

    local pid=$!
    echo "Training started (PID: ${pid})"
    echo "Log: ${log_file}"
}
