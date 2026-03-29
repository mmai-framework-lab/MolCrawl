#!/bin/bash
# Common functions for bootstrap scripts

# ---------------------------------------------------------------------------
# Python executable — prefer local miniconda, then molcrawl conda env
# ---------------------------------------------------------------------------
_SCRIPT_DIR_CF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_LOCAL_MINICONDA_PYTHON="${_SCRIPT_DIR_CF}/../miniconda/bin/python"
if [ -f "$_LOCAL_MINICONDA_PYTHON" ]; then
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
unset _LOCAL_MINICONDA_PYTHON _MOLCRAWL_PYTHON _SCRIPT_DIR_CF

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
