import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
src_path = os.path.join(current_dir, "..", "..", "..", "src")

try:
    from config.paths import get_gpt2_output_path

    tensorboard_dir = get_gpt2_output_path("rna", "yigarashi-2025-10-08-large")
    out_dir = get_gpt2_output_path("rna", "yigarashi-2025-10-08-large")
except ImportError:
    # Fallback if config.paths is not available
    tensorboard_dir = "gpt2-output/rna-yigarashi-2025-10-08-large"
    out_dir = "gpt2-output/rna-yigarashi-2025-10-08-large"

# Tensorboard and output settings
tensorboard = True  # log training metrics to tensorboard

# Training parameters optimized for RNA transcriptome data
batch_size = 8  # RNA data can be large, so smaller batch size
block_size = 512  # Reasonable context length for gene sequences
gradient_accumulation_steps = 8  # Compensate for smaller batch size

# Training schedule
max_iters = 100000
lr_decay_iters = 100000
warmup_iters = 1000  # Warmup steps
learning_rate = 3e-4  # Learning rate for RNA data
min_lr = learning_rate / 10  # Minimum learning rate

# Evaluation
eval_interval = 2000
eval_iters = 100
log_interval = 100

# Regularization
weight_decay = 1e-2
dropout = 0.1  # Some dropout for generalization

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - デフォルトでチェックポイントから再開

# Large-Sized GPT2 Model
n_layer = 36
n_head = 20
n_embd = 1280
bias = False

# Dataset
dataset = "rna"

# RNA-specific settings - these will be overridden by the actual vocab size from the data
meta_vocab_size = 60666  # Size from gene_vocab.json

# Other settings
compile = False  # Disable compilation for debugging
device = "cuda" if os.path.exists("/usr/bin/nvidia-smi") else "cpu"
dtype = "bfloat16" if device == "cuda" else "float32"

# Dataset parameters (empty since we're using hardcoded paths in train.py)
dataset = "rna"

dataset_params = {"dataset_dir": "/wren/yigarashi/molcrawl/parquet_sample_1pct"}
