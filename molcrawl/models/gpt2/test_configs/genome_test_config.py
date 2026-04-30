# GPT2 test configuration for GENOME domain
import torch

# Basic settings
domain = "genome_sequence"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
dataset_params = {"dataset_dir": "outputs/genome_sequence/training_ready_hf_dataset"}

# Output settings
output_dir = "test_results_genome"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
