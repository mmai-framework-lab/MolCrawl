# GPT2 test settings for RNA domain
import torch

# Basic settings
domain = "rna"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
dataset_params = {"dataset_dir": "outputs/rna/training_ready_hf_dataset"}

# Output settings
output_dir = "test_results_rna"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
