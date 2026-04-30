# GPT2 test configuration for PROTEIN_SEQUENCE domain
import torch

# Basic settings
domain = "protein_sequence"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
dataset_params = {"dataset_dir": "outputs/protein_sequence/training_ready_hf_dataset"}

# Output settings
output_dir = "test_results_protein_sequence"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
