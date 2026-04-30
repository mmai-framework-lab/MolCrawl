# GPT2 test configuration for COMPOUNDS domain

import os
import sys

import torch

from molcrawl.core.paths import COMPOUNDS_DATASET_DIR

# Basic settings
domain = "compounds"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
dataset_params = {"dataset_dir": COMPOUNDS_DATASET_DIR}

# vocabulary file
vocab_path = "assets/molecules/vocab.txt"

# Output settings
output_dir = "test_results_compounds"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
