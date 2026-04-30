# GPT2 test configuration for MOLECULE_NAT_LANG domain

import os
import sys

import torch

from molcrawl.core.paths import MOLECULE_NAT_LANG_DATASET_DIR

# Basic settings
domain = "molecule_nat_lang"
max_test_samples = 1000
convert_to_hf = True

# datasetsetting
dataset_params = {"dataset_dir": MOLECULE_NAT_LANG_DATASET_DIR}

# Output settings
output_dir = "test_results_molecule_nat_lang"

# device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
