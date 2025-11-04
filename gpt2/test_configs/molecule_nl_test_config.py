# MOLECULE_NLドメイン用GPT2テスト設定

import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config.paths import MOLECULE_NL_DATASET_DIR

# 基本設定
domain = "molecule_nl"
max_test_samples = 1000
convert_to_hf = True

# データセット設定
dataset_params = {
    "dataset_dir": MOLECULE_NL_DATASET_DIR
}

# 出力設定
output_dir = "test_results_molecule_nl"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
