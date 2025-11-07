# RNAドメイン用GPT2テスト設定

# 基本設定
domain = "rna"
max_test_samples = 1000
convert_to_hf = True

# データセット設定
dataset_params = {"dataset_dir": "outputs/rna/training_ready_hf_dataset"}

# 出力設定
output_dir = "test_results_rna"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
