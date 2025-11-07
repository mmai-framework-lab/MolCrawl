# PROTEIN_SEQUENCEドメイン用GPT2テスト設定

# 基本設定
domain = "protein_sequence"
max_test_samples = 1000
convert_to_hf = True

# データセット設定
dataset_params = {"dataset_dir": "outputs/protein_sequence/training_ready_hf_dataset"}

# 出力設定
output_dir = "test_results_protein_sequence"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
