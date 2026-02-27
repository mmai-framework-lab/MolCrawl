"""
BERT版ProteinGym評価の設定ファイル
"""

import os
from pathlib import Path

# 共通環境チェックモジュールを追加
from molcrawl.utils.environment_check import check_learning_source_dir

# プロジェクトルートの取得
PROJECT_ROOT = Path(__file__).parent.parent


class BERTProteinGymConfig:
    """BERT ProteinGym評価設定クラス"""

    # モデル設定
    MODEL_PATH = "runs_train_bert_protein_sequence/checkpoint-5000"
    learning_source_dir = check_learning_source_dir()
    TOKENIZER_PATH = f"{learning_source_dir}/protein_sequence/spm_tokenizer.model"

    # 評価設定
    DEVICE = "cuda"
    BATCH_SIZE = 16
    MAX_SEQUENCE_LENGTH = 512

    # データ設定
    DEFAULT_OUTPUT_DIR = "./bert_proteingym_evaluation_results"

    # ログ設定
    LOG_DIR = "logs"
    LOG_LEVEL = "INFO"

    # BERT固有設定
    BERT_CONFIG = {
        "vocab_size": None,  # トークナイザーから自動取得
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 1024,
        "dropout": 0.0,  # 評価時は無効
        "attention_dropout": 0.0,  # 評価時は無効
    }

    # 評価指標の設定
    METRICS = ["spearman_correlation", "pearson_correlation", "mae", "rmse"]

    # サンプリング設定
    DEFAULT_SAMPLE_SIZE = None  # Noneで全件評価

    @classmethod
    def get_model_path(cls):
        """モデルパスを取得"""
        return os.path.join(PROJECT_ROOT, cls.MODEL_PATH)

    @classmethod
    def get_tokenizer_path(cls):
        """トークナイザーパスを取得"""
        return os.path.join(PROJECT_ROOT, cls.TOKENIZER_PATH)

    @classmethod
    def get_output_dir(cls, custom_dir=None):
        """出力ディレクトリを取得"""
        if custom_dir:
            return custom_dir
        return os.path.join(PROJECT_ROOT, cls.DEFAULT_OUTPUT_DIR)

    @classmethod
    def get_log_dir(cls):
        """ログディレクトリを取得"""
        return os.path.join(PROJECT_ROOT, cls.LOG_DIR)

    @classmethod
    def validate_paths(cls):
        """パスの存在確認"""
        model_path = cls.get_model_path()
        tokenizer_path = cls.get_tokenizer_path()

        issues = []

        if not os.path.exists(model_path):
            issues.append(f"Model path not found: {model_path}")

        if not os.path.exists(tokenizer_path):
            issues.append(f"Tokenizer path not found: {tokenizer_path}")

        return issues


# デフォルト設定のインスタンス
config = BERTProteinGymConfig()
