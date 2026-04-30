"""
BERT version ProteinGym evaluation configuration file
"""

import os
from pathlib import Path

# Add common environment check module
from molcrawl.core.utils.environment_check import check_learning_source_dir

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


class BERTProteinGymConfig:
    """BERT ProteinGym evaluation setting class"""

    # Model settings
    MODEL_PATH = "runs_train_bert_protein_sequence/checkpoint-5000"
    # ``TOKENIZER_PATH`` used to be computed at class-definition time via
    # ``check_learning_source_dir()``, which made *importing* this module
    # require the env var to be set. Use a property (resolved at access
    # time) so the module is import-safe; the env-var check still fires
    # the first time a script reads ``cfg.TOKENIZER_PATH``.
    @property
    def TOKENIZER_PATH(self) -> str:  # noqa: N802 — preserve legacy attr name
        learning_source_dir = check_learning_source_dir()
        return f"{learning_source_dir}/protein_sequence/spm_tokenizer.model"

    # Evaluation settings
    DEVICE = "cuda"
    BATCH_SIZE = 16
    MAX_SEQUENCE_LENGTH = 512

    # Data settings
    DEFAULT_OUTPUT_DIR = "./bert_proteingym_evaluation_results"

    # Log settings
    LOG_DIR = "logs"
    LOG_LEVEL = "INFO"

    # BERT specific settings
    BERT_CONFIG = {
        "vocab_size": None,  # Automatically obtained from tokenizer
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 1024,
        "dropout": 0.0,  # disabled during evaluation
        "attention_dropout": 0.0,  # Disabled during evaluation
    }

    # Setting evaluation indicators
    METRICS = ["spearman_correlation", "pearson_correlation", "mae", "rmse"]

    # Sampling settings
    DEFAULT_SAMPLE_SIZE = None  # Rating all items with None

    @classmethod
    def get_model_path(cls):
        """Get model path"""
        return os.path.join(PROJECT_ROOT, cls.MODEL_PATH)

    @classmethod
    def get_tokenizer_path(cls):
        """Get Tokenizer Pass"""
        return os.path.join(PROJECT_ROOT, cls.TOKENIZER_PATH)

    @classmethod
    def get_output_dir(cls, custom_dir=None):
        """Get output directory"""
        if custom_dir:
            return custom_dir
        return os.path.join(PROJECT_ROOT, cls.DEFAULT_OUTPUT_DIR)

    @classmethod
    def get_log_dir(cls):
        """Get log directory"""
        return os.path.join(PROJECT_ROOT, cls.LOG_DIR)

    @classmethod
    def validate_paths(cls):
        """Check the existence of the path"""
        model_path = cls.get_model_path()
        tokenizer_path = cls.get_tokenizer_path()

        issues = []

        if not os.path.exists(model_path):
            issues.append(f"Model path not found: {model_path}")

        if not os.path.exists(tokenizer_path):
            issues.append(f"Tokenizer path not found: {tokenizer_path}")

        return issues


# Instance with default settings
config = BERTProteinGymConfig()
