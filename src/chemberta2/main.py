#!/usr/bin/env python3
"""
ChemBERTa-2 Training Script

SMILES化合物データに特化したRoBERTaベースのTransformerモデルの学習スクリプト。
ChemBERTa-2アーキテクチャを使用し、大規模な化合物データで学習します。

Features:
- SMILES専用のトークナイゼーション
- RoBERTaアーキテクチャ（BERTの改良版）
- 化合物特性予測への転移学習が容易
- 効率的なバッチ処理とメモリ管理
"""

import os
import logging
from datetime import datetime

try:
    from datasets import load_from_disk
    from transformers import (
        RobertaConfig,
        RobertaForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    pass  # Not available in documentation-only environments (e.g. pdoc)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Default config values

if __name__ == "__main__":
    model_size = None
    model_path = None
    max_length = 256
    dataset_dir = None
    tokenizer = None
    vocab_size = None
    meta_vocab_size = None
    learning_rate = 6e-5  # ChemBERTa-2 optimized learning rate
    weight_decay = 0.01
    max_steps = 300000
    batch_size = 128
    per_device_eval_batch_size = 128
    gradient_accumulation_steps = 1  # Effective batch size = 128
    log_interval = 100
    save_steps = 5000
    mlm_probability = 0.15
    warmup_steps = 10000
    preprocess_function = None
    use_wandb = os.environ.get("USE_WANDB", "False").lower() in ("true", "1", "yes")
    wandb_project = os.environ.get("WANDB_PROJECT", "chemberta2-compounds")
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    fp16 = True  # Mixed precision training

    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    # Load config from file
    configurator_path = (
        "chemberta2/configurator.py"
        if os.path.exists("chemberta2/configurator.py")
        else "configurator.py"
    )
    if os.path.exists(configurator_path):
        exec(open(configurator_path).read())

    config = {k: globals()[k] for k in config_keys}
    # -----------------------------------------------------------------------------

    # Validate required parameters
    if model_size is None:
        raise ValueError("model_size must be specified in config file")
    if tokenizer is None:
        raise ValueError("tokenizer must be specified in config file")
    if not dataset_dir:
        raise ValueError("dataset_dir must be specified in config file")

    # Get vocab size from tokenizer
    if not ("meta_vocab_size" in vars() and "meta_vocab_size" in globals()):
        try:
            meta_vocab_size = (len(tokenizer) // 8 + 1) * 8
            print(f"📊 Calculated meta_vocab_size: {meta_vocab_size}")
        except Exception as e:
            raise ImportError(
                "Please initialize the variable meta_vocab_size in the config.py file with the size of your vocabulary."
            ) from e

    # ChemBERTa-2 Model Configuration
    # Based on RoBERTa architecture optimized for SMILES compounds
    if model_size == "small":
        model_config = RobertaConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length + 2,  # +2 for special tokens
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,  # RoBERTa doesn't use token_type_ids
        )
    elif model_size == "medium":
        model_config = RobertaConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length + 2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )
    elif model_size == "large":
        model_config = RobertaConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length + 2,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=1,
        )
    else:
        raise ValueError(
            f"Unknown model_size: {model_size}. Choose from: small, medium, large"
        )

    logger.info("✅ Configuration loaded")
    logger.info(f"🧪 ChemBERTa-2 Model Configuration ({model_size}):")
    logger.info(f"   - Vocab size: {meta_vocab_size}")
    logger.info(f"   - Max length: {max_length}")
    logger.info(f"   - Hidden size: {model_config.hidden_size}")
    logger.info(f"   - Layers: {model_config.num_hidden_layers}")
    logger.info(f"   - Attention heads: {model_config.num_attention_heads}")

    # Initialize wandb if enabled
    if use_wandb:
        import wandb

        # Determine dataset name from config
        dataset_name = config.get("dataset_name", "compounds")

        # Add metadata tags for experiment management
        tags = ["chemberta2", "training", model_size, dataset_name]

        # Add experiment metadata to config
        experiment_config = {
            **config,
            "experiment_type": "training",
            "model_type": "chemberta2",
            "dataset_type": dataset_name,
            "model_size": model_size,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"chemberta2-{model_size}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=experiment_config,
            tags=tags,
        )
        logger.info(f"📊 Wandb initialized: {wandb.run.get_url()}")

    class CompoundsDatasetLoader:
        """
        SMILES Compounds Dataset Loader

        SMILES化合物データを読み込み、ChemBERTa-2学習用に前処理します。
        複数のデータセットを動的に読み込み、結合することができます。
        """

        def __init__(self, dataset_dir, tokenizer, max_length=256):
            self.dataset_dir = dataset_dir
            self.tokenizer = tokenizer
            self.max_length = max_length

        def load_datasets(self):
            """Load train and test datasets"""
            logger.info("📂 Loading datasets...")

            # 新しいマルチデータセットローダーを試す
            try:
                from pathlib import Path
                from compounds.dataset.multi_loader import MultiDatasetLoader

                compounds_dir = Path(self.dataset_dir).parent
                loader = MultiDatasetLoader(compounds_dir)

                # 利用可能なデータセットを取得
                available = loader.get_available_datasets()

                if available:
                    logger.info(
                        f"📊 Found {len(available)} available datasets: {[d.value for d in available]}"
                    )

                    # 全データセットを結合して読み込み
                    dataset_dict = loader.load_datasets(combine=True)

                    train_dataset = dataset_dict.get("train")
                    test_dataset = dataset_dict.get("valid") or dataset_dict.get("test")

                    if train_dataset and test_dataset:
                        logger.info(
                            f"✓ Loaded combined datasets: train={len(train_dataset)}, test={len(test_dataset)}"
                        )
                        return train_dataset, test_dataset
                    else:
                        logger.warning(
                            "⚠ Multi-loader succeeded but missing splits, falling back to legacy loader"
                        )

            except Exception as e:
                logger.warning(
                    f"⚠ Multi-loader failed ({e}), falling back to legacy single-dataset loader"
                )

            # レガシーローダー（後方互換性）
            train_path = os.path.join(self.dataset_dir, "train")
            test_path = os.path.join(self.dataset_dir, "test")
            valid_path = os.path.join(self.dataset_dir, "valid")

            # Load datasets
            if os.path.exists(train_path):
                logger.info("📂 Using legacy single-dataset loading")
                train_dataset = load_from_disk(train_path)
            else:
                raise FileNotFoundError(
                    f"Training dataset not found at {train_path}\n"
                    f"Please run the preparation pipeline first."
                )

            # Prefer valid over test
            if os.path.exists(valid_path):
                test_dataset = load_from_disk(valid_path)
                logger.info("📊 Using validation dataset for evaluation")
            elif os.path.exists(test_path):
                test_dataset = load_from_disk(test_path)
                logger.info("📊 Using test dataset for evaluation")
            else:
                # Use a small subset for testing if test set doesn't exist
                logger.info(
                    "📊 Test/valid dataset not found, using subset of training data"
                )
                test_size = min(5000, len(train_dataset) // 10)
                test_dataset = train_dataset.select(range(test_size))
                logger.info(
                    f"📊 Limited test dataset to {test_size} samples for faster evaluation"
                )

            return train_dataset, test_dataset

    # Initialize model
    logger.info("🔧 Initializing ChemBERTa-2 model...")
    model = RobertaForMaskedLM(model_config)
    logger.info(
        f"✅ Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    logger.info("Using default DataCollatorForLanguageModeling")

    # Load datasets
    dataset_loader = CompoundsDatasetLoader(dataset_dir, tokenizer, max_length)
    train_dataset, test_dataset = dataset_loader.load_datasets()

    # Apply preprocessing if defined
    if preprocess_function is not None:
        logger.info("🔄 Applying preprocessing function...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing train dataset",
        )
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing test dataset",
        )

        logger.info("✅ Preprocessing completed")
        logger.info(f"   Train dataset columns: {train_dataset.column_names}")
        logger.info(f"   Test dataset columns: {test_dataset.column_names}")

        # Verify the preprocessing worked
        sample = train_dataset[0]
        logger.info(f"   Sample keys: {list(sample.keys())}")
        if "attention_mask" in sample:
            logger.info("   ✓ attention_mask successfully added")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        logging_steps=log_interval,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=fp16,
        dataloader_num_workers=4,
        report_to="wandb" if use_wandb else "none",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Check for existing checkpoints
    resume_checkpoint = None
    if os.path.exists(model_path):
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(model_path, latest_checkpoint)
            logger.info(f"🔄 Found checkpoint: {resume_checkpoint}")
            logger.info(
                f"   Resuming training from step {latest_checkpoint.split('-')[1]}"
            )
        else:
            logger.info(f"ℹ️  No checkpoints found in {model_path}")
            logger.info("   Starting training from scratch...")
    else:
        logger.info(f"ℹ️  Output directory does not exist: {model_path}")
        logger.info("   Starting training from scratch...")

    # Train
    logger.info("🚀 Starting ChemBERTa-2 training...")
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info("✅ Training completed successfully!")

        # Save final model
        logger.info(f"💾 Saving final model to {model_path}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("✅ Model saved successfully!")

    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        logger.info(f"💾 Saving checkpoint to {model_path}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("✅ Checkpoint saved")

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise

    finally:
        if use_wandb:
            wandb.finish()
            logger.info("📊 Wandb run finished.")
