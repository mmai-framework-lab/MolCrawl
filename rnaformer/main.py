#!/usr/bin/env python3
"""
RNAformer Training Script

RNA配列（遺伝子発現データ）に特化したTransformerモデルの学習スクリプト。
Geneformerアーキテクチャをベースに、RNA transcriptomeデータの学習に最適化。

Features:
- 遺伝子発現データ用のカスタムトークナイゼーション
- セルタイプ特異的な学習
- 長いコンテキスト（1024トークン）のサポート
- 効率的なバッチ処理とメモリ管理
"""

import os
import logging
from datetime import datetime

import wandb
from datasets import load_from_disk
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Default config values
model_size = None
model_path = None
max_length = 1024
dataset_dir = None
tokenizer = None
vocab_size = None
meta_vocab_size = None
learning_rate = 1e-4  # RNAformer optimized learning rate
weight_decay = 0.1
max_steps = 100000
batch_size = 8
per_device_eval_batch_size = 4
gradient_accumulation_steps = 16  # Effective batch size = 8 * 16 = 128
log_interval = 100
save_steps = 1000
mlm_probability = 0.15
warmup_steps = 10000
preprocess_function = None
use_wandb = os.environ.get("USE_WANDB", "False").lower() in ("true", "1", "yes")
wandb_project = os.environ.get("WANDB_PROJECT", "rnaformer-transcriptome")
wandb_entity = os.environ.get("WANDB_ENTITY", None)
fp16 = True  # Mixed precision training

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]

# Load config from file
configurator_path = "rnaformer/configurator.py" if os.path.exists("rnaformer/configurator.py") else "configurator.py"
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

# RNAformer Model Configuration
# Based on Geneformer architecture optimized for RNA transcriptome data
if model_size == "small":
    model_config = BertConfig(
        vocab_size=meta_vocab_size,
        max_position_embeddings=max_length,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
elif model_size == "medium":
    model_config = BertConfig(
        vocab_size=meta_vocab_size,
        max_position_embeddings=max_length,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
elif model_size == "large":
    model_config = BertConfig(
        vocab_size=meta_vocab_size,
        max_position_embeddings=max_length,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
else:
    raise ValueError(f"Unknown model_size: {model_size}. Choose from: small, medium, large")

logger.info("✅ Configuration loaded")
logger.info(f"🧬 RNAformer Model Configuration ({model_size}):")
logger.info(f"   - Vocab size: {meta_vocab_size}")
logger.info(f"   - Max length: {max_length}")
logger.info(f"   - Hidden size: {model_config.hidden_size}")
logger.info(f"   - Layers: {model_config.num_hidden_layers}")
logger.info(f"   - Attention heads: {model_config.num_attention_heads}")

# Initialize wandb if enabled
if use_wandb:
    # Determine dataset name from config
    dataset_name = config.get('dataset_name', 'rna')
    
    # Add metadata tags for experiment management
    tags = ['rnaformer', 'training', model_size, dataset_name]
    
    # Add experiment metadata to config
    experiment_config = {
        **config,
        'experiment_type': 'training',
        'model_type': 'rnaformer',
        'dataset_type': dataset_name,
        'model_size': model_size,
    }
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"rnaformer-{model_size}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=experiment_config,
        tags=tags,
    )
    logger.info(f"📊 Wandb initialized: {wandb.run.get_url()}")


class RNADatasetLoader:
    """
    RNA Transcriptome Dataset Loader
    
    遺伝子発現データを読み込み、RNAformer学習用に前処理します。
    """
    
    def __init__(self, dataset_dir, tokenizer, max_length=1024):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_datasets(self):
        """Load train and test datasets"""
        logger.info("📂 Loading datasets...")
        
        train_path = os.path.join(self.dataset_dir, "train")
        test_path = os.path.join(self.dataset_dir, "test")
        
        # Load datasets
        if os.path.exists(train_path):
            logger.info("📂 Using standard HuggingFace dataset loading")
            train_dataset = load_from_disk(train_path)
        else:
            raise FileNotFoundError(f"Training dataset not found at {train_path}")
        
        if os.path.exists(test_path):
            test_dataset = load_from_disk(test_path)
        else:
            # Use a small subset for testing if test set doesn't exist
            logger.info("📊 Test dataset not found, using subset of training data")
            test_size = min(5000, len(train_dataset) // 10)
            test_dataset = train_dataset.select(range(test_size))
            logger.info(f"📊 Limited test dataset to {test_size} samples for faster evaluation")
        
        return train_dataset, test_dataset


# Initialize model
logger.info("🔧 Initializing RNAformer model...")
model = BertForMaskedLM(model_config)
logger.info(f"✅ Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_probability,
)
logger.info("Using default DataCollatorForLanguageModeling")

# Load datasets
dataset_loader = RNADatasetLoader(dataset_dir, tokenizer, max_length)
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
    lr_scheduler_type="cosine",
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
        logger.info(f"   Resuming training from step {latest_checkpoint.split('-')[1]}")
    else:
        logger.info(f"ℹ️  No checkpoints found in {model_path}")
        logger.info("   Starting training from scratch...")
else:
    logger.info(f"ℹ️  Output directory does not exist: {model_path}")
    logger.info("   Starting training from scratch...")

# Train
logger.info("🚀 Starting RNAformer training...")
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
