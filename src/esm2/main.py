"""
ESM-2 Training Script for Protein Sequence Data

ESM-2 (Evolutionary Scale Modeling 2) は、Metaが開発したタンパク質配列専用の
最先端トランスフォーマーモデルです。

主な特徴:
- タンパク質配列に特化した事前学習
- 6.5億パラメータまでのスケーラブルなアーキテクチャ
- Structure prediction, function annotation, variant effect predictionなど幅広いタスクに対応
- ESM-1bよりも高速で高精度

参考:
- Language models of protein sequences at the scale of evolution enable accurate structure prediction
- https://github.com/facebookresearch/esm
"""

import os
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, load_from_disk
from transformers import (
    EsmConfig,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class ProteinDatasetLoader:
    """
    タンパク質配列データセット用のローダー

    既存のprotein_sequenceデータセットを読み込み、
    ESM-2用に前処理を行います。
    """

    def __init__(self, data_dir, split="train", test_size=0.1):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size

        print(f"📂 Loading protein dataset from {data_dir}")

        # Load data from Arrow files or HuggingFace format
        try:
            arrow_files = list(Path(data_dir).glob("*.arrow"))
            if arrow_files:
                print(f"📁 Found {len(arrow_files)} arrow files")
                all_batches = []
                for arrow_file in arrow_files:
                    try:
                        print(f"📖 Reading {arrow_file.name}...")
                        with pa.memory_map(str(arrow_file), "r") as mmap:
                            with pa.ipc.open_stream(mmap) as reader:
                                table = reader.read_all()
                                print(f"🔢 Read table: {len(table)} rows")
                                all_batches.append(table)
                    except Exception as e:
                        print(f"❌ Failed to read {arrow_file.name}: {e}")
                        continue

                if all_batches:
                    combined_table = pa.concat_tables(all_batches)
                    print(f"📊 Combined {len(all_batches)} tables: {len(combined_table)} total rows")
                    df = combined_table.to_pandas()

                    # Convert numpy arrays to lists for HuggingFace compatibility
                    if "token" in df.columns:
                        df["token"] = df["token"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
                    if "sequence_tokens" in df.columns:
                        df["sequence_tokens"] = df["sequence_tokens"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)

                    self.dataset = Dataset.from_pandas(df)
                    print("✅ Created HuggingFace Dataset")
                    print(f"🔍 Dataset columns: {self.dataset.column_names}")
                else:
                    raise ValueError("No arrow files could be read successfully")
            else:
                raise FileNotFoundError(f"No .arrow files found in {data_dir}")

        except Exception as e:
            print(f"❌ Arrow loading failed: {e}")
            raise FileNotFoundError(f"Could not load data from {data_dir}") from e

        # Split into train/valid if needed
        if hasattr(self.dataset, "keys") and isinstance(self.dataset, dict) and "train" in self.dataset:
            if split == "train":
                self.data = self.dataset["train"]
            elif split in ["valid", "val", "test"]:
                self.data = self.dataset.get("valid", self.dataset.get("test", self.dataset["train"]))
        else:
            if test_size > 0:
                split_dataset = self.dataset.train_test_split(test_size=test_size, seed=42)
                if split == "train":
                    self.data = split_dataset["train"]
                elif split in ["valid", "val", "test"]:
                    self.data = split_dataset["test"]
            else:
                self.data = self.dataset

        print(f"✅ Loaded {len(self.data)} samples for {split}")

        # Show sample
        if len(self.data) > 0:
            sample = self.data[0]
            print("Sample keys:", list(sample.keys()))
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"  {key}: {type(value)} of length {len(value)}")
                else:
                    print(f"  {key}: {type(value)} = {value}")

    def get_dataset(self):
        """Return the HuggingFace Dataset object"""
        return self.data


# Default configuration values
model_size = None
tokenizer = None

# wandb settings
use_wandb = os.environ.get("USE_WANDB", "False").lower() in ("true", "1", "yes")
wandb_project = os.environ.get("WANDB_PROJECT", "esm2-training")
wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
wandb_entity = os.environ.get("WANDB_ENTITY", None)
wandb_log_model = os.environ.get("WANDB_LOG_MODEL", "True").lower() in ("true", "1", "yes")

model_path = ""
max_length = 1024  # ESM-2 default: 1024
dataset_dir = ""
learning_rate = 4e-4  # ESM-2推奨値（論文より）
weight_decay = 0.01
warmup_steps = 2000
max_steps = 500000
batch_size = 4  # タンパク質配列は長いためバッチサイズは小さめ
gradient_accumulation_steps = 32  # Effective batch size = 4 * 32 = 128
per_device_eval_batch_size = 2
log_interval = 100
save_steps = 5000

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]

# Load config from file
configurator_path = "esm2/configurator.py" if os.path.exists("esm2/configurator.py") else "configurator.py"
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
        meta_vocab_size = len(tokenizer.get_vocab())
        print(f"📊 Calculated meta_vocab_size: {meta_vocab_size}")
    except Exception as e:
        raise ImportError(
            "Please initialize the variable meta_vocab_size in the config.py file with the size of your vocabulary."
        ) from e

# ESM-2 Model Configuration
# Based on original ESM-2 architecture
# ESM-2のモデルサイズ: 8M, 35M, 150M, 650M, 3B, 15B
# ここでは学習可能なサイズとして small/medium/large を定義
if model_size == "small":
    # ESM-2 8M parameters equivalent
    model_config = EsmConfig(
        vocab_size=meta_vocab_size,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 32,
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 1,
        hidden_size=320,
        num_hidden_layers=6,
        num_attention_heads=20,
        intermediate_size=1280,
        max_position_embeddings=max_length + 2,  # +2 for BOS/EOS
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
    )
elif model_size == "medium":
    # ESM-2 35M parameters equivalent
    model_config = EsmConfig(
        vocab_size=meta_vocab_size,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 32,
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 1,
        hidden_size=480,
        num_hidden_layers=12,
        num_attention_heads=20,
        intermediate_size=1920,
        max_position_embeddings=max_length + 2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
    )
elif model_size == "large":
    # ESM-2 150M parameters equivalent
    model_config = EsmConfig(
        vocab_size=meta_vocab_size,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 32,
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 1,
        hidden_size=640,
        num_hidden_layers=30,
        num_attention_heads=20,
        intermediate_size=2560,
        max_position_embeddings=max_length + 2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
    )
else:
    raise ValueError(f"model_size: {model_size} is not supported. Choose between small, medium, and large")

print(f"🧬 ESM-2 Model Configuration ({model_size}):")
print(f"   - Vocab size: {meta_vocab_size}")
print(f"   - Max length: {max_length}")
print(f"   - Hidden size: {model_config.hidden_size}")
print(f"   - Layers: {model_config.num_hidden_layers}")
print(f"   - Attention heads: {model_config.num_attention_heads}")
print(f"   - Parameters: ~{model_config.hidden_size * model_config.num_hidden_layers * 12 // 1_000_000}M")

model = EsmForMaskedLM(config=model_config)

# Initialize wandb if enabled
wandb_run = None
if use_wandb:
    import wandb
    from datetime import datetime

    if wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"esm2-{model_size}-{timestamp}"

    # Determine dataset name from config
    dataset_name = config.get('dataset_name', 'protein_sequence')

    # Add metadata tags for experiment management
    tags = ['esm2', 'training', model_size, dataset_name]

    # Add experiment metadata to config
    experiment_config = {
        **config,
        'experiment_type': 'training',
        'model_type': 'esm2',
        'dataset_type': dataset_name,
        'model_size': model_size,
    }

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        config=experiment_config,
        tags=tags,
        resume="allow",
    )
    print(f"📊 Wandb initialized: {wandb_run.url}")

# Setup data collator
if "data_collator" in globals():
    print("Using custom data collator from config")
else:
    print("Using default DataCollatorForLanguageModeling")
    tokenizer_obj = globals().get("tokenizer", None)

    if tokenizer_obj is not None and hasattr(tokenizer_obj, "tokenizer"):
        actual_tokenizer = tokenizer_obj.tokenizer
    else:
        actual_tokenizer = tokenizer_obj

    if actual_tokenizer is None:
        raise ValueError("No tokenizer found in config. Please define 'tokenizer' in your config file.")

    # ESM-2: MLM probability 0.15 (BERT standard)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=actual_tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

# Training arguments
training_args = TrainingArguments(
    output_dir=model_path,
    logging_strategy="steps",
    logging_steps=log_interval,
    eval_strategy="steps",
    eval_steps=log_interval * 10,
    overwrite_output_dir=True,
    max_steps=max_steps,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    save_steps=save_steps,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,
    report_to="wandb" if use_wandb else "none",
    save_total_limit=3,
    load_best_model_at_end=False,
)

# Load datasets
print("📂 Loading datasets...")
if "use_custom_dataset_loader" in globals() and globals()["use_custom_dataset_loader"]:
    print("🧬 Using custom protein dataset loader")
    train_data_loader = ProteinDatasetLoader(dataset_dir, split="train", test_size=0.1)
    test_data_loader = ProteinDatasetLoader(dataset_dir, split="test", test_size=0.1)
    train_dataset = train_data_loader.get_dataset()
    test_dataset = test_data_loader.get_dataset()
else:
    print("📂 Using standard HuggingFace dataset loading")
    dataset_path = Path(dataset_dir)

    # Try to load from arrow format
    train_arrow = dataset_path / "train.arrow"
    test_arrow = dataset_path / "test.arrow"
    valid_arrow = dataset_path / "valid.arrow"

    if train_arrow.exists():
        print(f"Loading from arrow format: {train_arrow}")
        train_dataset = load_from_disk(str(train_arrow))
        if test_arrow.exists():
            test_dataset = load_from_disk(str(test_arrow))
        elif valid_arrow.exists():
            test_dataset = load_from_disk(str(valid_arrow))
        else:
            raise FileNotFoundError(f"No test or valid split found in {dataset_path}")
    else:
        # Fall back to standard format
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset["train"]
        test_split_name = "test" if "test" in dataset else "valid"
        test_dataset = dataset[test_split_name]

# Limit test dataset size for faster evaluation
if len(test_dataset) > 5000:
    test_dataset = test_dataset.select(range(5000))
    print("📊 Limited test dataset to 5000 samples for faster evaluation")

# Apply preprocessing function if defined in config
if "preprocess_function" in globals() and callable(globals()["preprocess_function"]):
    print("🔄 Applying preprocessing function...")
    train_dataset = train_dataset.map(globals()["preprocess_function"], batched=True)
    test_dataset = test_dataset.map(globals()["preprocess_function"], batched=True)
    print("✅ Preprocessing completed")
    print(f"   Train dataset columns: {train_dataset.column_names}")
    print(f"   Test dataset columns: {test_dataset.column_names}")

    # Verify
    sample = train_dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    if "attention_mask" in sample:
        print("   ✓ attention_mask successfully added")
else:
    print("ℹ️  No preprocessing function found")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Resume from checkpoint if available
resume_checkpoint = None
if os.path.exists(model_path):
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(model_path, checkpoints[-1])
        resume_checkpoint = latest_checkpoint
        print(f"✅ Found {len(checkpoints)} checkpoint(s)")
        print(f"   Resuming training from: {latest_checkpoint}")
    else:
        print(f"ℹ️  No checkpoints found in {model_path}")
        print("   Starting training from scratch...")
else:
    print(f"ℹ️  Output directory {model_path} does not exist")
    print("   Starting training from scratch...")

# Train
print("🚀 Starting ESM-2 training...")
try:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("✅ Training completed successfully!")
except Exception as e:
    if "checkpoint" in str(e).lower() and resume_checkpoint:
        print(f"⚠️  Failed to resume from checkpoint: {e}")
        print("   Starting training from scratch instead...")
        trainer.train(resume_from_checkpoint=None)
    else:
        raise
finally:
    if wandb_run is not None:
        wandb_run.finish()
        print("📊 Wandb run finished.")

print("🎉 ESM-2 training script completed!")
