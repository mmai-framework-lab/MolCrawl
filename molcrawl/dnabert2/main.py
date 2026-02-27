"""
DNABERT-2 Training Script for Genome Sequence Data

DNABERT-2は、DNA配列解析に特化したBERTベースモデルです。
主な改良点：
- BPE (Byte Pair Encoding) トークナイゼーション（k-mer不要）
- より効率的なアテンション機構
- DNA特有の特性を考慮したアーキテクチャ

参考: DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome
https://github.com/MAGICS-LAB/DNABERT_2
"""

import os
from pathlib import Path

import pyarrow as pa

try:
    from datasets import Dataset, load_from_disk
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    pass  # Not available in documentation-only environments (e.g. pdoc)


class DNADatasetLoader:
    """
    DNA配列データセット用のローダー

    既存のgenome_sequenceデータセットを読み込み、
    DNABERT-2用に前処理を行います。
    """

    def __init__(self, data_dir, split="train", test_size=0.1):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size

        print(f"📂 Loading DNA dataset from {data_dir}")

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

if __name__ == "__main__":
    model_size = None
    tokenizer = None

    # wandb settings
    use_wandb = os.environ.get("USE_WANDB", "False").lower() in ("true", "1", "yes")
    wandb_project = os.environ.get("WANDB_PROJECT", "dnabert2-training")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    wandb_log_model = os.environ.get("WANDB_LOG_MODEL", "True").lower() in (
        "true",
        "1",
        "yes",
    )

    model_path = ""
    max_length = 512  # DNABERT-2 default: 512 (より長い配列の場合は増やす)
    dataset_dir = ""
    learning_rate = 3e-5  # DNABERT-2推奨値
    weight_decay = 0.01
    warmup_steps = 10000
    max_steps = 200000
    batch_size = 16
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 8
    log_interval = 100
    save_steps = 5000

    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]

    # Load config from file
    configurator_path = "dnabert2/configurator.py" if os.path.exists("dnabert2/configurator.py") else "configurator.py"
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

    # DNABERT-2 Model Configuration
    # Based on original DNABERT-2 architecture but adapted for different sizes
    if model_size == "small":
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
    elif model_size == "medium":
        model_config = BertConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    elif model_size == "large":
        model_config = BertConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length,
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=20,
            intermediate_size=5120,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    else:
        raise ValueError(f"model_size: {model_size} is not supported. Choose between small, medium, and large")

    print(f"🧬 DNABERT-2 Model Configuration ({model_size}):")
    print(f"   - Vocab size: {meta_vocab_size}")
    print(f"   - Max length: {max_length}")
    print(f"   - Hidden size: {model_config.hidden_size}")
    print(f"   - Layers: {model_config.num_hidden_layers}")
    print(f"   - Attention heads: {model_config.num_attention_heads}")

    model = BertForMaskedLM(config=model_config)

    # Initialize wandb if enabled
    wandb_run = None
    if use_wandb:
        import wandb
        from datetime import datetime

        if wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"dnabert2-{model_size}-{timestamp}"

        # Determine dataset name from config
        dataset_name = config.get("dataset_name", "genome_sequence")

        # Add metadata tags for experiment management
        tags = ["dnabert2", "training", model_size, dataset_name]

        # Add experiment metadata to config
        experiment_config = {
            **config,
            "experiment_type": "training",
            "model_type": "dnabert2",
            "dataset_type": dataset_name,
            "model_size": model_size,
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

        # DNABERT-2: MLM probability 0.15 (BERT standard)
        data_collator = DataCollatorForLanguageModeling(tokenizer=actual_tokenizer, mlm=True, mlm_probability=0.15)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        logging_strategy="steps",
        logging_steps=log_interval,
        eval_strategy="steps",
        eval_steps=log_interval * 10,  # Evaluate less frequently for efficiency
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,  # Enable mixed precision training for efficiency
        dataloader_num_workers=4,  # Parallel data loading
        report_to="wandb" if use_wandb else "none",
        save_total_limit=3,  # Keep only 3 most recent checkpoints
        load_best_model_at_end=False,  # Don't load best model (saves memory)
    )

    # Load datasets
    print("📂 Loading datasets...")
    if "use_custom_dataset_loader" in globals() and globals()["use_custom_dataset_loader"]:
        print("🧬 Using custom DNA dataset loader")
        train_data_loader = DNADatasetLoader(dataset_dir, split="train", test_size=0.1)
        test_data_loader = DNADatasetLoader(dataset_dir, split="test", test_size=0.1)
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
    if len(test_dataset) > 5000:  # type: ignore[used-before-def]
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
    print("🚀 Starting DNABERT-2 training...")
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

    print("🎉 DNABERT-2 training script completed!")
