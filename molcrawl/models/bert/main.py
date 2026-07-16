import json
import os
from pathlib import Path

import pyarrow as pa

try:
    from datasets import Dataset, load_from_disk
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    pass  # Not available in documentation-only environments (e.g. pdoc)

# Allow numpy globals embedded in HF Trainer rng_state.pth to deserialize
# under torch >= 2.6 (default weights_only=True). See molcrawl/core/torch_compat.
from molcrawl.core.torch_compat import enable_full_torch_load

enable_full_torch_load()


class RNADatasetForBERT:
    """Custom dataset class for loading RNA data"""

    def __init__(self, data_dir, split="train", vocab_file=None, test_size=0.1):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size

        print(f"📂 Attempting to load BERT data from {data_dir}")

        # Load vocabulary if provided
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
            print(f"📖 Loaded vocabulary: {len(self.vocab)} tokens")
        else:
            print("⚠️ No vocabulary file provided")
            self.vocab = None

        # Load data from Arrow files
        try:
            arrow_files = list(Path(data_dir).glob("*.arrow"))
            if arrow_files:
                print(f"📁 Found {len(arrow_files)} arrow files: {[f.name for f in arrow_files]}")

                all_batches = []
                for arrow_file in arrow_files:
                    try:
                        print(f"📖 Reading {arrow_file.name}...")
                        with pa.memory_map(str(arrow_file), "r") as mmap:
                            with pa.ipc.open_stream(mmap) as reader:
                                table = reader.read_all()
                                print(f"🔢 Read table via stream: {len(table)} rows")
                                all_batches.append(table)
                    except Exception as e:
                        print(f"❌ Failed to read {arrow_file.name}: {e}")
                        continue

                if all_batches:
                    # Combine all tables
                    combined_table = pa.concat_tables(all_batches)
                    print(f"📊 Combined {len(all_batches)} tables: {len(combined_table)} total rows")

                    # Convert PyArrow table to pandas DataFrame, then to HuggingFace Dataset
                    df = combined_table.to_pandas()
                    print(f"📋 Converted to pandas DataFrame: {len(df)} rows")

                    # Convert numpy arrays to lists for HuggingFace compatibility
                    if "token" in df.columns:
                        df["token"] = df["token"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)

                    # Create dataset from pandas DataFrame (bypasses metadata issues)
                    self.dataset = Dataset.from_pandas(df)
                    print("✅ Created HuggingFace Dataset from pandas DataFrame")
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
            # Already has splits
            if split == "train":
                self.data = self.dataset["train"]
            elif split in ["valid", "val", "test"]:
                self.data = self.dataset.get("valid", self.dataset.get("test", self.dataset["train"]))
        else:
            # Create splits
            if test_size > 0:
                split_dataset = self.dataset.train_test_split(test_size=test_size, seed=42)
                if split == "train":
                    self.data = split_dataset["train"]
                elif split in ["valid", "val", "test"]:
                    self.data = split_dataset["test"]
            else:
                self.data = self.dataset

        print(f"Loaded {len(self.data)} samples for {split}")

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


if __name__ == "__main__":
    # PyTorch >= 2.6 changed the default of torch.load to weights_only=True.
    # Older HuggingFace checkpoints (optimizer states, RNG states) contain
    # arbitrary Python objects that cannot be enumerated upfront.  Patch
    # torch.load to restore the pre-2.6 behaviour for this process only.
    # This is safe because we only load checkpoints from our own training runs.
    try:
        import torch as _torch

        _orig_torch_load = _torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        _torch.load = _patched_torch_load
        del _torch  # _orig_torch_load must NOT be deleted — the closure captures it by name
    except Exception:
        pass

    model_size = None
    use_custom_rna_dataset = False
    tokenizer = None

    # wandb settings (can be overridden by environment variables, config file, or command line args)
    use_wandb = os.environ.get("USE_WANDB", "False").lower() in (
        "true",
        "1",
        "yes",
    )  # log training metrics to wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "bert-training")  # wandb project name
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)  # wandb run name (None = auto-generate)
    wandb_entity = os.environ.get("WANDB_ENTITY", None)  # wandb entity/team name (None = default)
    wandb_log_model = os.environ.get("WANDB_LOG_MODEL", "False").lower() in (
        "true",
        "1",
        "yes",
    )  # log model checkpoints as wandb artifacts

    model_path = ""
    pretrain_model_path = ""  # Path to pretraining checkpoint dir; used when no fine-tune checkpoint exists
    max_length = 1024
    dataset_dir = ""
    learning_rate = 6e-6
    weight_decay = 1e-2  # BERT default (production spec 2026-07-08). Applied 2D-only via _WeightDecayNoEmbedTrainer.
    warmup_steps = 200
    max_steps = 60000
    batch_size = 10

    gradient_accumulation_steps = 5 * 8
    per_device_eval_batch_size = 8
    log_interval = 100
    save_steps = 1000  # Default value, can be overridden in config
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
    # Handle configurator path (support repo-root invocation and direct invocation)
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(_this_dir, "configurator.py")):
        configurator_path = os.path.join(_this_dir, "configurator.py")
    elif os.path.exists("src/bert/configurator.py"):
        configurator_path = "src/bert/configurator.py"
    elif os.path.exists("bert/configurator.py"):
        configurator_path = "bert/configurator.py"
    else:
        configurator_path = "configurator.py"
    exec(open(configurator_path).read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    if not ("meta_vocab_size" in vars() and "meta_vocab_size" in globals()):
        try:
            meta_vocab_size = (len(tokenizer) // 8 + 1) * 8
        except Exception as e:
            raise ImportError(
                "Please initialize the variable meta_vocab_size in the *_config.py file with the size of your vocabulary."
            ) from e

    if model_size == "small":
        model_config = BertConfig(vocab_size=meta_vocab_size, max_position_embeddings=max_length)
    elif model_size == "medium":
        # Note that this would be bert-large but the size is equivalent to gpt2-medium so we name it medium here as well
        model_config = BertConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length,
            hidden_size=1024,  # Dimensionality of the encoder layers and the pooler layer
            num_hidden_layers=24,  # Number of hidden layers in the Transformer encoder
            num_attention_heads=16,  # Number of attention heads
            intermediate_size=4096,  # Dimensionality of the "intermediate" (feed-forward) layer
        )
    elif model_size == "large":
        # Custom config to match the size of gpt2 large model.
        model_config = BertConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length,
            hidden_size=1152,  # Hidden layer size
            num_hidden_layers=36,  # Number of transformer layers
            num_attention_heads=18,  # Number of attention heads
            intermediate_size=4608,  # Size of intermediate (feed-forward) layer
        )
    elif model_size == "xl":
        # Custom config matching GPT-2 XL scale (n_embd=1600, n_layer=48,
        # n_head=25). hidden / num_attention_heads = 64 per head,
        # intermediate = 4 × hidden. ~1.5B params.
        model_config = BertConfig(
            vocab_size=meta_vocab_size,
            max_position_embeddings=max_length,
            hidden_size=1600,
            num_hidden_layers=48,
            num_attention_heads=25,
            intermediate_size=6400,
        )
    else:
        raise ValueError(
            f"model_size '{model_size}' is not supported. "
            "Choose: small, medium, large, xl"
        )

    # Opt-in: enable Flash Attention 2 implementation if the per-config
    # global ``flash_attention`` is True. Requires the ``flash-attn``
    # package; falls back to the model's default attention if the
    # config / hardware combination is unsupported (HF will raise at
    # ``from_config`` / ``from_pretrained`` time).
    if globals().get("flash_attention", False):
        model_config._attn_implementation = "flash_attention_2"

    # Determine whether fine-tune checkpoints already exist in model_path
    _has_finetune_ckpt = os.path.exists(model_path) and any(d.startswith("checkpoint-") for d in os.listdir(model_path))

    if _has_finetune_ckpt:
        # Trainer.train(resume_from_checkpoint=...) will load weights; just
        # need the architecture here.
        print(f"Fine-tune checkpoint found in {model_path} — will resume.")
        model = BertForMaskedLM(config=model_config)
    elif pretrain_model_path and os.path.exists(pretrain_model_path):
        # Load weights from pretraining checkpoint.
        # Find the latest checkpoint-N sub-directory if present.
        _ckpt_dirs = sorted(
            [d for d in os.listdir(pretrain_model_path) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        _load_path = os.path.join(pretrain_model_path, _ckpt_dirs[-1]) if _ckpt_dirs else pretrain_model_path
        print(f"Loading pretraining weights from {_load_path}")
        model = BertForMaskedLM.from_pretrained(_load_path, config=model_config, ignore_mismatched_sizes=True)
    else:
        if pretrain_model_path:
            raise ValueError(
                f"pretrain_model_path={pretrain_model_path!r} is set but does not exist.\n"
                "Run pretraining first, or unset pretrain_model_path to train from scratch."
            )
        model = BertForMaskedLM(config=model_config)

    # Initialize wandb if enabled
    wandb_run = None
    if use_wandb:
        from datetime import datetime

        import wandb

        # Generate run name if not provided
        if wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"bert-{model_size}-{timestamp}"

        # Determine dataset name from config
        dataset_name = config.get("dataset_name", "unknown")

        # Add metadata tags for experiment management
        tags = ["bert", "training", model_size, dataset_name]

        # Add experiment metadata to config
        experiment_config = {
            **config,
            "experiment_type": "training",
            "model_type": "bert",
            "dataset_type": dataset_name,
            "model_size": model_size,
        }

        # Initialize wandb
        wandb_run = wandb.init(  # type: ignore[attr-defined]
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=experiment_config,
            tags=tags,
            resume="allow",  # Allow resuming if run exists
        )
        print(f"Wandb initialized: {wandb_run.url}")

    # Use custom data collator if defined in config, otherwise use the
    # ambiguity-aware default. The wrapper falls back to a plain
    # DataCollatorForLanguageModeling when no ambiguous tokens are configured
    # for the modality (compounds / rna / molecule_nat_lang).
    if "data_collator" in globals():
        print("Using custom data collator from config")
        # data_collator is already defined in the config file
    else:
        # Get the tokenizer from globals
        tokenizer_obj = globals().get("tokenizer", None)

        # If tokenizer is a wrapper class (like MoleculeNatLangTokenizer), extract the actual tokenizer
        if tokenizer_obj is not None and hasattr(tokenizer_obj, "tokenizer"):
            actual_tokenizer = tokenizer_obj.tokenizer
        else:
            actual_tokenizer = tokenizer_obj

        # Verify we have a valid tokenizer
        if actual_tokenizer is None:
            raise ValueError("No tokenizer found in config. Please define 'tokenizer' in your config file.")

        from molcrawl.models._collators import (
            ambiguous_tokens_for_modality,
            infer_modality_from_path,
            make_mlm_collator,
        )

        _modality = infer_modality_from_path(globals().get("dataset_dir")) \
            or infer_modality_from_path(globals().get("model_path"))
        _ambig = ambiguous_tokens_for_modality(_modality) if _modality else []
        print(
            f"Using {'AmbiguityAwareMLMCollator' if _ambig else 'DataCollatorForLanguageModeling'} "
            f"(modality={_modality!r}, ambiguous_tokens={_ambig})"
        )
        data_collator = make_mlm_collator(
            actual_tokenizer, ambiguous_tokens=_ambig, mlm_probability=0.2
        )

    # Early stopping configuration
    early_stopping = globals().get("early_stopping", True)  # Enable by default
    early_stopping_patience = globals().get("early_stopping_patience", 10)  # Default patience
    if early_stopping:
        print(f"⏰ Early stopping enabled with patience={early_stopping_patience}")
    else:
        print("⏰ Early stopping disabled")

    training_args = TrainingArguments(
        output_dir=model_path,  # output directory to where save model checkpoint
        logging_strategy="steps",  # log every `logging_steps`
        logging_steps=log_interval,  # log every 1000 steps
        eval_strategy="steps",  # evaluate each `logging_steps` steps
        eval_steps=log_interval,  # evaluate every log_interval steps
        overwrite_output_dir=True,
        max_steps=max_steps,  # number of training epochs, feel free to tweak
        per_device_train_batch_size=batch_size,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=gradient_accumulation_steps,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=per_device_eval_batch_size,  # evaluation batch size
        save_steps=save_steps,  # Use save_steps from config
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # AdamW optimizer settings — production spec (2026-07-08):
        # betas = (0.9, 0.95) instead of HF default (0.9, 0.999).
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,  # grad clip = 1.0 (production spec)
        report_to="none",  # Disable wandb integration to prevent artifact bloat
        load_best_model_at_end=early_stopping,  # Load best model at end when early stopping is enabled
        metric_for_best_model="eval_loss",  # Use eval_loss to determine best model
        greater_is_better=False,  # Lower loss is better
        save_total_limit=5,  # Keep only the 5 most recent checkpoints
        # save_safetensors=True is the HF default; tied weights are restored
        # at resume time by molcrawl.core.utils.trainer_utils.install_tie_weights_on_resume.
        # Setting this to False would break resume from existing safetensors-only
        # checkpoints (HF Trainer 4.45 _load_from_checkpoint requires args.save_safetensors=True
        # to enter the safetensors load branch).
        save_safetensors=True,
        # Performance opt-ins (read from the per-config globals so existing
        # configs are unaffected; subset configs can flip these on individually).
        #   bf16=True                      : bf16 mixed precision on Hopper/Blackwell
        #   tf32=True                      : TF32 matmuls on Ampere+ (GB200/Blackwell)
        #   dataloader_num_workers=N       : multi-process input pipeline
        #   dataloader_pin_memory=True     : pinned CPU buffers for H2D transfer
        #   ddp_find_unused_parameters     : BERT uses every parameter each step,
        #                                    so leave False to skip the DDP scan.
        bf16=bool(globals().get("bf16", False)),
        tf32=bool(globals().get("tf32", False)),
        dataloader_num_workers=int(globals().get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(globals().get("dataloader_pin_memory", False)),
        ddp_find_unused_parameters=bool(globals().get("ddp_find_unused_parameters", False)),
        # XL-scale opt-ins (defaults preserve existing config behaviour):
        #   torch_compile=True               : inductor kernel fusion,
        #                                      ~20-40% speedup after warmup
        #   optim="adamw_torch_fused"        : fused CUDA AdamW kernel,
        #                                      ~10-20% optimizer step speedup
        #   dataloader_persistent_workers=T  : skip per-epoch worker
        #                                      respawn cost
        #   ddp_bucket_cap_mb=512            : larger AllReduce buckets
        #                                      (HF default 25 MB) → better
        #                                      8-GPU comms efficiency
        torch_compile=bool(globals().get("torch_compile", False)),
        torch_compile_backend=str(globals().get("torch_compile_backend", "inductor")),
        optim=str(globals().get("optim", "adamw_torch")),
        dataloader_persistent_workers=bool(globals().get("dataloader_persistent_workers", False)),
        ddp_bucket_cap_mb=int(globals().get("ddp_bucket_cap_mb", 25)),
    )

    # Check if we should use custom dataset loading (for RNA data)
    if "use_custom_rna_dataset" in globals() and use_custom_rna_dataset:
        print("🧬 Using custom RNA dataset loader")

        # Get vocab file path if available
        vocab_file_path = globals().get("rna_vocab_file", None)

        # Load training and test datasets using custom loader
        train_data_loader = RNADatasetForBERT(dataset_dir, split="train", vocab_file=vocab_file_path, test_size=0.1)
        test_data_loader = RNADatasetForBERT(dataset_dir, split="test", vocab_file=vocab_file_path, test_size=0.1)

        train_dataset = train_data_loader.get_dataset()
        test_dataset = test_data_loader.get_dataset()

        # Limit test dataset size for faster evaluation
        if len(test_dataset) > 10000:
            test_dataset = test_dataset.select(range(10000))
            print("📊 Limited test dataset to 10000 samples for faster evaluation")
    else:
        print("📂 Using standard HuggingFace dataset loading")
        from pathlib import Path

        dataset_path = Path(dataset_dir)

        # Try new multi-dataset loader first (for compounds)
        train_dataset = None
        test_dataset = None

        try:
            # Check if this is a compounds dataset by looking at the parent directory structure
            if "compounds" in str(dataset_path):
                from molcrawl.data.compounds.dataset.multi_loader import MultiDatasetLoader

                # Determine compounds directory
                compounds_dir = dataset_path
                if dataset_path.name in [
                    "train",
                    "test",
                    "valid",
                    "train.arrow",
                    "test.arrow",
                    "valid.arrow",
                ]:
                    compounds_dir = dataset_path.parent

                loader = MultiDatasetLoader(compounds_dir)
                available = loader.get_available_datasets()

                if available:
                    print(f"📊 Found {len(available)} available compound datasets: {[d.value for d in available]}")

                    # Load and combine all available datasets
                    dataset_dict = loader.load_datasets(combine=True)

                    train_dataset = dataset_dict.get("train")
                    test_dataset = dataset_dict.get("valid") or dataset_dict.get("test")

                    if train_dataset and test_dataset:
                        print(f"✓ Loaded combined datasets: train={len(train_dataset)}, test={len(test_dataset)}")
                    else:
                        print("⚠ Multi-loader succeeded but missing splits, falling back to legacy loader")
                        train_dataset = None
                        test_dataset = None
        except Exception as e:
            print(f"⚠ Multi-loader failed ({e}), falling back to legacy single-dataset loader")

        # Legacy loader (if multi-loader failed or not applicable)
        if train_dataset is None or test_dataset is None:
            # Try to load from arrow format (with .arrow suffix)
            train_arrow = dataset_path / "train.arrow"
            test_arrow = dataset_path / "test.arrow"
            valid_arrow = dataset_path / "valid.arrow"

            if train_arrow.exists():
                print(f"Loading from arrow format: {train_arrow}")
                train_dataset = load_from_disk(str(train_arrow))
                # Try test first, fall back to valid
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
        if len(test_dataset) > 10000:
            test_dataset = test_dataset.select(range(10000))
            print("📊 Limited test dataset to 10000 samples for faster evaluation")

    # Apply preprocessing for RNA data if using custom dataset
    if "use_custom_rna_dataset" in globals() and globals().get("use_custom_rna_dataset", False):
        print("🧬 Applying RNA-specific preprocessing...")

        def preprocess_rna_for_bert(examples):
            """Convert RNA token data to BERT format"""
            input_ids = []
            attention_masks = []

            for tokens in examples["token"]:
                # Ensure tokens is a list
                if not isinstance(tokens, list):
                    tokens = tokens.tolist() if hasattr(tokens, "tolist") else [tokens]

                # Truncate or pad to max_length
                if len(tokens) > max_length - 2:  # -2 for [CLS] and [SEP]
                    tokens = tokens[: max_length - 2]

                # Add [CLS] at beginning and [SEP] at end (using token IDs)
                # Note: These should be actual token IDs from vocabulary
                cls_token_id = 1  # Assuming CLS token ID
                sep_token_id = 2  # Assuming SEP token ID

                input_id = [cls_token_id] + tokens + [sep_token_id]
                attention_mask = [1] * len(input_id)

                # Pad to max_length
                while len(input_id) < max_length:
                    input_id.append(0)  # PAD token ID
                    attention_mask.append(0)

                input_ids.append(input_id)
                attention_masks.append(attention_mask)

            return {"input_ids": input_ids, "attention_mask": attention_masks}

        print("🔄 Mapping preprocessing function to datasets...")
        # Use parallel preprocessing when configured
        preprocess_num_proc: int = int(globals().get("preprocess_num_proc", 1))
        train_dataset = train_dataset.map(
            preprocess_rna_for_bert,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=preprocess_num_proc,
        )
        test_dataset = test_dataset.map(
            preprocess_rna_for_bert,
            batched=True,
            remove_columns=test_dataset.column_names,
            num_proc=preprocess_num_proc,
        )

        print("✅ RNA preprocessing completed.")
        print("Train dataset columns after preprocessing:", train_dataset.column_names)
        print("Test dataset columns after preprocessing:", test_dataset.column_names)

        # Verify format
        sample = train_dataset[0]
        print("Sample keys:", list(sample.keys()))
        print(f"Sample input_ids length: {len(sample['input_ids'])}")
        print(f"Sample attention_mask length: {len(sample['attention_mask'])}")

    # Apply preprocessing function if it exists in config (for non-RNA datasets)
    elif "preprocess_function" in globals() and callable(globals()["preprocess_function"]):
        print("Applying preprocessing function to add attention_mask...")
        # Use parallel preprocessing when configured
        preprocess_num_proc = int(globals().get("preprocess_num_proc", 1))
        train_dataset = train_dataset.map(globals()["preprocess_function"], batched=True, num_proc=preprocess_num_proc)
        test_dataset = test_dataset.map(globals()["preprocess_function"], batched=True, num_proc=preprocess_num_proc)
        print("Preprocessing completed.")
        print("Train dataset columns after preprocessing:", train_dataset.column_names)
        print("Test dataset columns after preprocessing:", test_dataset.column_names)

        # Verify attention_mask was added
        sample = train_dataset[0]
        print("Sample keys:", list(sample.keys()))
        if "attention_mask" in sample:
            print("✓ attention_mask successfully added")
        else:
            print("✗ attention_mask not found after preprocessing")
    else:
        print("No preprocessing function found or not callable.")

    # Configure callbacks
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"📋 Added EarlyStoppingCallback (patience={early_stopping_patience})")

    # HF Trainer's default get_decay_parameter_names excludes bias and LayerNorm
    # from the weight-decay group but INCLUDES nn.Embedding.weight (word_embeddings,
    # position_embeddings, token_type_embeddings for BERT). Production spec (2026-07-08)
    # requires weight decay be applied to 2D matmul weights only — bias, LayerNorm
    # and embeddings all get wd=0. Subclass Trainer to also drop embedding names.
    import torch as _torch

    class _WeightDecayNoEmbedTrainer(Trainer):
        """Trainer that additionally excludes nn.Embedding.weight from weight decay."""

        def get_decay_parameter_names(self, model):
            names = super().get_decay_parameter_names(model)
            embedding_names = set()
            for module_name, module in model.named_modules():
                if isinstance(module, _torch.nn.Embedding):
                    for pn, _ in module.named_parameters(recurse=False):
                        embedding_names.add(f"{module_name}.{pn}" if module_name else pn)
            return [n for n in names if n not in embedding_names]

    trainer = _WeightDecayNoEmbedTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks if callbacks else None,
    )

    # Resume from checkpoint by default if available
    # Check if checkpoints exist in output directory
    resume_checkpoint = None
    if os.path.exists(model_path):
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        if checkpoints:
            # Find the latest checkpoint
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(model_path, checkpoints[-1])
            resume_checkpoint = latest_checkpoint
            print(f"✅ Found {len(checkpoints)} checkpoint(s) in {model_path}")
            print(f"   Resuming training from: {latest_checkpoint}")
        else:
            print(f"ℹ️  No checkpoints found in {model_path}")
            print("   Starting training from scratch...")
    else:
        print(f"ℹ️  Output directory {model_path} does not exist")
        print("   Starting training from scratch...")

    # Train with or without checkpoint
    # If resume_checkpoint is None, training starts from scratch
    # If resume_checkpoint is provided, training resumes from that checkpoint
    from molcrawl.core.utils.trainer_utils import install_tie_weights_on_resume
    install_tie_weights_on_resume(trainer)
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as e:
        if resume_checkpoint and (
            "checkpoint" in str(e).lower() or "weights only" in str(e).lower() or "weightsunpickler" in str(e).lower()
        ):
            print(f"⚠️  Failed to resume from checkpoint: {e}")
            # Re-initialise the model with pretrain weights (if available) so
            # we don't accidentally train from random weights.
            if pretrain_model_path and os.path.exists(pretrain_model_path):
                _ckpt_dirs = sorted(
                    [d for d in os.listdir(pretrain_model_path) if d.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[1]),
                )
                _load_path = os.path.join(pretrain_model_path, _ckpt_dirs[-1]) if _ckpt_dirs else pretrain_model_path
                print(f"   Reloading pretrain weights from {_load_path} and restarting fine-tuning...")
                trainer.model = BertForMaskedLM.from_pretrained(_load_path, config=model_config, ignore_mismatched_sizes=True)
            else:
                print("   No pretrain_model_path available — starting from current model weights...")
            trainer.train(resume_from_checkpoint=None)
        else:
            raise
    finally:
        # Finish wandb run
        if wandb_run is not None:
            wandb_run.finish()
            print("Wandb run finished.")
