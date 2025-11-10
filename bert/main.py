from transformers import (
    BertConfig,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, Dataset
import os
import json
import pyarrow as pa
from pathlib import Path


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
                print(
                    f"📁 Found {len(arrow_files)} arrow files: {[f.name for f in arrow_files]}"
                )

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
                    print(
                        f"📊 Combined {len(all_batches)} tables: {len(combined_table)} total rows"
                    )

                    # Convert PyArrow table to pandas DataFrame, then to HuggingFace Dataset
                    df = combined_table.to_pandas()
                    print(f"📋 Converted to pandas DataFrame: {len(df)} rows")

                    # Convert numpy arrays to lists for HuggingFace compatibility
                    if "token" in df.columns:
                        df["token"] = df["token"].apply(
                            lambda x: x.tolist() if hasattr(x, "tolist") else x
                        )

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
        if (
            hasattr(self.dataset, "keys")
            and isinstance(self.dataset, dict)
            and "train" in self.dataset
        ):
            # Already has splits
            if split == "train":
                self.data = self.dataset["train"]
            elif split in ["valid", "val", "test"]:
                self.data = self.dataset.get(
                    "valid", self.dataset.get("test", self.dataset["train"])
                )
        else:
            # Create splits
            if test_size > 0:
                split_dataset = self.dataset.train_test_split(
                    test_size=test_size, seed=42
                )
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

model_size = None
use_custom_rna_dataset = False
tokenizer = None

model_path = ""
max_length = 1024
dataset_dir = ""
learning_rate = 6e-6
weight_decay = 1e-1
warmup_steps = 200
max_steps = 600000
batch_size = 10

gradient_accumulation_steps = 5 * 8
per_device_eval_batch_size = 1
log_interval = 100
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# Handle configurator path
configurator_path = (
    "bert/configurator.py"
    if os.path.exists("bert/configurator.py")
    else "configurator.py"
)
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
    model_config = BertConfig(
        vocab_size=meta_vocab_size, max_position_embeddings=max_length
    )
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
else:
    raise ValueError(
        "model_size: {model_size} is not supported choose between small, medium and large"
    )

model = BertForMaskedLM(config=model_config)

# Use custom data collator if defined in config, otherwise use default
if "data_collator" in globals():
    print("Using custom data collator from config")
    # data_collator is already defined in the config file
else:
    print("Using default DataCollatorForLanguageModeling")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=globals()["tokenizer"], mlm=True, mlm_probability=0.2
    )

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
    save_steps=1000,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)


# Check if we should use custom dataset loading (for RNA data)
if "use_custom_rna_dataset" in globals() and use_custom_rna_dataset:
    print("🧬 Using custom RNA dataset loader")

    # Get vocab file path if available
    vocab_file_path = globals().get("rna_vocab_file", None)

    # Load training and test datasets using custom loader
    train_data_loader = RNADatasetForBERT(
        dataset_dir, split="train", vocab_file=vocab_file_path, test_size=0.1
    )
    test_data_loader = RNADatasetForBERT(
        dataset_dir, split="test", vocab_file=vocab_file_path, test_size=0.1
    )

    train_dataset = train_data_loader.get_dataset()
    test_dataset = test_data_loader.get_dataset()

    # Limit test dataset size for faster evaluation
    if len(test_dataset) > 10000:
        test_dataset = test_dataset.select(range(10000))
        print("📊 Limited test dataset to 10000 samples for faster evaluation")
else:
    print("📂 Using standard HuggingFace dataset loading")
    train_dataset = load_from_disk(dataset_dir)["train"]
    test_dataset = load_from_disk(dataset_dir)["test"].select(
        range(min(10000, load_from_disk(dataset_dir)["test"].num_rows))
    )  # for testing purposes, select only 10000 samples

# Apply preprocessing for RNA data if using custom dataset
if "use_custom_rna_dataset" in globals() and globals().get(
    "use_custom_rna_dataset", False
):
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
    train_dataset = train_dataset.map(
        preprocess_rna_for_bert, batched=True, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        preprocess_rna_for_bert, batched=True, remove_columns=test_dataset.column_names
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
    train_dataset = train_dataset.map(globals()["preprocess_function"], batched=True)
    test_dataset = test_dataset.map(globals()["preprocess_function"], batched=True)
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


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
