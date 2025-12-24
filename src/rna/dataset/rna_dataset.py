"""
RNA Transcriptome Dataset
"""

import os
import json
from pathlib import Path

import torch
import pyarrow as pa
from datasets import load_from_disk, Dataset


class RNADataset:
    """RNA Transcriptome Dataset"""

    def __init__(self, data_dir, split="train", vocab_file=None, test_size=0.1):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size

        # Load vocabulary
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)
        else:
            # Default RNA vocabulary
            self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
            self.vocab_size = 3

        # Load dataset - direct arrow file reading to bypass metadata issues
        print(f"📂 Attempting to load data from {data_dir}")

        try:
            data_path = Path(data_dir)
            arrow_files = sorted(list(data_path.glob("*.arrow")))

            if arrow_files:
                print(f"📁 Found {len(arrow_files)} arrow files: {[f.name for f in arrow_files]}")

                all_batches = []
                for arrow_file in arrow_files:
                    print(f"📖 Reading {arrow_file.name}...")
                    try:
                        # Try as memory mapped stream first
                        with pa.memory_map(str(arrow_file)) as mmap:
                            with pa.ipc.open_stream(mmap) as reader:
                                table = reader.read_all()
                                print(f"✓ Read table via stream: {len(table)} rows")
                                all_batches.append(table)
                    except Exception:
                        try:
                            # Fallback to RecordBatch file
                            with pa.memory_map(str(arrow_file)) as mmap:
                                with pa.ipc.open_file(mmap) as reader:
                                    table = reader.read_all()
                                    print(f"✓ Read table via file: {len(table)} rows")
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
            # Fallback to other methods
            try:
                print("🔄 Trying HuggingFace format as fallback...")
                self.dataset = load_from_disk(data_dir)
                print(f"✅ Loaded HuggingFace dataset from {data_dir}")
            except Exception as e2:
                print(f"❌ All loading methods failed: {e2}")
                raise FileNotFoundError(f"Could not load data from {data_dir}")

        # Split into train/valid if needed
        if hasattr(self.dataset, "keys") and isinstance(self.dataset, dict) and "train" in self.dataset:
            # Already has splits
            if split == "train":
                self.data = self.dataset["train"]
            elif split == "valid" or split == "val":
                self.data = self.dataset.get("valid", self.dataset.get("test", self.dataset["train"]))
        else:
            # Single dataset, need to split
            total_size = len(self.dataset)
            if split == "train":
                self.data = self.dataset.select(range(int(total_size * (1 - self.test_size))))
            else:  # valid
                self.data = self.dataset.select(range(int(total_size * (1 - self.test_size)), total_size))

        print(f"Loaded {len(self.data)} samples for {split}")

        # Sample a few examples to understand data structure
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, (list, str)):
                    print(f"  {key}: {type(value)} of length {len(value)}")
                else:
                    print(f"  {key}: {type(value)} = {value}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # RNA data has 'token' column with numpy arrays
        tokens = None

        # Try 'token' column first (RNA data format)
        if "token" in item and item["token"] is not None:
            tokens = item["token"]
        else:
            # Try other possible token column names
            for key in ["input_ids", "tokens", "token_ids", "tokenized"]:
                if key in item and item[key] is not None:
                    tokens = item[key]
                    break

        if tokens is None:
            # If no tokens, try to find text and tokenize it
            text = None
            for key in ["text", "sequence", "input_text"]:
                if key in item and item[key] is not None:
                    text = item[key]
                    break

            if text is not None:
                # Simple tokenization (this is a fallback)
                tokens = [self.vocab.get(char, self.vocab.get("<unk>", 1)) for char in str(text)]
            else:
                # Last resort: use all numeric values as tokens
                numeric_values = [v for v in item.values() if isinstance(v, (int, list))]
                if numeric_values:
                    tokens = numeric_values[0] if isinstance(numeric_values[0], list) else [numeric_values[0]]
                else:
                    tokens = [0]  # padding token

        # Handle numpy array or list
        if hasattr(tokens, "tolist"):
            # Convert numpy array to list
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens)

        # Convert to integers if needed
        try:
            tokens = [int(t) for t in tokens]
        except (ValueError, TypeError):
            tokens = [self.vocab.get("<unk>", 1) for _ in tokens]

        return torch.tensor(tokens, dtype=torch.long)
