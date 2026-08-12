"""
RNA Transcriptome Dataset
"""

import os
import json
import pickle
from pathlib import Path


def load_gene_vocab(vocab_file):
    """Load a {gene: token_id} mapping from a .json or .pkl vocabulary file.

    The Geneformer token space the RNA corpus is tokenized in ships as
    ``geneformer/token_dictionary.pkl``; the CellxGene census gene-symbol list
    written next to the data is a ``.json``. Both are valid inputs, so accept
    either rather than forcing a derived copy in the other format.
    """
    if vocab_file is None:
        raise ValueError("vocab_file is required; got None")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Gene vocabulary not found at {vocab_file}")
    if str(vocab_file).endswith(".pkl"):
        with open(vocab_file, "rb") as f:
            return pickle.load(f)
    with open(vocab_file, "r") as f:
        return json.load(f)


class RNABinDataset:
    """An RNA split backed by a flat uint16 memmap instead of Arrow.

    Same rows, same order, same dtype as :class:`RNADataset` — row i of the
    ``.bin`` is row i of the split — so a given index draws the same block and
    training is unaffected. Only the storage differs.

    Why it exists: fetching a 16-row micro-batch out of the Arrow-backed
    HuggingFace dataset costs ~175 ms on a compute node with four ranks reading
    concurrently, i.e. 11 ms for an 8 KB row. That is read latency, not
    bandwidth, and it left RNA GPT-2 spending 85 % of every iteration waiting on
    data at ~2 % MFU (job 19223). A flat memmap turns the same fetch into an
    offset read the page cache can handle, measured 6.3x faster across all four
    ranks (job 19237). Every other modality already reads a flat array.

    Build the files with ``scripts``-side ``rna_pack_to_bin.py``; each split is
    ``<split>.bin`` plus a ``<split>.json`` describing rows, block and dtype.
    """

    _SPLIT_ALIASES = {"val": "valid"}

    def __init__(self, bin_dir, split="train", vocab_file=None):
        import numpy as np

        split = self._SPLIT_ALIASES.get(split, split)
        self.bin_dir = bin_dir
        self.split = split

        self.vocab = load_gene_vocab(vocab_file)
        self.vocab_size = len(self.vocab)

        meta_path = Path(bin_dir) / f"{split}.json"
        bin_path = Path(bin_dir) / f"{split}.bin"
        if not meta_path.exists() or not bin_path.exists():
            raise FileNotFoundError(
                f"packed RNA split not found: expected {bin_path} and {meta_path}"
            )
        meta = json.loads(meta_path.read_text())
        if meta.get("dtype") != "uint16":
            raise ValueError(f"unsupported dtype {meta.get('dtype')!r} in {meta_path}")

        self.rows = int(meta["rows"])
        self.block = int(meta["block"])
        expected = self.rows * self.block * 2
        actual = bin_path.stat().st_size
        if actual != expected:
            # A truncated pack would otherwise surface as silently wrong tokens
            # in the tail of the split.
            raise ValueError(
                f"{bin_path} is {actual} bytes, expected {expected} "
                f"({self.rows} rows x {self.block} x uint16)"
            )
        if int(meta.get("max_token_id", 0)) >= self.vocab_size:
            raise ValueError(
                f"max_token_id {meta['max_token_id']} does not fit vocabulary of "
                f"{self.vocab_size}"
            )

        self._arr = np.memmap(bin_path, dtype=np.uint16, mode="r",
                              shape=(self.rows, self.block))
        print(f"📦 RNA memmap {split}: {self.rows:,} rows x {self.block} "
              f"({actual/1e9:.2f} GB) from {bin_path}")

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        import numpy as np
        import torch

        return torch.from_numpy(np.asarray(self._arr[idx], dtype=np.int64))

    # Deliberately no batched-fetch helper here. Collapsing get_batch's per-row
    # lookups is a separate change, and landing both at once would make it
    # impossible to attribute the speed-up to either. Storage first; revisit
    # batching only if the memmap alone does not close the gap.


class RNADataset:
    """RNA Transcriptome Dataset"""

    def __init__(self, data_dir, split="train", vocab_file=None, test_size=0.1):
        import pyarrow as pa
        from datasets import Dataset, load_from_disk

        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size

        # Load vocabulary. Silently falling back to a 3-token default produced
        # models with wte=Embedding(3, n_embd) that died at the first forward
        # pass once the loader fed token ids drawn from the real ~25k-entry
        # vocabulary — fail loudly instead.
        self.vocab = load_gene_vocab(vocab_file)
        self.vocab_size = len(self.vocab)

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
                raise FileNotFoundError(f"Could not load data from {data_dir}") from e2

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
        import torch

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
