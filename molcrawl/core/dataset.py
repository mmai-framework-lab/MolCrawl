from pathlib import Path


class PreparedDataset:
    def __init__(self, dataset_dir, split):
        super().__init__()
        from datasets import load_from_disk

        dataset_path = Path(dataset_dir)
        # How and from where this split was opened, for run_manifest.json. Set by
        # whichever branch below succeeds; reading it changes nothing here.
        self.source = {"method": None, "path": None}

        # Try to load from arrow format (with .arrow suffix)
        arrow_split_path = dataset_path / f"{split}.arrow"
        if arrow_split_path.exists():
            print(f"Loading from arrow format: {arrow_split_path}")
            self.data = load_from_disk(str(arrow_split_path))
            self.source = {"method": "load_from_disk(<dir>/<split>.arrow)", "path": str(arrow_split_path.resolve())}
        else:
            # Fall back to standard HuggingFace dataset format
            try:
                self.data = load_from_disk(str(dataset_path))[split]
                self.source = {"method": "load_from_disk(<dir>)[split]", "path": str(dataset_path.resolve())}
            except Exception:
                # Try split subdirectory (e.g., {dataset_dir}/train/)
                split_path = dataset_path / split
                if split_path.exists():
                    print(f"Trying to load from split subdirectory {split_path}...")
                    self.data = load_from_disk(str(split_path))
                    self.source = {"method": "load_from_disk(<dir>/<split>)", "path": str(split_path.resolve())}
                else:
                    # Try direct path (no split subdirectory)
                    print(f"Trying to load from {dataset_path} directly...")
                    self.data = load_from_disk(str(dataset_path))
                    self.source = {"method": "load_from_disk(<dir>)", "path": str(dataset_path.resolve())}
                    if hasattr(self.data, "keys") and split in self.data:
                        self.data = self.data[split]
                        self.source["method"] = "load_from_disk(<dir>)[split] after a failed first attempt"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch

        sample = self.data[idx]

        # Backward/format compatibility:
        # compounds datasets often store token ids under `tokens`.
        if "input_ids" not in sample and "tokens" in sample:
            sample["input_ids"] = sample["tokens"]
        if "input_ids" not in sample and "sequence_tokens" in sample:
            sample["input_ids"] = sample["sequence_tokens"]

        # For GPT-2: return combined input_ids and output_ids as single sequence
        if "output_ids" in sample and "input_ids" in sample:
            # Combine input and output for autoregressive training
            input_ids = sample["input_ids"]
            output_ids = sample["output_ids"]
            combined = input_ids + output_ids
            return torch.tensor(combined, dtype=torch.long)
        elif "input_ids" in sample:
            # Standard format
            input_ids = sample["input_ids"]
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise KeyError(
                "Sample does not contain any token id field. "
                f"Expected one of ['input_ids', 'tokens', 'sequence_tokens'], got: {sample.keys()}"
            )
