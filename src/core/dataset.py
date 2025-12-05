from pathlib import Path

import torch
from datasets import load_from_disk


class PreparedDataset:
    def __init__(self, dataset_dir, split):
        super().__init__()
        dataset_path = Path(dataset_dir)

        # Try to load from arrow format (with .arrow suffix)
        arrow_split_path = dataset_path / f"{split}.arrow"
        if arrow_split_path.exists():
            print(f"Loading from arrow format: {arrow_split_path}")
            self.data = load_from_disk(str(arrow_split_path))
        else:
            # Fall back to standard HuggingFace dataset format
            try:
                self.data = load_from_disk(str(dataset_path))[split]
            except Exception:
                # Try direct path (no split subdirectory)
                print(f"Trying to load from {dataset_path} directly...")
                self.data = load_from_disk(str(dataset_path))
                if hasattr(self.data, "keys") and split in self.data:
                    self.data = self.data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

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
            raise KeyError(f"Sample does not contain 'input_ids': {sample.keys()}")
