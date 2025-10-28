from datasets import load_from_disk
import torch


class PreparedDataset:

    def __init__(self, dataset_dir, split):
        super().__init__()
        self.data = load_from_disk(dataset_dir)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]["input_ids"]
        # Ensure tensor is long type for embedding layer compatibility
        return torch.tensor(input_ids, dtype=torch.long)
