import os
from datasets import Dataset

def read_dataset(dataset_path: str):
    splits = {}
    for folder in os.listdir(dataset_path):
        splits[folder] = Dataset.load_from_disk(os.path.join(dataset_path, folder))

    return splits

def save_dataset(dataset, dataset_path: str):
    for split in dataset.keys():
        dataset[split].save_to_disk(os.path.join(dataset_path, split))
