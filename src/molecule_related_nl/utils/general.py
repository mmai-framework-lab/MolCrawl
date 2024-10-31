import os
from datasets import Dataset

def read_dataset(dataset_path: str):
    splits = {}
    for folder in os.listdir(dataset_path):
        splits[folder] = Dataset.load_from_disk(os.path.join(dataset_path, folder))

    return splits

def count_number_of_tokens(dataset):
    tokens_dis = []
    def internal_count(x):
        nonlocal tokens_dis
        tokens_dis.append(x["input_ids"] + x["output_ids"])
        return x
        
    dataset.map(internal_count)

    return tokens_dis

def save_dataset(dataset, dataset_path: str):
    for split in dataset.keys():
        dataset[split].save_to_disk(os.path.join(dataset_path, split))
