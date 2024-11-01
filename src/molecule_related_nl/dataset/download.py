import os

from pathlib import Path
from datasets import load_dataset


def download_hf_dataset(save_path):
    
    os.path.exists(save_path) or os.makedirs(save_path)

    data = load_dataset('osunlp/SMolInstruct', trust_remote_code=True)

    for split in data.keys():
        data[split].save_to_disk(str(Path(save_path) / Path(split)))


if __name__ == "__main__":
    download_hf_dataset("src/molecule_related_nl/assets/raw_data")
