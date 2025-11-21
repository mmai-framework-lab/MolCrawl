from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from compounds.utils.config import CompoundConfig
from compounds.utils.tokenizer import CompoundsTokenizer
from datasets import Dataset, DatasetDict


def tokenize_batch_dataset(vocab_path, max_length):
    tokenizer = CompoundsTokenizer(
        vocab_path,
        max_length,
    )

    dataset_dic = {}
    for split in ["train", "valid", "test"]:
        path = f"/data2/sagawatatsuya/riken-dataset-fundational-model/benchmark/GuacaMol/guacamol_v1_{split}.smiles"
        with open(path) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line]
        df = pd.DataFrame(lines, columns=["smiles"])
        df["tokens"] = df["smiles"].apply(tokenizer.tokenize_text)
        print(tokenizer.decode(df["tokens"].iloc[0]))
        dataset_dic[split] = df

    d = {
        "train": Dataset.from_dict(
            {"input_ids": dataset_dic["train"]["tokens"].to_numpy()}
        ),
        "valid": Dataset.from_dict(
            {"input_ids": dataset_dic["valid"]["tokens"].to_numpy()}
        ),
        "test": Dataset.from_dict(
            {"input_ids": dataset_dic["test"]["tokens"].to_numpy()}
        ),
    }

    dataset = DatasetDict(d)

    path_dataset = str(
        Path("/data2/sagawatatsuya/riken-dataset-fundational-model/benchmark/GuacaMol/")
        / "compounds"
        / "training_ready_hf_dataset"
    )
    print(
        f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter."
    )
    dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample = None

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation
    context_length = cfg.max_length

    tokenize_batch_dataset(cfg.vocab_path, cfg.max_length)
