from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

from rna.utils.config import RnaConfig


def tokenize(output_dir):
    import pandas as pd
    from datasets import load_dataset

    tokenizer = TranscriptomeTokenizer()

    loom_outdir = Path(output_dir) / "loom_dir"
    parquet_outdir = Path(output_dir) / "parquet_files"

    parquet_outdir.mkdir(exist_ok=True, parents=True)
    paths = list(loom_outdir.iterdir())

    for path in tqdm(paths):
        parquet_path = parquet_outdir / path.with_suffix(".parquet").name
        try:
            tokens, _ = tokenizer.tokenize_loom(loom_file_path=path)
            tokenized_sequences = []

            for line in tokens:
                tokenized_sequences.append((line, len(line)))

            df = pd.DataFrame(tokenized_sequences, columns=["token", "token_count"])
            df.to_parquet(parquet_path, index=False)

        except Exception as e:
            print(f"Error with {path}: {e}")

    # Read dataset once with hugging face to preload the dataset can be commented out
    cache_dir = Path(output_dir) / "hf_cache"
    load_dataset("parquet", data_dir=str(parquet_outdir), cache_dir=str(cache_dir))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    tokenize(cfg.output_dir)
