import json
import pickle
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from molcrawl.data.rna.utils.config import RnaConfig


# Process the 'genes' column
def process_partition(partition):
    local_counter = Counter()
    local_shapes = []

    for genes_array in partition["genes"]:
        # Update the Counter and collect shapes
        local_counter.update(genes_array)
        local_shapes.append(len(genes_array))

    return local_counter, local_shapes


def reverse_remap_genes_dict(genes_dict, vocab_dict):
    """
    Remap the numerical keys of a dictionary back to gene names using a given vocabulary dictionary.

    Args:
        genes_dict (dict): The original dictionary with numerical values as keys.
        vocab_dict (dict): A dictionary where gene names are mapped to numerical values.

    Returns:
        dict: A new dictionary with the remapped keys (gene names).
    """
    # Create a reverse dictionary mapping numerical values back to gene names
    reverse_vocab_dict = {v: k for k, v in vocab_dict.items()}

    # Initialize a new dictionary for the remapped data
    remapped_dict = {}

    # Iterate through the original dictionary
    for gene_id, count in genes_dict.items():
        # Check if the numerical key exists in the reverse vocabulary
        if gene_id in reverse_vocab_dict:
            # Get the corresponding gene name and update the new dictionary
            remapped_dict[reverse_vocab_dict[gene_id]] = count

    return remapped_dict


def compute_stats(output_dir):
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    # Load Parquet files with Dask
    parquet_dir = Path(output_dir) / "parquet_files"
    ddf = dd.read_parquet(parquet_dir)

    # Initialize Counter and array shapes
    genes_counter = Counter()
    array_shapes = []

    # Apply the function to each partition (map_partitions allows parallel processing)
    # job = ddf.map_partitions(process_partition)
    # progress(job)
    # result = job.compute()
    with ProgressBar():
        result = ddf.map_partitions(process_partition).compute()

    # Aggregate results
    for local_counter, local_shapes in result:
        genes_counter.update(local_counter)
        array_shapes.extend(local_shapes)

    with open(Path(output_dir) / "gene_vocab.json", "r") as file:
        vocab = json.load(file)

    genes_counter = reverse_remap_genes_dict(genes_counter, vocab)

    save_dir = Path(output_dir) / "stats"
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / "gene_counts.json", "w") as file:
        json.dump(genes_counter, file, indent=4)

    with open(save_dir / "array_shapes.pkl", "wb") as handle:
        pickle.dump(array_shapes, handle)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    compute_stats(cfg.output_dir)
