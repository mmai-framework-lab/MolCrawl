"""
This script will download on of the Uniprot dataset base on the name in the config.

The output will be the a subdir of the output_dir containing a dataset name directory (ex uniprot_50) containing the rest of the file:

- Archive file, for uniprot a archive dir will be creating containing all the files
- A fasta file extracted from the archive, for uniprot a fasta_file directory will be created containing all the file.
- A raw_files directory containing multiple file with one protein sequence per line.
- A parquet_files directory, containing two column parquet file tokenized sequence ("token") and the number of ("token_count")
- A token_counts.pkl file which contains a list of int corresponding to token_count for computing statistics of the dataset.

"""

from argparse import ArgumentParser

from protein_sequence.dataset.tokenizer import tokenize_to_parquet
from protein_sequence.dataset.uniprot.fasta_to_raw import fasta_to_raw_protein
from protein_sequence.dataset.uniprot.uniprot_download import process_dataset
from protein_sequence.utils.configs import ProteinSequenceConfig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation

    process_dataset(cfg.dataset, cfg.output_dir, cfg.num_worker, cfg.use_md5)
    fasta_to_raw_protein(cfg.dataset, cfg.output_dir, cfg.max_lines_per_file)
    tokenize_to_parquet(cfg.output_dir, cfg.num_worker)
