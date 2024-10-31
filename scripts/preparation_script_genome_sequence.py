from argparse import ArgumentParser

from genome_sequence.dataset.refseq.download_refseq import download_refseq
from genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw
from genome_sequence.dataset.train_tokenizer import train_tokenizer
from genome_sequence.dataset.tokenizer import raw_to_parquet
from genome_sequence.utils.config import GenomeSequenceConfig
from core.base import setup_logging


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    setup_logging(cfg.output_dir)

    download_refseq(cfg.output_dir, cfg.path_species, cfg.num_worker)
    fasta_to_raw(cfg.output_dir, cfg.num_worker, cfg.max_lines_per_file)
    train_tokenizer(cfg.output_dir, cfg.vocab_size)
    raw_to_parquet(cfg.output_dir)
