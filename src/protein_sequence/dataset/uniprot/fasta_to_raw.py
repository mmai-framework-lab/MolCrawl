from typing import Union, Iterator, List
from pathlib import Path
from argparse import ArgumentParser

import rich.progress

from protein_sequence.utils.configs import ProteinSequenceConfig
from protein_sequence.dataset.uniprot.uniprot_download import UniProtDatasetEnum


def read_fasta_sequences(fasta_filepaths: List[Path]) -> Iterator[str]:
    """
    Reads sequences from a FASTA file and yields them one by one.

    Parameters:
    - fasta_filepath: Path to the input FASTA file.

    Yields:
    - A sequence string (without the header).
    """
    current_sequence = []
    for fasta_filepath in fasta_filepaths:
        with open(fasta_filepath, "r") as fasta_file:
            for line in fasta_file:
                line = line.strip()
                if line.startswith(">"):
                    if current_sequence:
                        yield "".join(current_sequence) + "\n"
                        current_sequence = []
                else:
                    current_sequence.append(line)
            if current_sequence:
                yield "".join(current_sequence) + "\n"


def iterate_over_chunk_raw_files(fasta_filepaths: List[Path], max_lines_per_file: int) -> Iterator[list[str]]:
    sequence_iterator = read_fasta_sequences(fasta_filepaths)
    sequence_chunk = []
    for sequence in sequence_iterator:
        sequence_chunk.append(sequence)
        if len(sequence_chunk) == max_lines_per_file:
            yield sequence_chunk
            sequence_chunk = []


def write_chunk_file(path_file, chunk_sequence: list[str]):
    with open(path_file, "w") as raw_file:
        raw_file.writelines(chunk_sequence)


def parse_fasta_to_raw_sequence(fasta_dir, raw_dir, max_lines_per_file: int) -> None:
    """
    Parses FASTA file and writes the sequences to raw files, splitting them into chunks if necessary.

    Parameters:
    - fasta_dir: Path to the input FASTA files.
    - raw_dir: Dir to save the raw files.
    - max_lines_per_file: Maximum number of lines per output file.
    """

    fasta_filepaths = [path for path in Path(fasta_dir).iterdir() if path.suffix == ".fasta"]
    print(f"Found {len(fasta_filepaths)} FASTA files in {fasta_dir}.")
    chunk_content_iterator = iterate_over_chunk_raw_files(fasta_filepaths, max_lines_per_file=max_lines_per_file)
    for i, chunks in rich.progress.track(enumerate(chunk_content_iterator), "Reading and splitting fasta file in chunks..."):
        path_chunk = Path(raw_dir) / f"chunk_{max_lines_per_file * i}_{max_lines_per_file * (i + 1)}.raw"
        write_chunk_file(path_chunk, chunks)


def fasta_to_raw(dataset: str, output_dir: Union[str, Path], max_lines_per_file: int):
    fasta_dir = Path(output_dir) / dataset
    if dataset == UniProtDatasetEnum.UniParc:
        fasta_dir = Path(output_dir) / "fasta_files"

    raw_dir = Path(output_dir) / "raw_files"
    parse_fasta_to_raw_sequence(fasta_dir, raw_dir, max_lines_per_file)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation
    fasta_to_raw(cfg.dataset, cfg.output_dir, cfg.max_lines_per_file)
