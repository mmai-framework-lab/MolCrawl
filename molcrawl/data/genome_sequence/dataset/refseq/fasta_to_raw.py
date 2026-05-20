from typing import Union, Iterator, List, Tuple
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
import concurrent.futures
import re
import threading
import time

import rich.progress as pb

from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

# Minimum length of an ACGT segment (after N-run splitting) to keep. Segments
# shorter than this are dropped to avoid swamping the corpus with sub-context
# fragments. 100 bp is well below typical model context windows (>=1024 bp)
# while still recovering most chromosome content that the old N-filter lost.
DEFAULT_MIN_SEGMENT_LEN = 100

# Run of one-or-more Ns. Used to split chromosome-scale sequences at assembly
# gaps so that the surrounding ACGT segments can be kept rather than discarding
# the entire chromosome.
_N_RUN_RE = re.compile(r"N+")


def _split_at_n_runs(seq: str, min_len: int = DEFAULT_MIN_SEGMENT_LEN) -> Iterator[str]:
    """Yield uppercase ACGT segments obtained by splitting ``seq`` at N runs.

    Replaces the previous "drop the whole sequence if it contains any N"
    behaviour, which silently discarded every chromosome-scale assembly entry
    that carries centromeric/heterochromatic N gaps (~96% of GRCh38 bases were
    lost). Empty splits and segments shorter than ``min_len`` are filtered out.
    """
    upper = seq.upper()
    for segment in _N_RUN_RE.split(upper):
        if len(segment) >= min_len:
            yield segment


def get_sequence_from_fasta(fasta_filepath: Path, max_lines_per_file: int, num_worker: int, sequence_list: ListProxy) -> None:
    sequence_chunk: List[str] = []
    for sequence in read_fasta_sequences(fasta_filepath):
        sequence_chunk.append(sequence)
        if len(sequence_chunk) == max_lines_per_file:
            while len(sequence_list) > num_worker:
                time.sleep(0.5)
            sequence_list.append(sequence_chunk)
            sequence_chunk = []

    if len(sequence_chunk):
        sequence_list.append(sequence_chunk)


def read_fasta_sequences(
    fasta_filepath: Path,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
) -> Iterator[str]:
    """Yield ACGT segments from a FASTA file, splitting each entry at N runs.

    Parameters:
    - fasta_filepath: Path to the input FASTA file.
    - min_segment_len: Drop N-free segments shorter than this length.

    Yields:
    - One uppercase ACGT segment per ``yield`` (terminated by newline). A
      single FASTA entry may emit zero, one, or many segments depending on
      where its assembly gaps fall.
    """
    current_sequence: List[str] = []

    def _flush(chunks: List[str]) -> Iterator[str]:
        sequence_str = "".join(chunks)
        for segment in _split_at_n_runs(sequence_str, min_segment_len):
            yield segment + "\n"

    with open(fasta_filepath, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    yield from _flush(current_sequence)
                    current_sequence = []
            else:
                current_sequence.append(line)
        if current_sequence:
            yield from _flush(current_sequence)


def iterate_over_chunk_raw_files(fasta_filepaths: List[Path], num_worker: int, max_lines_per_file: int) -> Iterator[list[str]]:
    """
    Reads sequences from a FASTA file and yields them one by one.

    Parameters:
    - fasta_filepath: Path to the input FASTA file.

    Yields:
    - A sequence string (without the header).
    """
    unfinished_jobs = len(fasta_filepaths)
    _lock = threading.Lock()
    columns = list(pb.Progress.get_default_columns()) + [pb.MofNCompleteColumn()]
    with pb.Progress(*columns) as progress:
        task = progress.add_task("Processing fasta files to raw...", total=unfinished_jobs)

        def job_done_callback(future):
            nonlocal unfinished_jobs
            with _lock:
                unfinished_jobs -= 1
            progress.advance(task)

        sequence_chunk: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(num_worker) as executor, Manager() as manager:
            sequence_list = manager.list()
            jobs = [
                executor.submit(get_sequence_from_fasta, path, max_lines_per_file, num_worker, sequence_list)
                for path in fasta_filepaths
            ]
            for job in jobs:
                job.add_done_callback(job_done_callback)
            while unfinished_jobs > 0 or len(sequence_list) > 0:
                if len(sequence_list):
                    sequence_chunk = sequence_chunk + sequence_list.pop(0)
                if len(sequence_chunk) > max_lines_per_file:
                    yield sequence_chunk[:max_lines_per_file]
                    sequence_chunk = sequence_chunk[max_lines_per_file:]
        if len(sequence_chunk):
            yield sequence_chunk


def write_chunk_file(path_file, chunk_sequence: List[str]):
    with open(path_file, "w") as raw_file:
        raw_file.writelines(chunk_sequence)


def process_chunk(id_and_chunk: Tuple[int, List[str]], raw_dir, max_lines_per_file):
    i, chunk = id_and_chunk
    path_chunk = Path(raw_dir) / f"chunk_{max_lines_per_file * i}_{max_lines_per_file * (i + 1)}.raw"
    write_chunk_file(path_chunk, chunk)


def parse_fasta_to_raw_sequence(fasta_dir, raw_dir, num_worker: int, max_lines_per_file: int) -> None:
    """
    Parses FASTA file and writes the sequences to raw files, splitting them into chunks if necessary.

    Parameters:
    - fasta_dir: Path to the input FASTA files.
    - raw_dir: Dir to save the raw files.
    - max_lines_per_file: Maximum number of lines per output file.
    """

    fasta_filepaths = [path for path in Path(fasta_dir).rglob("*.fna")]
    chunk_content_iterator = iterate_over_chunk_raw_files(fasta_filepaths, num_worker, max_lines_per_file=max_lines_per_file)

    with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
        func = partial(process_chunk, raw_dir=raw_dir, max_lines_per_file=max_lines_per_file)
        # list(rich.progress.track(pool.imap(func, enumerate(chunk_content_iterator)), "Reading and splitting fasta file in chunks..."))
        list(executor.map(func, enumerate(chunk_content_iterator)))


def fasta_to_raw_genome(output_dir: Union[str, Path], num_worker: int, max_lines_per_file: int):
    fasta_dir = Path(output_dir) / "extracted_files"
    raw_dir = Path(output_dir) / "raw_files"
    print(f"⌛ Parsing fasta files in {fasta_dir} to raw files in {raw_dir} with {num_worker} workers.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    parse_fasta_to_raw_sequence(fasta_dir, raw_dir, num_worker, max_lines_per_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    fasta_to_raw_genome(cfg.output_dir, cfg.num_worker, cfg.max_lines_per_file)
