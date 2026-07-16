from typing import Union, Iterator, List, Tuple
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
import concurrent.futures
import gzip
import logging
import os
import queue
import re
import threading
import time

import rich.progress as pb

from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)

# Minimum length of an ACGT segment (after N-run splitting) to keep. Segments
# shorter than this are dropped to avoid swamping the corpus with sub-context
# fragments. 100 bp is well below typical model context windows (>=1024 bp)
# while still recovering most chromosome content that the old N-filter lost.
DEFAULT_MIN_SEGMENT_LEN = 100

# Maximum characters written per .raw line. LCM(510, 1024) = 261,120, so a line
# divides evenly into both BERT (510-nt) and GPT-2 (1024-nt) chunks with zero
# boundary loss in Phase 3, while bounding per-line memory (~255 KB).
RAW_LINE_LEN = 261_120

# Run of one-or-more Ns. Used to split chromosome-scale sequences at assembly
# gaps so that the surrounding ACGT segments can be kept rather than discarding
# the entire chromosome.
_N_RUN_RE = re.compile(r"N+")

# Any base that is not a canonical A/C/G/T. Used by the per-accession pipeline to
# fold N and IUPAC ambiguity codes (R/Y/W/S/K/M/B/D/H/V) to N before splitting,
# so the single-nucleotide tokenizer in Phase 3 only ever sees A/C/G/T. (The
# legacy aggregating flow keeps using ``_split_at_n_runs`` unchanged.)
_NON_ACGT_RE = re.compile(r"[^ACGT]")

# FASTA extensions accepted by the per-accession pipeline (plain or gzip).
_FASTA_SUFFIXES = (".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta")


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


def get_sequence_from_fasta(
    fasta_filepath: Path,
    max_lines_per_file: int,
    num_worker: int,
    sequence_queue: "queue.Queue[List[str]]",
) -> None:
    sequence_chunk: List[str] = []
    for sequence in read_fasta_sequences(fasta_filepath):
        sequence_chunk.append(sequence)
        if len(sequence_chunk) == max_lines_per_file:
            # Back-pressure: pause when the queue is saturated to bound memory.
            while sequence_queue.qsize() > num_worker:
                time.sleep(0.5)
            sequence_queue.put(sequence_chunk)
            sequence_chunk = []

    if len(sequence_chunk):
        sequence_queue.put(sequence_chunk)


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
        # In-process queue.Queue (thread-safe) instead of multiprocessing.Manager().list().
        # The Manager spawns a subprocess and routes every list mutation through a Unix
        # socket; with hundreds of FASTA files the IPC layer would silently deadlock
        # (parent stuck in __skb_wait_for_more_packets, no progress, no error). A Queue
        # stays in-process, is faster, and removes the deadlock failure mode entirely.
        sequence_queue: "queue.Queue[List[str]]" = queue.Queue()
        with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
            jobs = [
                executor.submit(get_sequence_from_fasta, path, max_lines_per_file, num_worker, sequence_queue)
                for path in fasta_filepaths
            ]
            for job in jobs:
                job.add_done_callback(job_done_callback)
            while unfinished_jobs > 0 or not sequence_queue.empty():
                try:
                    sequence_chunk = sequence_chunk + sequence_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                while len(sequence_chunk) > max_lines_per_file:
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


# --------------------------------------------------------------------------- #
# Per-accession pipeline (subset/CSV-driven flow)
#
# The functions above aggregate the whole corpus into chunk_<n>_<m>.raw files,
# which suits the legacy single-corpus flow. For the subset flow we instead
# emit one .raw per assembly accession so that downloads, raw conversion, and
# (Phase 3) tokenization stay traceable and independently resumable.
# --------------------------------------------------------------------------- #


def _open_fasta(path: Union[str, Path]):
    """Open a FASTA file as text, transparently handling gzip (.gz) input."""
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "r")


def _accession_stem(path: Path) -> str:
    """Strip a FASTA suffix to recover the accession-based file stem."""
    name = path.name
    for suf in _FASTA_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem


def _normalize_and_split(seq: str, min_len: int) -> Iterator[str]:
    """Uppercase, fold every non-ACGT base to N, then split at N runs.

    Yields ACGT-only segments of length >= ``min_len``. Folding non-ACGT to N
    means N gaps and sparse IUPAC ambiguity codes are both removed at the split,
    leaving a pure A/C/G/T alphabet for single-nucleotide tokenization.
    """
    normalized = _NON_ACGT_RE.sub("N", seq.upper())
    for segment in _N_RUN_RE.split(normalized):
        if len(segment) >= min_len:
            yield segment


def iter_acgt_segments(
    fasta_path: Union[str, Path],
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
) -> Iterator[str]:
    """Yield uppercase ACGT-only segments from a plain/gzip FASTA.

    N gaps and IUPAC ambiguity codes are folded to N and removed at the split,
    so every yielded segment contains only A/C/G/T.
    """
    for _contig, seg in iter_acgt_segments_with_contig(fasta_path, min_segment_len):
        yield seg


def _contig_id_from_header(header_line: str) -> str:
    """Extract the contig id (first whitespace token) from a FASTA ``>`` header.

    ``>NC_000022.11 Homo sapiens chromosome 22, GRCh38 ...`` -> ``NC_000022.11``.
    Tabs are stripped defensively so the id can never collide with the ``\\t``
    field separator used in the raw file.
    """
    token = header_line[1:].strip().split()[0] if header_line[1:].strip() else ""
    return token.replace("\t", "_")


def iter_acgt_segments_with_contig(
    fasta_path: Union[str, Path],
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
) -> Iterator[Tuple[str, str]]:
    """Yield ``(contig_id, segment)`` pairs from a plain/gzip FASTA.

    ``contig_id`` is the source FASTA record id (first token of the ``>``
    header) that each ACGT segment was extracted from. This is what lets the
    downstream split hold out whole contigs/chromosomes (F2-a) instead of
    scattering a genome's adjacent windows across train/eval. N gaps and IUPAC
    ambiguity codes are folded to N and removed at the split, so every yielded
    segment contains only A/C/G/T.
    """
    current: List[str] = []
    contig_id = ""
    with _open_fasta(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    for seg in _normalize_and_split("".join(current), min_segment_len):
                        yield contig_id, seg
                    current = []
                contig_id = _contig_id_from_header(line)
            else:
                current.append(line)
        if current:
            for seg in _normalize_and_split("".join(current), min_segment_len):
                yield contig_id, seg


def fasta_file_to_raw(
    fasta_path: Union[str, Path],
    raw_path: Union[str, Path],
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    max_line_len: int = RAW_LINE_LEN,
) -> int:
    """Convert one FASTA file to a single .raw file, one segment per line.

    Format (F2-a, contig-aware): each line is ``<contig_id>\\t<sequence>``.
    Storing the source contig id per line lets Phase 3 stamp every window with
    its contig so the split can hold out whole contigs/chromosomes. Segments
    longer than ``max_line_len`` are wrapped across multiple lines (the contig
    id is repeated on each wrapped line) so no single line exceeds the limit.
    Written atomically via a ``.part`` file. Returns the number of lines written.
    """
    raw_path = Path(raw_path)
    tmp = raw_path.with_suffix(raw_path.suffix + ".part")
    n_lines = 0
    with open(tmp, "w") as out:
        for contig_id, seg in iter_acgt_segments_with_contig(fasta_path, min_segment_len):
            for i in range(0, len(seg), max_line_len):
                out.write(contig_id)
                out.write("\t")
                out.write(seg[i : i + max_line_len])
                out.write("\n")
                n_lines += 1
    os.replace(tmp, raw_path)
    return n_lines


def fasta_to_raw_per_accession(
    base_dir: Union[str, Path],
    num_worker: int = 8,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    max_line_len: int = RAW_LINE_LEN,
    force: bool = False,
) -> bool:
    """Convert every FASTA in ``base_dir/extracted_files`` to one .raw per accession.

    Output: ``base_dir/raw_files/<accession>.raw``. Already-present non-empty
    raw files are skipped unless ``force``. Writes ``fasta_to_raw_complete.marker``
    on success. Returns ``True`` if at least one .raw file exists afterwards.
    """
    base_dir = Path(base_dir)
    fasta_dir = base_dir / "extracted_files"
    raw_dir = base_dir / "raw_files"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(
        p for p in fasta_dir.rglob("*") if p.name.endswith(_FASTA_SUFFIXES)
    )
    if not fasta_files:
        logger.error(f"No FASTA files found in {fasta_dir}")
        return False

    logger.info(
        f"FASTA → raw (per-accession): {len(fasta_files)} files, "
        f"min_segment_len={min_segment_len}, max_line_len={max_line_len}, workers={num_worker}"
    )

    def _one(fasta_path: Path) -> Tuple[str, int]:
        stem = _accession_stem(fasta_path)
        raw_path = raw_dir / f"{stem}.raw"
        if not force and raw_path.exists() and raw_path.stat().st_size > 0:
            return stem, -1  # skipped
        n = fasta_file_to_raw(fasta_path, raw_path, min_segment_len, max_line_len)
        return stem, n

    total_lines = n_done = n_skip = 0
    with concurrent.futures.ThreadPoolExecutor(num_worker) as ex:
        for stem, n in ex.map(_one, fasta_files):
            if n == -1:
                n_skip += 1
            else:
                n_done += 1
                total_lines += n
                if n == 0:
                    logger.warning(f"{stem}: produced 0 segments (>= {min_segment_len} bp)")

    raw_count = len(list(raw_dir.glob("*.raw")))
    logger.info(
        f"FASTA → raw done: converted={n_done} skipped={n_skip} "
        f"raw_files={raw_count} total_lines={total_lines:,}"
    )
    if raw_count == 0:
        return False
    (base_dir / "fasta_to_raw_complete.marker").touch()
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    fasta_to_raw_genome(cfg.output_dir, cfg.num_worker, cfg.max_lines_per_file)
