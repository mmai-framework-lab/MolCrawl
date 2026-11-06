#!/usr/bin/env python3
"""
FASTA preprocessing for genomic language model training.

Pipeline
--------
  Phase 1 : FASTA (.fna.gz) → .raw   (nucleotide strings, one chunk per line)
  Phase 2 : .raw             → .parquet (token-ID arrays, HuggingFace-compatible)

Design philosophy
-----------------
Preprocessing choices are modelled on the OpenGenome2 / Evo 2 pipeline
(Brixi et al., Nature 2026; https://doi.org/10.1038/s41586-026-10176-5).
Each non-trivial decision is annotated with [EVO2-REF-N].

Preprocessing steps (Phase 1, in order)
----------------------------------------
1. Upper-case normalization         [EVO2-REF-1]
2. IUPAC ambiguous base → N         [EVO2-REF-2]
3. No additional repeat masking     [EVO2-REF-3]
4. N-run splitting (≥10 Ns)         [EVO2-REF-8]
5. Minimum contig length filter     [EVO2-REF-4]
6. Fixed-length chunking            [EVO2-REF-5]
7. High-N chunk filtering           [EVO2-REF-6]

Preprocessing steps (Phase 2)
-------------------------------
8. Single-nucleotide tokenization   [EVO2-REF-7]
   BERT  : [CLS] + chunk + [SEP], 15 % MLM masking
   GPT-2 : raw token IDs, causal LM
   Evo2  : raw token IDs, causal LM, 8192-token context

[EVO2-REF-1] Upper-case normalization
  NCBI RefSeq uses lowercase for soft-masked repeats (real DNA, not unknown
  bases).  For BERT/GPT-2 with 5-token vocab (A/T/G/C/N) lowercase and
  uppercase are the same nucleotide, so we upper-case everything.
  Evo 2 uses byte-level tokenization (vocab=512) and keeps lowercase; for
  an Evo2-compatible pipeline skip this step.

[EVO2-REF-2] IUPAC ambiguous base → N
  Non-ACGTN characters (R, Y, S, W, K, M, B, D, H, V) → N.

[EVO2-REF-3] No repeat masking
  Evo 2 does not apply additional repeat masking.  Soft-masked repeats
  (lowercase in RefSeq FASTA) are treated as ordinary sequence.

[EVO2-REF-8] N-run splitting
  NCBI RefSeq encodes scaffold gaps as consecutive N runs (≥10 by convention).
  OpenGenome2 preprocessed data contain "almost no consecutive Ns" (personal
  observation), showing that Evo 2 preprocessing splits at N-runs.
  The exact threshold is not publicly disclosed; we adopt the NCBI standard
  of N ≥ 10.  References:
    - NCBI Assembly Basics: https://www.ncbi.nlm.nih.gov/assembly/basics/
    - Brixi et al. (2026) — threshold undisclosed.
  Note: transcript preprocessing uses N ≥ 3 (Brixi et al., 2026 supplement);
  that threshold applies to mRNA/cDNA, not genomic FASTA.

[EVO2-REF-4] Minimum contig length filter
  Sub-sequences shorter than chunk_size are skipped.

[EVO2-REF-5] Fixed-length chunking
  BERT=510 nt (512 with [CLS]/[SEP]), GPT-2=1024 nt, Evo2=8192 nt.
  Trailing incomplete chunks are discarded (no padding).

[EVO2-REF-6] High-N chunk filtering
  Chunks with N fraction > max_n_frac (default 0.1) are discarded.

[EVO2-REF-7] Single-nucleotide tokenization
  A=0, T=1, G=2, C=3, N=4.  Evo 2 operates at "single-nucleotide
  resolution" (Brixi et al., 2026).

Citation
--------
Brixi, G. et al. (2026). Genome modelling and design across all domains of
life with Evo 2. Nature. https://doi.org/10.1038/s41586-026-10176-5

Output layout
-------------
  <output>/
    raw_files/
      <accession>.raw     ← Phase 1: one chunk (nucleotide string) per line
    parquet_<model>/
      <accession>.parquet ← Phase 2: tokenized, HuggingFace-compatible

Vocabulary
----------
  A=0, T=1, G=2, C=3, N=4
  BERT special tokens: [PAD]=5, [UNK]=6, [CLS]=7, [SEP]=8, [MASK]=9

Usage
-----
# Full pipeline (Phase 1 + 2)
python -m molcrawl.preparation.genome.preprocess_fasta \\
    --subset_csv subsets/mammal_centered.csv \\
    --fasta_dir  /path/to/fasta/ \\
    --output     data/processed/mammal_centered_bert/ \\
    --model bert

# Phase 1 only (write .raw, skip tokenization)
python -m molcrawl.preparation.genome.preprocess_fasta \\
    --subset_csv subsets/mammal_centered.csv \\
    --fasta_dir  /path/to/fasta/ \\
    --output     data/processed/mammal_centered_bert/ \\
    --model bert --phase fasta_to_raw

# Phase 2 only (tokenize existing .raw files)
python -m molcrawl.preparation.genome.preprocess_fasta \\
    --subset_csv subsets/mammal_centered.csv \\
    --fasta_dir  /path/to/fasta/ \\
    --output     data/processed/mammal_centered_bert/ \\
    --model bert --phase raw_to_parquet
"""

import argparse
import gzip
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary  [EVO2-REF-7]
# --------------------------------------------------------------------------- #

NUC_TO_ID: dict[str, int] = {
    "A": 0, "T": 1, "G": 2, "C": 3, "N": 4,
}
PAD_ID  = 5
UNK_ID  = 6
CLS_ID  = 7
SEP_ID  = 8
MASK_ID = 9

VOCAB_SIZE = 10

# N-run splitting threshold  [EVO2-REF-8]
N_SPLIT_THRESHOLD = 10

# Pre-compiled regex for N-run splitting
_N_RUN_RE = re.compile(r"[Nn]{" + str(N_SPLIT_THRESHOLD) + r",}")


# --------------------------------------------------------------------------- #
# Phase 1 helpers
# --------------------------------------------------------------------------- #

def normalize_sequence(seq: str) -> str:
    """Steps 1-2: uppercase + IUPAC ambiguous bases → N.

    Returns an ACGTN-only string.  [EVO2-REF-1, EVO2-REF-2, EVO2-REF-3]
    Lowercase soft-masked repeats are upper-cased (not converted to N).
    """
    seq = seq.upper()                                    # [EVO2-REF-1]
    return "".join(c if c in NUC_TO_ID else "N" for c in seq)  # [EVO2-REF-2]


def split_on_n_runs(seq: str, threshold: int = N_SPLIT_THRESHOLD) -> List[str]:
    """Split at runs of >= threshold consecutive Ns.  [EVO2-REF-8]

    Returns non-empty sub-sequences with N-runs removed.
    """
    if threshold != N_SPLIT_THRESHOLD:
        pattern = re.compile(r"[Nn]{" + str(threshold) + r",}")
    else:
        pattern = _N_RUN_RE
    return [p for p in pattern.split(seq) if p]


def n_fraction_str(chunk: str) -> float:
    """Fraction of N characters in a nucleotide string."""
    return chunk.count("N") / len(chunk) if chunk else 0.0


def chunk_sequence_str(
    seq: str, chunk_size: int, min_contig_len: int
) -> Iterator[str]:
    """Yield fixed-length non-overlapping substrings.  [EVO2-REF-5]

    Sequences shorter than min_contig_len produce no chunks.  [EVO2-REF-4]
    Trailing incomplete chunks are discarded.
    """
    if len(seq) < min_contig_len:
        return
    start = 0
    while start + chunk_size <= len(seq):
        yield seq[start : start + chunk_size]
        start += chunk_size


# --------------------------------------------------------------------------- #
# Phase 2 helpers
# --------------------------------------------------------------------------- #

def tokenize_str(chunk: str) -> List[int]:
    """Convert an ACGTN string to token IDs.  [EVO2-REF-7]"""
    return [NUC_TO_ID[c] for c in chunk]


def process_chunk_bert(
    ids: List[int], mlm_prob: float, rng: random.Random
) -> dict:
    """Add [CLS]/[SEP] and apply MLM masking (standard BERT).

    80 % masked → [MASK], 10 % → random nucleotide, 10 % → unchanged.
    Labels of non-masked positions are -100 (ignored in cross-entropy).
    """
    input_ids = [CLS_ID] + ids + [SEP_ID]
    labels    = [-100]   + list(ids) + [-100]

    nuc_positions = list(range(1, len(ids) + 1))
    n_mask = max(1, int(len(nuc_positions) * mlm_prob))
    masked_positions = rng.sample(nuc_positions, n_mask)

    for pos in masked_positions:
        original = input_ids[pos]
        rand_val = rng.random()
        if rand_val < 0.80:
            input_ids[pos] = MASK_ID
        elif rand_val < 0.90:
            input_ids[pos] = rng.randint(0, 3)
        labels[pos] = original

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


def process_chunk_causal(ids: List[int]) -> dict:
    """GPT-2 / Evo2: raw token IDs, no special tokens.  [EVO2-REF-7]"""
    return {"input_ids": ids}


# --------------------------------------------------------------------------- #
# FASTA reading
# --------------------------------------------------------------------------- #

def open_fasta(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def iter_sequences(fasta_path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
    header: Optional[str] = None
    seqparts: List[str] = []
    with open_fasta(fasta_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seqparts)
                header = line[1:].split()[0]
                seqparts = []
            else:
                seqparts.append(line)
    if header is not None:
        yield header, "".join(seqparts)


# --------------------------------------------------------------------------- #
# FASTA discovery
# --------------------------------------------------------------------------- #

FASTA_EXTENSIONS = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


def is_fasta(path: str) -> bool:
    p = path.lower()
    return any(p.endswith(ext) for ext in FASTA_EXTENSIONS)


def accession_to_fasta_path(accession: str, fasta_dir: str) -> Optional[str]:
    d = Path(fasta_dir)
    for ext in (".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta"):
        candidate = d / (accession + ext)
        if candidate.exists():
            return str(candidate)
    sub = d / accession
    if sub.is_dir():
        for f in sub.iterdir():
            if is_fasta(str(f)):
                return str(f)
    matches = [m for m in d.rglob(f"*{accession}*") if is_fasta(str(m))]
    return str(matches[0]) if matches else None


# Maximum nucleotides per .raw line.
# Using LCM(510, 1024) = 261,120 so that both BERT and GPT-2 chunk sizes
# divide evenly, producing zero boundary waste.
# Evo2 (8192) divides 261,120 with 7,168 nt waste per line (<2.8 %).
RAW_LINE_LEN = 261_120  # = 512 × 510 = 255 × 1024


# --------------------------------------------------------------------------- #
# Phase 1: FASTA → .raw  (model-agnostic, no chunking)
# --------------------------------------------------------------------------- #

def process_fasta_to_raw(
    fasta_path: str,
    raw_path: str,
    n_split_threshold: int = N_SPLIT_THRESHOLD,
    max_line_len: int = RAW_LINE_LEN,
) -> Tuple[int, int]:
    """Convert one FASTA file to a .raw file.

    Each line is a normalized (uppercase, IUPAC→N) sub-sequence produced by
    N-run splitting.  Lines longer than max_line_len are split into
    max_line_len-nt pieces so that .raw lines stay at a manageable length.

    Chunking and N-fraction filtering are intentionally deferred to Phase 2
    so that a single .raw file can serve BERT, GPT-2, and Evo2 without
    re-reading the original FASTA.

    Returns
    -------
    (n_lines_written, n_n_run_splits)
    """
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    n_written = n_splits = 0

    with open(raw_path, "w") as fh:
        for _header, seq in iter_sequences(fasta_path):

            # [EVO2-REF-8] split at assembly N-runs
            if n_split_threshold > 0:
                sub_seqs = split_on_n_runs(seq, threshold=n_split_threshold)
                if len(sub_seqs) > 1:
                    n_splits += len(sub_seqs) - 1
            else:
                sub_seqs = [seq]

            for sub_seq in sub_seqs:
                norm = normalize_sequence(sub_seq)  # [EVO2-REF-1,2,3]
                if not norm:
                    continue

                # Split into max_line_len pieces (safety cap for very long
                # chromosomes that have no internal N-runs)
                start = 0
                while start < len(norm):
                    line = norm[start : start + max_line_len]
                    fh.write(line + "\n")
                    n_written += 1
                    start += max_line_len

    return n_written, n_splits


# --------------------------------------------------------------------------- #
# Phase 2: .raw → .parquet  (model-specific chunking + tokenization)
# --------------------------------------------------------------------------- #

def raw_file_to_parquet(
    raw_path, parquet_path, model, chunk_size, max_n_frac,
    mlm_prob, rng, parquet_batch_size=2048,
):
    """Write tokenized records in bounded batches, preserving record order."""
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(parquet_batch_size) is not int or parquet_batch_size < 1:
        raise ValueError("parquet_batch_size must be a positive integer")

    records = []
    n_written = 0
    n_discarded = 0
    writer = None
    temporary = None

    def flush():
        nonlocal writer, temporary, n_written
        if not records:
            return
        table = pa.Table.from_pylist(records)
        if writer is None:
            directory = os.path.dirname(parquet_path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=".parquet-", suffix=".part", dir=directory,
            )
            os.close(fd)
            writer = pq.ParquetWriter(
                temporary, table.schema, compression="snappy",
            )
        writer.write_table(table, row_group_size=len(records))
        n_written += len(records)
        records.clear()

    try:
        with open(raw_path) as fh:
            for line in fh:
                subseq = line.rstrip("\n")
                if not subseq:
                    continue
                for chunk_str in chunk_sequence_str(
                    subseq, chunk_size, chunk_size,
                ):
                    if n_fraction_str(chunk_str) > max_n_frac:
                        n_discarded += 1
                        continue
                    ids = tokenize_str(chunk_str)
                    if model == "bert":
                        record = process_chunk_bert(ids, mlm_prob, rng)
                    else:
                        record = process_chunk_causal(ids)
                    records.append(record)
                    if len(records) >= parquet_batch_size:
                        flush()
        flush()
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, parquet_path)
            temporary = None
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    return n_written, n_discarded


# --------------------------------------------------------------------------- #
# Main flows
# --------------------------------------------------------------------------- #

def _log_phase1(i, total, name, n_w, n_d, n_s, n_split):
    print(
        f"  [{i}/{total}] {name}"
        f"  chunks={n_w:,}  discarded_N={n_d:,}"
        f"  short_contigs={n_s:,}  n_run_splits={n_split:,}",
        file=sys.stderr,
    )


def _log_phase2(i, total, name, n_records):
    print(
        f"  [{i}/{total}] {name}  records={n_records:,}",
        file=sys.stderr,
    )


def run_pipeline(args) -> None:
    """Run Phase 1, Phase 2, or both depending on args.phase."""

    # ── resolve accession list ─────────────────────────────────────────────
    if args.subset_csv:
        import pandas as pd
        df = pd.read_csv(args.subset_csv, low_memory=False)
        acc_col = next(
            (c for c in ("assembly_accession", "accession") if c in df.columns), None
        )
        if acc_col is None:
            sys.exit("[ERROR] subset_csv needs 'assembly_accession' or 'accession' column.")
        accessions = df[acc_col].dropna().tolist()
        print(f"Loaded {len(accessions)} accessions from {args.subset_csv}", file=sys.stderr)
    else:
        input_path = args.input
        if os.path.isdir(input_path):
            fasta_files = sorted(
                p for p in Path(input_path).rglob("*") if is_fasta(str(p))
            )
        elif os.path.isfile(input_path):
            fasta_files = [Path(input_path)]
        else:
            sys.exit(f"[ERROR] --input not found: {input_path}")
        accessions = None

    # raw_files/ is shared across models; parquet dir is model-specific
    raw_dir     = os.path.join(args.output, "raw_files")
    parquet_dir = os.path.join(args.output, f"parquet_{args.model}")
    rng = random.Random(args.seed)

    # ── Phase 1: FASTA → .raw  (model-agnostic) ───────────────────────────
    if args.phase in ("all", "fasta_to_raw"):
        os.makedirs(raw_dir, exist_ok=True)
        print(
            f"\n[Phase 1] FASTA → raw_files/  (shared across models)\n"
            f"  n_split_threshold={args.n_split_threshold}"
            f"  max_line_len={RAW_LINE_LEN:,}",
            file=sys.stderr,
        )

        if accessions is not None:
            items = list(enumerate(accessions, 1))
            total = len(accessions)
            def get_paths(acc):
                fasta = accession_to_fasta_path(acc, args.fasta_dir)
                raw   = os.path.join(raw_dir, acc + ".raw")
                return fasta, raw, acc
        else:
            items = list(enumerate(fasta_files, 1))
            total = len(fasta_files)
            def get_paths(p):
                stem = re.sub(r"\.(fa|fna|fasta)(\.gz)?$", "", p.name, flags=re.IGNORECASE)
                raw  = os.path.join(raw_dir, stem + ".raw")
                return str(p), raw, p.name

        total_lines = total_split = missing = 0
        for i, item in items:
            fasta_path, raw_path, name = get_paths(item)
            if fasta_path is None:
                print(f"  [{i}/{total}] MISSING FASTA for {name} — skipped", file=sys.stderr)
                missing += 1
                continue
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                print(f"  [{i}/{total}] {name}  already exists — skipped", file=sys.stderr)
                continue
            n_lines, n_split = process_fasta_to_raw(
                fasta_path, raw_path,
                n_split_threshold=args.n_split_threshold,
            )
            total_lines += n_lines; total_split += n_split
            print(
                f"  [{i}/{total}] {name}"
                f"  lines={n_lines:,}  n_run_splits={n_split:,}",
                file=sys.stderr,
            )

        print(
            f"\n[Phase 1 done]  lines={total_lines:,}  n_run_splits={total_split:,}"
            + (f"  missing_fasta={missing}" if missing else ""),
            file=sys.stderr,
        )

    # ── Phase 2: .raw → .parquet  (model-specific) ────────────────────────
    if args.phase in ("all", "raw_to_parquet"):
        os.makedirs(parquet_dir, exist_ok=True)
        print(
            f"\n[Phase 2] raw_files/ → parquet_{args.model}/\n"
            f"  model={args.model}  chunk_size={args.chunk_size}"
            f"  max_n_frac={args.max_n_frac}  mlm_prob={args.mlm_prob}",
            file=sys.stderr,
        )

        raw_files = sorted(Path(raw_dir).glob("*.raw"))
        if not raw_files:
            sys.exit(f"[ERROR] No .raw files found in {raw_dir}. Run Phase 1 first.")

        total_records = total_discarded = 0
        for i, raw_path in enumerate(raw_files, 1):
            parquet_path = os.path.join(parquet_dir, raw_path.stem + ".parquet")
            if os.path.exists(parquet_path):
                print(f"  [{i}/{len(raw_files)}] {raw_path.name}  already exists — skipped",
                      file=sys.stderr)
                continue
            n_rec, n_disc = raw_file_to_parquet(
                str(raw_path), parquet_path,
                model=args.model,
                chunk_size=args.chunk_size,
                max_n_frac=args.max_n_frac,
                mlm_prob=args.mlm_prob,
                rng=rng,
                parquet_batch_size=args.parquet_batch_size,
            )
            total_records += n_rec; total_discarded += n_disc
            print(
                f"  [{i}/{len(raw_files)}] {raw_path.name}"
                f"  records={n_rec:,}  discarded_N={n_disc:,}",
                file=sys.stderr,
            )

        print(
            f"\n[Phase 2 done]  records={total_records:,}"
            f"  discarded_N={total_discarded:,}",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def resolve_chunk_size(model: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return {"bert": 510, "gpt2": 1024, "evo2": 8192}[model]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "FASTA → .raw → .parquet preprocessing for genomic LM training.\n"
            "Phase 1: FASTA (.fna.gz) → .raw  (one nucleotide chunk per line)\n"
            "Phase 2: .raw → .parquet  (tokenized, HuggingFace-compatible)\n"
            "Preprocessing follows OpenGenome2 / Evo 2 (Brixi et al., Nature 2026)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",      metavar="PATH",
                     help="Single FASTA file or directory of FASTA files.")
    src.add_argument("--subset_csv", metavar="CSV",
                     help="Subset CSV from sample_species_subsets.py.")

    p.add_argument("--fasta_dir", metavar="DIR",
                   help="FASTA directory (required with --subset_csv).")
    p.add_argument("--output", "-o", required=True, metavar="DIR",
                   help="Output root directory (contains raw_files/ and parquet_<model>/).")
    p.add_argument(
        "--phase",
        choices=["all", "fasta_to_raw", "raw_to_parquet"],
        default="all",
        help=(
            "Which phase(s) to run. "
            "'all' (default): Phase 1 then Phase 2. "
            "'fasta_to_raw': Phase 1 only (write .raw). "
            "'raw_to_parquet': Phase 2 only (tokenize existing .raw)."
        ),
    )
    p.add_argument(
        "--model", choices=["bert", "gpt2", "evo2"], default="bert",
        help=(
            "bert: 510-nt chunks + [CLS]/[SEP] + 15%% MLM masking. "
            "gpt2: 1024-nt chunks, causal LM. "
            "evo2: 8192-nt chunks, causal LM. "
            "Default: bert."
        ),
    )
    p.add_argument("--chunk_size",   type=int,   default=None,
                   help="Override chunk size in nt. Defaults: bert=510, gpt2=1024, evo2=8192.")
    p.add_argument("--min_contig_len", type=int, default=None,
                   help="Skip contigs shorter than this. Default: same as chunk_size.")
    p.add_argument(
        "--n_split_threshold", type=int, default=N_SPLIT_THRESHOLD,
        help=(
            f"Split at runs of >= N consecutive Ns [EVO2-REF-8]. "
            f"NCBI GenBank scaffold-gap convention = 10 (default: {N_SPLIT_THRESHOLD}). "
            f"Set 0 to disable."
        ),
    )
    p.add_argument(
        "--max_n_frac", type=float, default=0.1,
        help="Discard chunks where N fraction exceeds this [EVO2-REF-6]. Default: 0.1.",
    )
    p.add_argument("--mlm_prob", type=float, default=0.15,
                   help="MLM masking probability (BERT only). Default: 0.15.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for MLM masking. Default: 42.")
    p.add_argument("--parquet_batch_size", type=int, default=2048,
                   help="Records per Parquet write batch; default: 2048.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.subset_csv and not args.fasta_dir:
        build_parser().error("--fasta_dir is required when using --subset_csv")

    args.chunk_size = resolve_chunk_size(args.model, args.chunk_size)
    if args.min_contig_len is None:
        args.min_contig_len = args.chunk_size

    n_split_info = (
        f"n_split_threshold={args.n_split_threshold} (NCBI convention, Evo2 threshold undisclosed)"
        if args.n_split_threshold > 0 else "n_split_threshold=disabled"
    )
    print(
        f"Preprocessing settings (Evo2-inspired, Brixi et al. Nature 2026):\n"
        f"  model={args.model}  chunk_size={args.chunk_size}  "
        f"min_contig_len={args.min_contig_len}\n"
        f"  max_n_frac={args.max_n_frac}  phase={args.phase}\n"
        f"  {n_split_info}  [EVO2-REF-8]",
        file=sys.stderr,
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()
