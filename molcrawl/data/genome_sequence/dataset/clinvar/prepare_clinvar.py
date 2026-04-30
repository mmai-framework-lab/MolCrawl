"""
ClinVar Dataset Preparation Script for GPT-2 / BERT Fine-tuning

Downloads ClinVar variants from HuggingFace (``gonzalobenegas/clinvar``) and
embeds each variant in its GRCh38 genomic context, then tokenises the
resulting sequences and saves a chunked HuggingFace DatasetDict for
language-model fine-tuning.

Pipeline:
    1. If ``$CLINVAR_DIR/clinvar_sequences.csv`` does not exist, download it
       automatically via :func:`download_clinvar_sequences`:
       a. Load ``gonzalobenegas/clinvar`` from HuggingFace Hub.
       b. For each variant, extract a (2*flank)-bp window from the GRCh38
          reference FASTA, substituting the ALT allele for the variant copy.
       c. Write ``chrom, pos, ref, alt, reference_sequence, variant_sequence,
          ClinicalSignificance`` to the CSV.
    2. Read ``reference_sequence`` and ``variant_sequence`` columns.
    3. Deduplicate by exact sequence string.
    4. Shuffle and apply an 80 / 10 / 10 train / valid / test split.
    5. Tokenise with the genome SentencePiece BPE tokenizer (vocab_size=4096).
    6. Concatenate all token sequences into one long stream (with EOS between
       sequences), then chunk into fixed-length blocks of *context_length*.
    7. Save as a HuggingFace DatasetDict to
       ``CLINVAR_DIR/training_ready_hf_dataset/``.

Usage (standalone):
    LEARNING_SOURCE_DIR=learning_source_20260311 \\
    python -m molcrawl.data.genome_sequence.dataset.clinvar.prepare_clinvar

    The output is saved to:
        $LEARNING_SOURCE_DIR/genome_sequence/clinvar/training_ready_hf_dataset/

    The intermediate CSV is saved to:
        $LEARNING_SOURCE_DIR/genome_sequence/clinvar/clinvar_sequences.csv
"""

import gzip
import logging
import re
import shutil
import urllib.request
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level helpers (top-level functions for num_proc > 1 compatibility)
# ---------------------------------------------------------------------------


def _tokenize_batch(examples: Dict, tokenizer_path: str) -> Dict:
    """Tokenise a batch of DNA sequences using SentencePiece BPE."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    results = []
    for seq in examples["sequence"]:
        results.append(sp.encode(str(seq).upper()))
    return {"input_ids": results}


def _concatenate_texts(examples: Dict, eos_token_id: int) -> Dict:
    """Concatenate all input_ids in a batch into one flat list with EOS separators."""
    all_ids: List[int] = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos_token_id)
    return {"input_ids": all_ids}


def _create_chunks(examples: Dict, context_length: int) -> Dict:
    """Split a flat input_ids list into fixed-length blocks."""
    ids = examples["input_ids"]
    n_chunks = len(ids) // context_length
    chunks = [ids[i * context_length : (i + 1) * context_length] for i in range(n_chunks)]
    return {"input_ids": chunks}


# ---------------------------------------------------------------------------
# ClinVar CSV download / generation
# ---------------------------------------------------------------------------


# NCBI FTP URL for GRCh38.p13 (GCA_000001405.28) reference genome FASTA
_GRCh38_FTP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
    "GCA_000001405.28_GRCh38.p13/"
    "GCA_000001405.28_GRCh38.p13_genomic.fna.gz"
)


def download_grch38_fasta(dest_fasta: Union[str, Path]) -> Path:
    """Download the GRCh38.p13 reference FASTA from NCBI FTP if not already present.

    Downloads the ``.fna.gz`` file and decompresses it to *dest_fasta*
    (the uncompressed ``.fna`` path).  If *dest_fasta* already exists the
    function returns immediately.

    Args:
        dest_fasta: Destination path for the uncompressed FASTA
                    (e.g. ``dataset/GCA_000001405.28_GRCh38.p13_genomic.fna``).

    Returns:
        Path to the uncompressed FASTA file.
    """
    dest_fasta = Path(dest_fasta)
    dest_gz = Path(str(dest_fasta) + ".gz")

    if dest_fasta.exists():
        logger.info("GRCh38 reference FASTA already exists at %s — skipping download.", dest_fasta)
        return dest_fasta

    dest_fasta.parent.mkdir(parents=True, exist_ok=True)

    if not dest_gz.exists():
        logger.info("Downloading GRCh38 reference FASTA (~3 GB) from NCBI FTP …")
        logger.info("  URL : %s", _GRCh38_FTP_URL)
        logger.info("  Dest: %s", dest_gz)

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0 and block_num % 500 == 0:
                downloaded = block_num * block_size
                pct = min(downloaded / total_size * 100, 100)
                logger.info("  … %.1f%%  (%d / %d bytes)", pct, downloaded, total_size)

        urllib.request.urlretrieve(_GRCh38_FTP_URL, dest_gz, reporthook=_reporthook)
        logger.info("Download complete: %s", dest_gz)

    logger.info("Decompressing %s …", dest_gz)
    with gzip.open(dest_gz, "rb") as f_in, open(dest_fasta, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("Decompression complete: %s", dest_fasta)

    return dest_fasta


def _build_chrom_mapping(ref_genome) -> Dict[str, str]:
    """Build a chromosome-name → sequence-ID mapping from a pyfaidx Fasta object."""
    mapping: Dict[str, str] = {}
    for seq in ref_genome.keys():
        header = ref_genome[seq].long_name
        m = re.search(r"^(CM\d+\.\d+).*chromosome (\w+)", header)
        if m:
            seq_id = m.group(1)
            chrom = m.group(2)
            if chrom.lower().startswith("mito"):
                chrom = "MT"
            mapping[chrom] = seq_id
    return mapping


def _get_sequences(ref_genome, mapping: Dict[str, str], chrom, pos: int, ref: str, alt: str, flank: int = 64):
    """Return (reference_sequence, variant_sequence) centred on *pos*."""
    seq_id = mapping[str(chrom)]
    start = pos - flank
    ref_seq: str = ref_genome[seq_id][start - 1 : pos + flank].seq.upper()

    center_base = ref_seq[flank]
    if center_base != ref.upper():
        logger.warning(
            "Reference mismatch at %s:%d — expected %s, got %s",
            chrom,
            pos,
            ref,
            center_base,
        )

    seq_list = list(ref_seq)
    seq_list[flank] = alt.upper()
    var_seq = "".join(seq_list)
    return ref_seq, var_seq


def download_clinvar_sequences(
    output_file: Union[str, Path],
    ref_fasta: Union[str, Path],
    flank: int = 64,
) -> None:
    """
    Download the ``gonzalobenegas/clinvar`` dataset from HuggingFace and
    embed each variant in its GRCh38 genomic context.

    The result is a CSV with columns:
        chrom, pos, ref, alt, reference_sequence, variant_sequence,
        ClinicalSignificance

    Args:
        output_file: Destination path for the generated CSV.
        ref_fasta: Path to the GRCh38 reference FASTA
                   (e.g. ``dataset/GCA_000001405.28_GRCh38.p13_genomic.fna``).
                   Both plain and ``.gz`` files are accepted; if a ``.gz`` is
                   given without a pre-existing uncompressed copy it will be
                   decompressed automatically.
        flank: Number of base pairs to extract on each side of the variant
               (default 64, producing a 128-bp window).
    """
    import pandas as pd
    from datasets import load_dataset
    from pyfaidx import Fasta

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ref_fasta_path = str(ref_fasta)
    if ref_fasta_path.endswith(".gz"):
        uncompressed = ref_fasta_path[:-3]
        if Path(uncompressed).exists():
            ref_fasta_path = uncompressed
        else:
            logger.info("Decompressing %s …", ref_fasta_path)
            with gzip.open(ref_fasta_path, "rb") as f_in, open(uncompressed, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            ref_fasta_path = uncompressed

    logger.info("Loading gonzalobenegas/clinvar from HuggingFace …")
    dataset = load_dataset("gonzalobenegas/clinvar")
    df: pd.DataFrame = dataset["test"].to_pandas()
    logger.info("Loaded %d variants", len(df))

    # Detect ClinicalSignificance column (name varies across dataset versions)
    clin_col: Optional[str] = None
    for candidate in ("ClinicalSignificance", "clinical_significance", "clin_sig", "clnsig", "significance"):
        if candidate in df.columns:
            clin_col = candidate
            break
    if clin_col is None:
        logger.warning("ClinicalSignificance column not found — available: %s", df.columns.tolist())

    ref_genome = Fasta(ref_fasta_path)
    mapping = _build_chrom_mapping(ref_genome)

    records = []
    for _, row in df.iterrows():
        try:
            ref_seq, var_seq = _get_sequences(
                ref_genome,
                mapping,
                row["chrom"],
                int(row["pos"]),
                str(row["ref"]),
                str(row["alt"]),
                flank=flank,
            )
            records.append(
                {
                    "chrom": row["chrom"],
                    "pos": row["pos"],
                    "ref": row["ref"],
                    "alt": row["alt"],
                    "reference_sequence": ref_seq,
                    "variant_sequence": var_seq,
                    "ClinicalSignificance": row[clin_col] if clin_col else None,
                }
            )
        except Exception as exc:
            logger.warning("Skipping %s:%s — %s", row.get("chrom"), row.get("pos"), exc)

    pd.DataFrame(records).to_csv(output_file, index=False)
    logger.info("Saved %d records to %s", len(records), output_file)


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------


def prepare_clinvar(
    source_file: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_path: str,
    *,
    ref_fasta: Optional[Union[str, Path]] = None,
    context_length: int = 1024,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_proc: int = 4,
) -> str:
    """
    Load ClinVar sequences from *source_file*, build a language-model
    training dataset, and save it to *output_dir*.

    If *source_file* does not exist it is generated automatically by
    :func:`download_clinvar_sequences` using the GRCh38 reference FASTA
    at *ref_fasta* (defaults to ``paths.GRCh38_REF_FASTA``).

    Args:
        source_file: Path to ``clinvar_sequences.csv`` inside ``CLINVAR_DIR``.
        output_dir: Parent directory; the dataset is written under
                    ``output_dir/training_ready_hf_dataset/``.
        tokenizer_path: Path to the SentencePiece ``.model`` file.
        ref_fasta: Path to the GRCh38 reference FASTA used when the source CSV
                   must be generated.  Defaults to ``paths.GRCh38_REF_FASTA``.
        context_length: Token block length (default 1024).
        train_ratio: Fraction of sequences used for training (rest split 50/50
                     between validation and test).
        seed: Random seed for reproducible shuffling.
        num_proc: Parallel workers for HuggingFace Dataset.map() operations.

    Returns:
        Absolute path string of the saved dataset directory.
    """
    import csv

    import sentencepiece as spm

    source_file = Path(source_file)
    output_dir = Path(output_dir)

    if not source_file.exists():
        from molcrawl.core.paths import GRCh38_REF_FASTA

        fasta_path = Path(ref_fasta) if ref_fasta is not None else Path(GRCh38_REF_FASTA)
        if not fasta_path.exists() and Path(str(fasta_path) + ".gz").exists():
            fasta_path = Path(str(fasta_path) + ".gz")
        if not fasta_path.exists():
            # FASTA が未存在の場合は NCBI FTP から自動ダウンロードする
            logger.info(
                "GRCh38 reference FASTA not found at %s — downloading from NCBI FTP …",
                fasta_path,
            )
            fasta_path = download_grch38_fasta(Path(GRCh38_REF_FASTA))
        logger.info(
            "ClinVar source CSV not found at %s — generating from HuggingFace + %s",
            source_file,
            fasta_path,
        )
        download_clinvar_sequences(output_file=source_file, ref_fasta=fasta_path)

    # ------------------------------------------------------------------
    # 1. Collect sequences from CSV
    # ------------------------------------------------------------------
    logger.info("Reading ClinVar sequences from %s", source_file)
    sequences: List[str] = []
    with open(source_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_seq = row.get("reference_sequence", "").strip()
            var_seq = row.get("variant_sequence", "").strip()
            if ref_seq:
                sequences.append(ref_seq.upper())
            if var_seq:
                sequences.append(var_seq.upper())

    logger.info("Total sequences collected (before dedup): %d", len(sequences))

    # ------------------------------------------------------------------
    # 2. Deduplicate
    # ------------------------------------------------------------------
    sequences = list(dict.fromkeys(s for s in sequences if s))
    logger.info("Unique sequences after deduplication: %d", len(sequences))

    # ------------------------------------------------------------------
    # 3. Random 80 / 10 / 10 split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))

    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * (1 - train_ratio) / 2)

    train_seqs = [sequences[i] for i in idx[:n_train]]
    val_seqs = [sequences[i] for i in idx[n_train : n_train + n_val]]
    test_seqs = [sequences[i] for i in idx[n_train + n_val :]]

    logger.info(
        "Split — train: %d, valid: %d, test: %d",
        len(train_seqs),
        len(val_seqs),
        len(test_seqs),
    )

    raw_split = DatasetDict(
        {
            "train": Dataset.from_dict({"sequence": train_seqs}),
            "valid": Dataset.from_dict({"sequence": val_seqs}),
            "test": Dataset.from_dict({"sequence": test_seqs}),
        }
    )

    # ------------------------------------------------------------------
    # 4. Tokenise with SentencePiece BPE genome tokenizer
    # ------------------------------------------------------------------
    logger.info("Tokenising sequences with SentencePiece BPE from %s", tokenizer_path)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    eos_token_id = sp.eos_id()
    logger.info("vocab_size=%d  eos_token_id=%d", sp.get_piece_size(), eos_token_id)

    tokenized = raw_split.map(
        partial(_tokenize_batch, tokenizer_path=tokenizer_path),
        batched=True,
        batch_size=1000,
        remove_columns=["sequence"],
        num_proc=num_proc,
        desc="Tokenising",
    )

    # ------------------------------------------------------------------
    # 5. Concatenate into a single stream, then chunk
    # ------------------------------------------------------------------
    logger.info("Concatenating and chunking to length %d...", context_length)

    concatenated = tokenized.map(
        partial(_concatenate_texts, eos_token_id=eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=tokenized["train"].column_names,
        desc="Concatenating",
    )

    chunked = concatenated.map(
        partial(_create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
        desc="Chunking",
    )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    output_path = output_dir / "training_ready_hf_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset to %s", output_path)
    chunked.save_to_disk(str(output_path))

    logger.info("Done! Dataset statistics:")
    for split_name in chunked:
        logger.info(
            "  %s: %d chunks of %d tokens",
            split_name,
            len(chunked[split_name]),
            context_length,
        )

    return str(output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    from molcrawl.core.paths import CLINVAR_DIR, CLINVAR_SOURCE_FILE, get_refseq_tokenizer_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # CLINVAR_SOURCE_FILE now lives inside CLINVAR_DIR (LEARNING_SOURCE_DIR).
    # If it does not exist, prepare_clinvar() will generate it automatically
    # using the GRCh38 reference FASTA in dataset/.
    prepare_clinvar(
        source_file=CLINVAR_SOURCE_FILE,
        output_dir=CLINVAR_DIR,
        tokenizer_path=get_refseq_tokenizer_path(),
    )
