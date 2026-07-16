import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Add project root src directory to path
from molcrawl.core.paths import (
    CLINVAR_DIR,
    CLINVAR_SOURCE_FILE,
    GENOME_SEQUENCE_DIR,
    PROJECT_ROOT,
    get_refseq_tokenizer_path,
)
from molcrawl.core.base import setup_logging
from molcrawl.data.genome_sequence.dataset.refseq.download_refseq import download_refseq
from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw_genome
from molcrawl.data.genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from molcrawl.data.genome_sequence.dataset.tokenizer import raw_to_parquet
from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    """Create and save distribution plot for tokenized sequence lengths"""
    try:
        from molcrawl.core.utils.image_manager import get_image_path

        plt.hist(data["train"]["num_tokens"], bins=np.arange(0, 200, 1))
        plt.xlabel("Length of tokenized dataset")
        plt.title("Distribution of tokenized lengths")

        image_path = get_image_path("genome_sequence", "genome_sequence_tokenized_lengths_dist.png")
        plt.savefig(image_path)
        plt.close()
        logger.info(f"Saved distribution of tokenized dataset lengths to {image_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create distribution plot: {e}")
        return False


def check_progress_status(base_dir):
    """Check the progress status of all processing steps

    Args:
        base_dir (str): Base directory for genome sequence data

    Returns:
        bool: True if all steps are completed, False otherwise
    """
    # Marker file path for each processing stage
    download_marker = Path(base_dir) / "download_complete.marker"
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    train_tokenizer_marker = Path(base_dir) / "train_tokenizer_complete.marker"
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"

    # Output directory and path to check file existence
    raw_files_dir = Path(base_dir) / "raw_files"
    tokenizer_model = Path(base_dir) / "spm_tokenizer.model"
    parquet_dir = Path(base_dir) / "parquet_files"

    # Check progress
    logger.info("=== Genome Sequence Dataset Preparation Progress ===")
    steps_completed = 0
    total_steps = 4

    if download_marker.exists():
        logger.info("✓ Step 1/4: RefSeq download - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 1/4: RefSeq download - PENDING")

    if fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("✓ Step 2/4: FASTA to raw conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 2/4: FASTA to raw conversion - PENDING")

    if train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("✓ Step 3/4: Tokenizer training - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 3/4: Tokenizer training - PENDING")

    if raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("✓ Step 4/4: Raw to Parquet conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 4/4: Raw to Parquet conversion - PENDING")

    logger.info(f"Progress: {steps_completed}/{total_steps} steps completed")
    logger.info("====================================================")

    return steps_completed == total_steps


def process1_download_refseq(
    base_dir,
    path_species,
    num_worker,
    species_timeout=30 * 60,
    max_retries=2,
    force=False,
):
    """Process 1: Download RefSeq dataset
    Args:
        base_dir (str): Base directory for genome sequence data
        path_species (str): Path to species list file
        num_worker (int): Number of workers for parallel processing
        species_timeout (int): Per-species download timeout in seconds
        max_retries (int): Maximum retries per species before giving up
        force (bool): Force re-download even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
    download_marker = Path(base_dir) / "download_complete.marker"

    if not force and download_marker.exists():
        logger.info("👉Process1 : RefSeq dataset download already completed. Skipping...")
        logger.info("Use --force option to re-download.")
        return True

    try:
        if force:
            logger.info("👉Process1 : Force option specified. Re-downloading RefSeq dataset...")
        else:
            logger.info("👉Process1 : Downloading RefSeq dataset...")

        logger.info(f" - Species timeout : {species_timeout}s")
        logger.info(f" - Max retries     : {max_retries}")

        download_refseq(
            base_dir,
            path_species,
            num_worker,
            species_timeout=species_timeout,
            max_retries=max_retries,
        )

        # Check if all species failed — treat that as a hard failure
        failed_path = Path(base_dir) / "failed_species.json"
        if failed_path.exists():
            import json

            with open(failed_path) as fp:
                failed = json.load(fp)
            if len(failed) > 0:
                logger.warning(f"{len(failed)} species failed to download.")
                # Count total species from directory of .txt files
                species_path = Path(path_species)
                if species_path.is_dir():
                    species_count = sum(
                        sum(1 for _ in open(f)) for f in species_path.glob("*.txt") if f.is_file()
                    )
                elif species_path.is_file():
                    species_count = sum(1 for _ in open(species_path))
                else:
                    species_count = 0
                if species_count > 0 and len(failed) >= species_count:
                    logger.error("ALL species failed to download. Not marking step as complete.")
                    return False
                elif len(failed) > 0:
                    logger.warning(f"Continuing despite {len(failed)} failed species.")

        download_marker.touch()
        logger.info("RefSeq download completed.")
        return True

    except Exception as e:
        logger.error(f"RefSeq download failed: {e}")
        return False


def process2_fasta_to_raw(base_dir, num_worker, max_lines_per_file, force=False):
    """Process 2: Convert FASTA files to raw text format
    Args:
        base_dir (str): Base directory for genome sequence data
        num_worker (int): Number of workers for parallel processing
        max_lines_per_file (int): Maximum lines per output file
        force (bool): Force reconversion even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    raw_files_dir = Path(base_dir) / "raw_files"

    if not force and fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("👉Process2 : FASTA to raw conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
        return True

    try:
        if force:
            logger.info("👉Process2 : Force option specified. Reconverting FASTA to raw text...")
        else:
            logger.info("👉Process2 : Converting FASTA to raw text...")

        logger.info(f" - Base Directory : {base_dir}")
        logger.info(f" - Number of Workers : {num_worker}")
        logger.info(f" - Max Lines per File : {max_lines_per_file}")

        fasta_to_raw_genome(base_dir, num_worker, max_lines_per_file)

        # Verify output was actually produced
        raw_count = len(list(raw_files_dir.glob("*.raw"))) if raw_files_dir.exists() else 0
        if raw_count == 0:
            logger.error("FASTA to raw conversion produced 0 files. Not marking step as complete.")
            return False

        fasta_to_raw_marker.touch()
        logger.info(f"FASTA to raw conversion completed ({raw_count} files).")
        return True

    except Exception as e:
        logger.error(f"FASTA to raw conversion failed: {e}")
        return False


def process3_train_tokenizer(
    base_dir, vocab_size, max_lines_per_file, input_sentence_size, max_sentence_length=4192, force=False
):
    """Process 3: Train SentencePiece tokenizer
    Args:
        base_dir (str): Base directory for genome sequence data
        vocab_size (int): Vocabulary size for tokenizer
        max_lines_per_file (int): Maximum lines per file for training
        input_sentence_size (int): Input sentence size for tokenizer
        max_sentence_length (int): Max characters per line; longer lines are skipped during training
        force (bool): Force retraining even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
    train_tokenizer_marker = Path(base_dir) / "train_tokenizer_complete.marker"
    tokenizer_model = Path(base_dir) / "spm_tokenizer.model"

    if not force and train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("👉Process3 : Tokenizer training already completed. Skipping...")
        logger.info("Use --force option to retrain tokenizer.")
        return True

    try:
        if force:
            logger.info("👉Process3 : Force option specified. Retraining tokenizer...")
        else:
            logger.info("👉Process3 : Training tokenizer...")

        logger.info(f" - Base Directory : {base_dir}")
        logger.info(f" - vocab size : {vocab_size}")
        logger.info(f" - max lines per file : {max_lines_per_file}")
        logger.info(f" - input sentence size : {input_sentence_size}")
        logger.info(f" - max sentence length : {max_sentence_length}")

        train_tokenizer(base_dir, vocab_size, max_lines_per_file, input_sentence_size, max_sentence_length)
        train_tokenizer_marker.touch()
        logger.info("Tokenizer training completed.")
        return True

    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        return False


def process4_raw_to_parquet(base_dir, num_proc=None, batch_size=None, force=False):
    """Process 4: Convert raw text files to Parquet format"""
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"
    parquet_dir = Path(base_dir) / "parquet_files"

    if not force and raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("👉Process4 : Raw to Parquet conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
        return True

    try:
        if force:
            logger.info("👉Process4 : Force option specified. Reconverting raw text to Parquet...")
        else:
            logger.info("👉Process4 : Converting raw text to Parquet...")

        logger.info(f" - Base Directory : {base_dir}")
        if num_proc is not None:
            logger.info(f" - num_proc for map : {num_proc}")
        if batch_size is not None:
            logger.info(f" - batch_size for map : {batch_size}")

        # Assuming that the raw_to_parquet side is an implementation that receives num_proc / batch_size
        raw_to_parquet(base_dir, num_proc=num_proc, batch_size=batch_size)

        raw_to_parquet_marker.touch()
        logger.info("Raw to Parquet conversion completed.")
        return True

    except Exception as e:
        logger.error(f"Raw to Parquet conversion failed: {e}")
        return False


def process5_generate_statistics(base_dir, vocab_size, force=False):
    """Process 5: Generate statistics and distribution plots
    Args:
        base_dir (str): Base directory for genome sequence data
        vocab_size (int): Vocabulary size
        force (bool): Force regeneration even if already exists
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("👉Process5 : Loading Parquet dataset and generating statistics...")

    try:
        data = load_dataset(
            "parquet",
            data_files=[str(Path(base_dir) / "parquet_files")],
            cache_dir=str(Path(base_dir) / "hf_cache"),
        )

        logger.info("👍Dataset loaded successfully.")
        logger.info(f"Number of sequence: {len(data['train'])}")
        logger.info(f"Size of the vocabulary: {vocab_size}")
        logger.info(f"Number of tokens: {sum(data['train']['num_tokens'])}")

        from molcrawl.core.utils.image_manager import get_image_path

        plot_file = Path(get_image_path("genome_sequence", "genome_sequence_tokenized_lengths_dist.png"))
        if force or not plot_file.exists():
            if force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")

            if not create_distribution_plot(data):
                logger.warning("Distribution plot generation failed, but continuing...")
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")
            logger.info("Use --force option to regenerate plot.")

        return True

    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        return False


def process_clinvar_finetune(force: bool = False) -> bool:
    """Prepare the ClinVar language-model fine-tuning dataset.

    If ``$LEARNING_SOURCE_DIR/genome_sequence/clinvar/clinvar_sequences.csv``
    does not exist, it is generated automatically by downloading the
    ``gonzalobenegas/clinvar`` dataset from HuggingFace and embedding each
    variant in its GRCh38 genomic context (requires
    ``dataset/GCA_000001405.28_GRCh38.p13_genomic.fna`` in the project root).

    Tokenises the reference + variant sequences with the genome SentencePiece
    BPE tokenizer and saves a chunked HuggingFace DatasetDict to
    ``$LEARNING_SOURCE_DIR/genome_sequence/clinvar/training_ready_hf_dataset/``.

    Args:
        force: Re-prepare even if outputs already exist.

    Returns:
        True on success, False on failure.
    """
    from molcrawl.data.genome_sequence.dataset.clinvar.prepare_clinvar import prepare_clinvar

    output_dir = Path(CLINVAR_DIR)
    ready_marker = output_dir / "clinvar_prepare_complete.marker"

    if not force and ready_marker.exists():
        logger.info("👉ClinVar: dataset already prepared. Skipping (use --force to redo).")
        return True

    logger.info("👉ClinVar: Preparing training_ready_hf_dataset from clinvar_sequences.csv...")
    try:
        prepare_clinvar(
            source_file=CLINVAR_SOURCE_FILE,
            output_dir=output_dir,
            tokenizer_path=get_refseq_tokenizer_path(),
        )
        ready_marker.touch()
        logger.info("🎉 ClinVar dataset preparation completed successfully!")
        return True
    except Exception as exc:
        logger.error("ClinVar preparation failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Subset (Evo2 species list) flow
#
# A parallel 3-step pipeline driven by a subset CSV (assembly_accession +
# ftp_path columns). Lives alongside the legacy 4-step BPE pipeline above;
# selected at runtime by ``--subset NAME``.
#
#     <base_dir>/extracted_files/<acc>.fna.gz     ← subset step 1
#     <base_dir>/raw_files/<acc>.raw              ← subset step 2 (pure ACGT)
#     <base_dir>/parquet_{bert,gpt2}/<acc>.parquet ← subset step 3
# --------------------------------------------------------------------------- #


def resolve_subset_paths(cfg, subset_name: str) -> str:
    """Derive ``cfg.path_species`` (CSV) and ``cfg.output_dir`` (base) from a subset name.

    Returns the resolved base_dir. Modifies ``cfg`` in place.
    """
    csv_path = Path(PROJECT_ROOT) / "assets" / "genome_species_list" / "subsets" / f"{subset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Subset CSV not found: {csv_path}\n"
            f"Place the subset CSV under assets/genome_species_list/subsets/."
        )
    cfg.path_species = str(csv_path)
    base_dir = f"{GENOME_SEQUENCE_DIR}/{subset_name}"
    cfg.output_dir = base_dir
    cfg.subset_name = subset_name
    logger.info(f"[subset={subset_name}] path_species → {cfg.path_species}")
    logger.info(f"[subset={subset_name}] output_dir   → {base_dir}")
    return base_dir


def check_subset_progress_status(base_dir, models):
    """Four-step progress view for the subset flow (download → raw → parquet → arrow)."""
    download_marker = Path(base_dir) / "download_complete.marker"
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"
    parquet_to_arrow_marker = Path(base_dir) / "parquet_to_arrow_complete.marker"

    raw_files_dir = Path(base_dir) / "raw_files"

    logger.info("=== Subset Pipeline Progress ===")
    completed = 0

    if download_marker.exists():
        logger.info("✓ Step 1/4: accession download              - COMPLETED")
        completed += 1
    else:
        logger.info("⏳ Step 1/4: accession download              - PENDING")

    if fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("✓ Step 2/4: FASTA → raw (single-nucleotide) - COMPLETED")
        completed += 1
    else:
        logger.info("⏳ Step 2/4: FASTA → raw (single-nucleotide) - PENDING")

    parquet_ready = raw_to_parquet_marker.exists() and all(
        (Path(base_dir) / f"parquet_{m}").exists()
        and any((Path(base_dir) / f"parquet_{m}").glob("*.parquet"))
        for m in models
    )
    if parquet_ready:
        logger.info(f"✓ Step 3/4: raw → parquet ({'+'.join(models)})         - COMPLETED")
        completed += 1
    else:
        logger.info(f"⏳ Step 3/4: raw → parquet ({'+'.join(models)})         - PENDING")

    arrow_ready = parquet_to_arrow_marker.exists() and all(
        (Path(base_dir) / f"training_ready_hf_dataset_{m}" / "dataset_dict.json").exists()
        for m in models
    )
    if arrow_ready:
        logger.info(f"✓ Step 4/4: parquet → Arrow ({'+'.join(models)})        - COMPLETED")
        completed += 1
    else:
        logger.info(f"⏳ Step 4/4: parquet → Arrow ({'+'.join(models)})        - PENDING")

    logger.info(f"Progress: {completed}/4 steps completed")
    logger.info("================================")
    return completed == 4


def process1_subset_download(base_dir, csv_path, num_worker, verify_md5=True, force=False):
    """Subset Step 1: download exact assemblies listed in the subset CSV.

    Uses ``download_subset_from_csv`` (accession-exact, ftp_path-based) instead
    of the species-name-based legacy downloader.
    """
    from molcrawl.data.genome_sequence.dataset.refseq.download_by_accession import (
        download_subset_from_csv,
    )

    marker = Path(base_dir) / "download_complete.marker"
    if not force and marker.exists():
        logger.info("👉Subset Step1: accession download already completed. Skipping...")
        return True

    logger.info("👉Subset Step1: downloading exact assemblies from CSV...")
    logger.info(f" - CSV          : {csv_path}")
    logger.info(f" - base_dir     : {base_dir}")
    logger.info(f" - num_worker   : {num_worker}")
    logger.info(f" - verify_md5   : {verify_md5}")
    try:
        return download_subset_from_csv(
            csv_path=csv_path,
            base_dir=base_dir,
            num_worker=num_worker,
            verify_md5=verify_md5,
        )
    except Exception as e:
        logger.error(f"Subset Step1 (download) failed: {e}")
        return False


def process2_subset_fasta_to_raw(base_dir, num_worker, min_segment_len, force=False):
    """Subset Step 2: per-accession FASTA → raw (single-nucleotide, gz aware, pure ACGT)."""
    from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import (
        fasta_to_raw_per_accession,
    )

    marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    raw_dir = Path(base_dir) / "raw_files"
    if (
        not force
        and marker.exists()
        and raw_dir.exists()
        and any(raw_dir.glob("*.raw"))
    ):
        logger.info("👉Subset Step2: FASTA → raw already completed. Skipping...")
        return True

    logger.info("👉Subset Step2: FASTA → raw (per-accession, ACGT only)...")
    logger.info(f" - base_dir        : {base_dir}")
    logger.info(f" - num_worker      : {num_worker}")
    logger.info(f" - min_segment_len : {min_segment_len}")
    try:
        return fasta_to_raw_per_accession(
            base_dir=base_dir,
            num_worker=num_worker,
            min_segment_len=min_segment_len,
            force=force,
        )
    except Exception as e:
        logger.error(f"Subset Step2 (FASTA → raw) failed: {e}")
        return False


def process3_subset_raw_to_parquet(
    base_dir,
    models,
    bert_chunk_size,
    gpt2_chunk_size,
    num_worker,
    force=False,
):
    """Subset Step 3: per-accession raw → single-nucleotide parquet for each model.

    No MLM masking is baked into the parquet — the BERT trainer's dynamic
    ``DataCollatorForLanguageModeling`` handles masking at train time.
    """
    from molcrawl.data.genome_sequence.dataset.refseq.raw_to_parquet_single_nuc import (
        raw_to_parquet_per_accession,
    )

    marker = Path(base_dir) / "raw_to_parquet_complete.marker"
    parquet_dirs = [Path(base_dir) / f"parquet_{m}" for m in models]
    if (
        not force
        and marker.exists()
        and all(d.exists() and any(d.glob("*.parquet")) for d in parquet_dirs)
    ):
        logger.info("👉Subset Step3: raw → parquet already completed. Skipping...")
        return True

    logger.info("👉Subset Step3: raw → parquet (single-nucleotide tokenizer)...")
    logger.info(f" - base_dir        : {base_dir}")
    logger.info(f" - models          : {models}")
    logger.info(f" - bert_chunk_size : {bert_chunk_size}")
    logger.info(f" - gpt2_chunk_size : {gpt2_chunk_size}")
    logger.info(f" - num_worker      : {num_worker}")
    try:
        return raw_to_parquet_per_accession(
            base_dir=base_dir,
            models=tuple(models),
            bert_chunk_size=bert_chunk_size,
            gpt2_chunk_size=gpt2_chunk_size,
            num_worker=num_worker,
            force=force,
        )
    except Exception as e:
        logger.error(f"Subset Step3 (raw → parquet) failed: {e}")
        return False


def _contig_unit_split(
    accessions,
    contigs,
    valid_size=50_000,
    test_size=50_000,
    valid_frac=0.005,
    test_frac=0.005,
    target_total_windows=None,
    seed=42,
):
    """Assign window indices to train/valid/test by whole (accession, contig).

    Returns ``(train_idx, valid_idx, test_idx, stats)`` where each ``*_idx`` is
    a sorted list of row indices and ``stats`` carries group / window counts.
    All windows of one (accession, contig_id) group go to the same split, so
    the resulting splits are disjoint at contig granularity (no adjacent-window
    leak). ``target_total_windows`` (F2-c) drops whole groups first.
    """
    import random as _random

    acc = np.asarray(accessions, dtype=object).astype("U")
    con = np.asarray(contigs, dtype=object).astype("U")
    # Combined group key; \x1f (unit separator) cannot occur in an accession or
    # a FASTA contig id, so join/split is unambiguous.
    keys = np.char.add(np.char.add(acc, "\x1f"), con)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_groups = len(uniq)
    n_total = int(counts.sum())

    order = list(range(n_groups))
    _random.Random(seed).shuffle(order)

    # F2-c: drop whole groups (seeded order) until kept <= target.
    keep = np.ones(n_groups, dtype=bool)
    if target_total_windows is not None and n_total > target_total_windows:
        running = n_total
        for g in order:
            if running <= target_total_windows:
                break
            keep[g] = False
            running -= int(counts[g])
    n_kept = int(counts[keep].sum())

    n_valid = min(valid_size, max(1_000, int(n_kept * valid_frac)))
    n_test = min(test_size, max(1_000, int(n_kept * test_frac)))

    # 0=train, 1=valid, 2=test, -1=dropped (F2-c).
    assign = np.zeros(n_groups, dtype=np.int8)
    assign[~keep] = -1
    nv = nt = 0
    for g in order:
        if not keep[g]:
            continue
        c = int(counts[g])
        if nv + c <= n_valid:
            assign[g] = 1
            nv += c
        elif nt + c <= n_test:
            assign[g] = 2
            nt += c
        # else: stays train
    row_assign = assign[inv]
    train_idx = np.nonzero(row_assign == 0)[0].tolist()
    valid_idx = np.nonzero(row_assign == 1)[0].tolist()
    test_idx = np.nonzero(row_assign == 2)[0].tolist()

    stats = {
        "n_groups": n_groups,
        "n_total_windows": n_total,
        "n_kept_windows": n_kept,
        "n_dropped_groups": int((~keep).sum()),
        "valid_target": n_valid,
        "test_target": n_test,
    }
    return train_idx, valid_idx, test_idx, stats


def verify_contig_split_disjoint(ds_dict) -> dict:
    """Assert no (accession, contig_id) group is shared across splits.

    Returns ``{'ok': bool, 'shared': [...], 'per_split_groups': {...}}``. Used
    by the F2 verify gate (verify②: train and val/test are contig-disjoint).
    """
    group_sets = {}
    for split in ("train", "valid", "test"):
        if split not in ds_dict:
            continue
        d = ds_dict[split]
        pairs = set(zip(d["accession"], d["contig_id"]))
        group_sets[split] = pairs
    shared = set()
    splits = list(group_sets)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            shared |= group_sets[splits[i]] & group_sets[splits[j]]
    return {
        "ok": len(shared) == 0,
        "shared": sorted(shared),
        "per_split_groups": {s: len(g) for s, g in group_sets.items()},
    }


def process4_subset_parquet_to_arrow(
    base_dir,
    models,
    valid_size=50_000,
    test_size=50_000,
    valid_frac=0.005,
    test_frac=0.005,
    remove_parquet=False,
    force=False,
    target_total_windows=None,
    split_seed=42,
):
    """Subset Step 4: parquet_{model}/ → training_ready_hf_dataset_{model}/ (HF Arrow).

    Converts the per-model snappy parquet produced by Step 3 into the
    standard HuggingFace ``DatasetDict.save_to_disk()`` layout that
    ``molcrawl/models/bert/main.py`` (and the GPT-2 trainer) consumes via
    ``load_from_disk``. The legacy genome trainer indexes the loaded object
    as ``dataset["train"]`` / ``dataset["test"]`` (or ``["valid"]``), so the
    output MUST be a DatasetDict — a flat single-split Dataset would crash
    with ``KeyError: 'train'`` before any training step runs.

    Split (F2-a, contig-unit)
    -------------------------
    Held-out membership is assigned by **whole (accession, contig_id) group**,
    not by window offset. Every window of a given chromosome / scaffold lands
    entirely in one split, so a genome's adjacent windows can never straddle
    train and eval (the old ``random.Random(42)`` per-window shuffle leaked
    neighbouring context). A species with several contigs still appears in both
    train and eval — only its *contigs* are partitioned — which keeps the
    composition-comparison independent variable intact. Groups are visited in a
    seeded shuffle and packed into valid then test with a fits-in-remaining rule
    (skip-and-continue), so val/test draw a little from many contigs/species
    rather than being dominated by one large one.

    F2-c trim
    ---------
    When ``target_total_windows`` is set and the subset exceeds it, whole groups
    are dropped (contig granularity, seeded order) until the kept window count
    is at or just below the target — equalising realized budget across subsets
    without breaking the contig-disjoint split.

    Args:
        base_dir       : subset base dir (.../<subset>/).
        models         : list of model names, e.g. ["bert", "gpt2"].
        valid_size / test_size : hard caps on held-out split sizes (rows).
        valid_frac / test_frac : fractional sizing (used when smaller than caps).
        remove_parquet : drop parquet_<model>/ after successful conversion.
        force          : overwrite existing Arrow output for a model.
        target_total_windows : F2-c realized-window budget; ``None`` = no trim.
        split_seed     : seed for the group shuffle / trim order.
    """
    from datasets import DatasetDict, load_dataset

    marker = Path(base_dir) / "parquet_to_arrow_complete.marker"

    all_arrow_done = all(
        (Path(base_dir) / f"training_ready_hf_dataset_{m}" / "dataset_dict.json").exists()
        for m in models
    )
    if not force and marker.exists() and all_arrow_done:
        logger.info("👉Subset Step4: parquet → Arrow already completed. Skipping...")
        return True

    logger.info("👉Subset Step4: parquet → training_ready_hf_dataset (HF Arrow / DatasetDict)...")
    logger.info(f" - base_dir       : {base_dir}")
    logger.info(f" - models         : {models}")
    logger.info(f" - valid/test cap : {valid_size:,} / {test_size:,} rows")
    logger.info(f" - valid/test frac: {valid_frac} / {test_frac}")
    logger.info(f" - remove_parquet : {remove_parquet}")

    success = True
    for model_name in models:
        parquet_dir = Path(base_dir) / f"parquet_{model_name}"
        arrow_dir = Path(base_dir) / f"training_ready_hf_dataset_{model_name}"

        if not parquet_dir.exists() or not any(parquet_dir.glob("*.parquet")):
            logger.error(f"parquet_{model_name}/ missing or empty. Run Step 3 first.")
            success = False
            continue

        if not force and (arrow_dir / "dataset_dict.json").exists():
            logger.info(f" - training_ready_hf_dataset_{model_name}/ already present. Skipping.")
            continue

        try:
            parquet_files = sorted(str(p) for p in parquet_dir.glob("*.parquet"))
            logger.info(
                f" - {model_name}: {len(parquet_files)} parquet files → "
                f"training_ready_hf_dataset_{model_name}/"
            )

            ds = load_dataset(
                "parquet",
                data_files={"train": parquet_files},
                split="train",
            )

            # F2-a contig-unit split. Group every window by its
            # (accession, contig_id) provenance (stamped in Phase 3) and hold
            # out whole groups, so adjacent windows of one chromosome/scaffold
            # never straddle train and eval. Groups are shuffled with a fixed
            # seed and packed into valid then test with a fits-in-remaining rule
            # (skip-and-continue) so the held-out sets draw a little from many
            # contigs/species. F2-c trims whole groups to a realized-window
            # budget before assignment.
            for col in ("accession", "contig_id"):
                if col not in ds.column_names:
                    raise ValueError(
                        f"{model_name}: parquet lacks '{col}' column — regenerate "
                        f"Phase 2/3 with the contig-aware pipeline (F2-a)."
                    )

            train_idx, valid_idx, test_idx, stats = _contig_unit_split(
                accessions=ds["accession"],
                contigs=ds["contig_id"],
                valid_size=valid_size,
                test_size=test_size,
                valid_frac=valid_frac,
                test_frac=test_frac,
                target_total_windows=target_total_windows,
                seed=split_seed,
            )
            n_train, n_valid, n_test = len(train_idx), len(valid_idx), len(test_idx)
            if n_train <= 0 or n_valid <= 0 or n_test <= 0:
                raise ValueError(
                    f"{model_name}: contig-unit split produced an empty split "
                    f"(train={n_train} valid={n_valid} test={n_test}); dataset "
                    f"has {stats['n_groups']} contig groups — too few to split."
                )

            ds_dict = DatasetDict({
                "train": ds.select(train_idx),
                "valid": ds.select(valid_idx),
                "test":  ds.select(test_idx),
            })

            arrow_dir.mkdir(parents=True, exist_ok=True)
            ds_dict.save_to_disk(str(arrow_dir))
            logger.info(
                f"   ✓ {model_name}: saved train={n_train:,} valid={n_valid:,} "
                f"test={n_test:,} over {stats['n_groups']:,} contigs "
                f"(kept {stats['n_kept_windows']:,}/{stats['n_total_windows']:,} "
                f"windows; dropped {stats['n_dropped_groups']:,} groups for F2-c) "
                f"(DatasetDict)"
            )

            if remove_parquet:
                import shutil
                shutil.rmtree(str(parquet_dir))
                logger.info(f"   ✓ removed intermediate parquet_{model_name}/")

        except Exception as e:
            logger.error(f"parquet → Arrow conversion failed (model={model_name}): {e}")
            success = False

    if success:
        marker.touch()
        logger.info("Subset Step4 (parquet → Arrow) completed.")

    return success


def run_subset_flow(cfg, force=False) -> None:
    """End-to-end subset pipeline (download → raw → parquet) driven by ``cfg``."""
    base_dir = cfg.output_dir
    models = list(cfg.models) if cfg.models else ["bert", "gpt2"]

    all_done = check_subset_progress_status(base_dir, models)
    if all_done and not force:
        logger.info("All subset steps already completed. Use --force to re-run.")
        return

    if not process1_subset_download(
        base_dir=base_dir,
        csv_path=cfg.path_species,
        num_worker=cfg.num_worker,
        verify_md5=getattr(cfg, "verify_md5", True),
        force=force,
    ):
        logger.error("Subset Step1 (download) failed. Stopping.")
        exit(1)

    if not process2_subset_fasta_to_raw(
        base_dir=base_dir,
        num_worker=cfg.num_worker,
        min_segment_len=getattr(cfg, "min_segment_len", 100),
        force=force,
    ):
        logger.error("Subset Step2 (FASTA → raw) failed. Stopping.")
        exit(1)

    if not process3_subset_raw_to_parquet(
        base_dir=base_dir,
        models=models,
        bert_chunk_size=getattr(cfg, "bert_chunk_size", 510),
        gpt2_chunk_size=getattr(cfg, "gpt2_chunk_size", 1024),
        num_worker=cfg.num_worker,
        force=force,
    ):
        logger.error("Subset Step3 (raw → parquet) failed. Stopping.")
        exit(1)

    if not process4_subset_parquet_to_arrow(
        base_dir=base_dir,
        models=models,
        valid_size=getattr(cfg, "valid_size", 50_000),
        test_size=getattr(cfg, "test_size", 50_000),
        valid_frac=getattr(cfg, "valid_frac", 0.005),
        test_frac=getattr(cfg, "test_frac", 0.005),
        remove_parquet=getattr(cfg, "remove_parquet", False),
        force=force,
    ):
        logger.error("Subset Step4 (parquet → Arrow) failed. Stopping.")
        exit(1)

    logger.info("🎉 Subset pipeline completed.")


def main():
    """Main function to orchestrate the genome sequence dataset preparation"""
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["refseq", "clinvar"],
        default=["refseq"],
        help="Which dataset(s) to prepare (default: refseq).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reprocessing even if files exist",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistics generation and plotting",
    )
    parser.add_argument(
        "--subset",
        default=None,
        metavar="NAME",
        help=(
            "Subset name (e.g. mammal_centered, global_random_seed1). When set, "
            "path_species and output_dir are derived from "
            "assets/genome_species_list/subsets/<NAME>.csv, and the 3-step "
            "single-nucleotide pipeline runs in place of the legacy 4-step BPE flow."
        ),
    )
    args = parser.parse_args()

    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    setup_logging(GENOME_SEQUENCE_DIR)

    # ── ClinVar fine-tuning branch ─────────────────────────────────────────────
    if "clinvar" in args.datasets:
        if not process_clinvar_finetune(force=args.force):
            exit(1)
        if "refseq" not in args.datasets:
            return

    # ── Subset (Evo2 species list) branch ──────────────────────────────────────
    if args.subset:
        resolve_subset_paths(cfg, args.subset)
        run_subset_flow(cfg, force=args.force)
        return

    # ── RefSeq pretraining branch (original pipeline) ─────────────────────────
    # base_dir for heavy processing (specify in config if you want to release to local SSD etc.)
    base_dir = GENOME_SEQUENCE_DIR + getattr(cfg, "local_base_dir", "scratch")
    logger.info(f"Using base_dir: {base_dir}")

    # Check progress
    all_completed = check_progress_status(base_dir)

    if all_completed and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        if not args.skip_stats:
            process5_generate_statistics(base_dir, cfg.vocab_size, args.force)
        return

    success = True

    # Process 1: Download RefSeq dataset
    success &= process1_download_refseq(
        base_dir,
        cfg.path_species,
        cfg.num_worker,
        species_timeout=getattr(cfg, "species_timeout", 30 * 60),
        max_retries=getattr(cfg, "max_retries", 2),
        force=args.force,
    )
    if not success:
        logger.error("Process 1 failed. Stopping execution.")
        exit(1)

    # Process 2: Convert FASTA to raw text
    success &= process2_fasta_to_raw(base_dir, cfg.num_worker, cfg.max_lines_per_file, args.force)
    if not success:
        logger.error("Process 2 failed. Stopping execution.")
        exit(1)

    # Process 3: Train tokenizer
    success &= process3_train_tokenizer(
        base_dir,
        cfg.vocab_size,
        cfg.max_lines_per_file,
        cfg.input_sentence_size,
        getattr(cfg, "max_sentence_length", 4192),
        args.force,
    )
    if not success:
        logger.error("Process 3 failed. Stopping execution.")
        exit(1)

    # Process 4: Convert raw to Parquet (parallel & batch settings)
    num_proc_parquet = getattr(cfg, "num_proc_parquet", cfg.num_worker)
    batch_size_parquet = getattr(cfg, "parquet_batch_size", 512)

    success &= process4_raw_to_parquet(
        base_dir,
        num_proc=num_proc_parquet,
        batch_size=batch_size_parquet,
        force=args.force,
    )
    if not success:
        logger.error("Process 4 failed. Stopping execution.")
        exit(1)

    # Process 5: Generate statistics and plots
    if not args.skip_stats:
        success &= process5_generate_statistics(base_dir, cfg.vocab_size, args.force)
        if not success:
            logger.error("Process 5 failed. Dataset preparation completed but statistics generation failed.")
            exit(1)

    logger.info("🎉 Genome sequence dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
