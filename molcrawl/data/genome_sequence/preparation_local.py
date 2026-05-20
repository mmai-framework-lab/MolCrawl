"""Prepare ``genome_sequence`` training data from a locally-staged FASTA file.

This is a sibling pipeline to :mod:`molcrawl.data.genome_sequence.preparation`,
which downloads RefSeq via ``ncbi_genome_download``. Use this entry point when a
specialist has already curated FASTA files outside of the project (e.g. a
GRCh38 reference with ``chr22`` held out for evaluation, staged under
``/lustre/home/kojima-t/data/species_links/``).

Pipeline:

1. **Stage** the input ``*.fna`` or ``*.fna.gz`` into
   ``${LEARNING_SOURCE_DIR}/genome_sequence/extracted_files/<group>/<species>/``,
   decompressing gzip on the fly. Then mark Process 1 (download) complete so
   the rest of the pipeline can resume cleanly via the existing marker logic.
2. **Process 2** — ``fasta_to_raw``: split FASTA into chunked ``.raw`` files.
3. **Process 3** — train the SentencePiece BPE tokenizer.
4. **Process 4** — ``raw_to_parquet``: tokenise into parquet shards.
5. **Process 5** (optional) — statistics + distribution plot.

The download-specific config fields in
:class:`~molcrawl.data.genome_sequence.utils.config.RefSeqPreparationConfig`
(``path_species``, ``species_timeout``, ``max_retries``) are ignored here.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

from molcrawl.core.base import setup_logging
from molcrawl.core.paths import GENOME_SEQUENCE_DIR
from molcrawl.data.genome_sequence.preparation import (
    process2_fasta_to_raw,
    process3_train_tokenizer,
    process4_raw_to_parquet,
    process5_generate_statistics,
)
from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)

NGD_GROUPS = (
    "archaea",
    "bacteria",
    "fungi",
    "invertebrate",
    "metagenomes",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
    "viral",
)


def stage_local_fasta(input_path: Path, base_dir: Path, species: str, group: str, force: bool = False) -> Path:
    """Stage a single FASTA into ``extracted_files/<group>/<species>/``.

    Decompresses ``.gz`` on the fly. Idempotent unless ``force`` is set.
    """
    target_dir = base_dir / "extracted_files" / group / species
    target_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".gz":
        target = target_dir / input_path.with_suffix("").name
    else:
        target = target_dir / input_path.name

    if target.exists() and not force:
        logger.info("Already staged (use --force to re-stage): %s", target)
        return target

    if input_path.suffix == ".gz":
        logger.info("Decompressing %s → %s", input_path, target)
        with gzip.open(input_path, "rb") as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=64 * 1024 * 1024)
    else:
        logger.info("Copying %s → %s", input_path, target)
        shutil.copyfile(input_path, target)

    return target


def mark_download_complete(base_dir: Path) -> None:
    """Touch the download marker so Process 2-4's resume logic treats Process 1 as done."""
    marker = base_dir / "download_complete.marker"
    if marker.exists():
        return
    marker.touch()
    logger.info("Marked download_complete.marker (Process 1 skipped — local FASTA staged).")


def main() -> None:
    parser = ArgumentParser(
        description="Prepare genome_sequence training data from a locally-staged FASTA file.",
    )
    parser.add_argument("config", help="Path to genome_sequence config yaml.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the local FASTA file (.fna or .fna.gz).",
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Species directory name under extracted_files/<group>/ (e.g. homo_sapiens).",
    )
    parser.add_argument(
        "--group",
        required=True,
        choices=NGD_GROUPS,
        help="NCBI taxonomic group (mirrors ncbi_genome_download).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-stage the FASTA and re-run all downstream processes.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip Process 5 (statistics and distribution plot).",
    )
    parser.add_argument(
        "--only-stage",
        action="store_true",
        help="Only stage the FASTA; do not run Process 2-4.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise SystemExit(f"--input does not point to a file: {input_path}")

    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    setup_logging(GENOME_SEQUENCE_DIR)

    base_dir = Path(GENOME_SEQUENCE_DIR + (getattr(cfg, "local_base_dir", "") or ""))
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using base_dir: %s", base_dir)

    staged = stage_local_fasta(input_path, base_dir, args.species, args.group, force=args.force)
    logger.info("Staged FASTA: %s (%.2f GB)", staged, staged.stat().st_size / 1024 / 1024 / 1024)

    mark_download_complete(base_dir)

    if args.only_stage:
        logger.info("--only-stage given; stopping after staging.")
        return

    base_dir_s = str(base_dir)

    if not process2_fasta_to_raw(base_dir_s, cfg.num_worker, cfg.max_lines_per_file, args.force):
        logger.error("Process 2 (FASTA → raw) failed.")
        raise SystemExit(1)

    if not process3_train_tokenizer(
        base_dir_s, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size, args.force,
    ):
        logger.error("Process 3 (tokenizer training) failed.")
        raise SystemExit(1)

    num_proc_parquet = getattr(cfg, "num_proc_parquet", None) or cfg.num_worker
    parquet_batch_size = getattr(cfg, "parquet_batch_size", None) or 512
    if not process4_raw_to_parquet(
        base_dir_s, num_proc=num_proc_parquet, batch_size=parquet_batch_size, force=args.force,
    ):
        logger.error("Process 4 (raw → parquet) failed.")
        raise SystemExit(1)

    if not args.skip_stats:
        process5_generate_statistics(base_dir_s, cfg.vocab_size, args.force)

    logger.info("🎉 Local genome_sequence preparation completed.")


if __name__ == "__main__":
    main()
