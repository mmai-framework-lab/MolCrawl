"""Prepare ``genome_sequence`` training data from pre-staged local FASTA files.

This is a sibling pipeline to :mod:`molcrawl.data.genome_sequence.preparation`,
which downloads RefSeq via ``ncbi_genome_download``. Use this entry point when a
specialist has already curated FASTA files outside of the project (e.g. a
GRCh38 reference with ``chr22`` held out for evaluation.

Two modes:

* ``--input <fna[.gz]>`` (single-file legacy mode)
    Stage one FASTA into ``${LEARNING_SOURCE_DIR}/genome_sequence/extracted_files/
    <group>/<species>/``, then run Process 2-5 of the existing shared pipeline.

* ``--input-dir <species_links>`` (per-species batch mode)
    Treat each subdirectory of ``species_links/`` as one species. For every species
    write a self-contained "ready to train" payload **alongside the original FASTA**:

        species_links/
        ├── spm_tokenizer.model     (one shared BPE tokenizer trained across all species)
        ├── spm_tokenizer.vocab
        ├── <Species>/
        │   ├── <FASTA>.fna.gz             (pre-existing input — untouched)
        │   └── <FASTA stem>/              ← new per-species directory
        │       ├── <FASTA stem>.fna       (decompressed source)
        │       ├── raw_files/             (N-split ACGT chunks)
        │       └── parquet_files          (single parquet, ready to feed a DataLoader)
        └── ...

    The shared tokenizer is trained once after every species' raw_files exist, so
    the BPE vocabulary reflects cross-species diversity. Per-species parquets are
    then tokenized with that shared tokenizer.

Download-specific config fields in
:class:`~molcrawl.data.genome_sequence.utils.config.RefSeqPreparationConfig`
(``path_species``, ``species_timeout``, ``max_retries``) are ignored here.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import logging
import os
import shutil
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import List, Tuple

from molcrawl.core.base import setup_logging
from molcrawl.core.paths import GENOME_SEQUENCE_DIR
from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import (
    parse_fasta_to_raw_sequence,
)
from molcrawl.data.genome_sequence.dataset.tokenizer import tokenize_function
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fasta_stem(input_path: Path) -> str:
    """Strip trailing ``.gz`` then ``.fna`` from a FASTA filename."""
    name = input_path.name
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    if name.endswith(".fna"):
        name = name[: -len(".fna")]
    return name


def _decompress_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=64 * 1024 * 1024)
    else:
        shutil.copyfile(src, dst)


def mark_download_complete(base_dir: Path) -> None:
    """Touch the download marker so Process 2-4's resume logic treats Process 1 as done."""
    marker = base_dir / "download_complete.marker"
    if marker.exists():
        return
    marker.touch()
    logger.info("Marked download_complete.marker (Process 1 skipped — local FASTA staged).")


# ---------------------------------------------------------------------------
# Single-file legacy mode (--input)
# ---------------------------------------------------------------------------


def stage_local_fasta(input_path: Path, base_dir: Path, species: str, group: str, force: bool = False) -> Path:
    """Stage a single FASTA into ``extracted_files/<group>/<species>/``."""
    target_dir = base_dir / "extracted_files" / group / species
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (input_path.with_suffix("").name if input_path.suffix == ".gz" else input_path.name)

    if target.exists() and not force:
        logger.info("Already staged (use --force to re-stage): %s", target)
        return target

    logger.info("%s %s → %s", "Decompressing" if input_path.suffix == ".gz" else "Copying", input_path, target)
    _decompress_or_copy(input_path, target)
    return target


# ---------------------------------------------------------------------------
# Per-species batch mode (--input-dir)
# ---------------------------------------------------------------------------


def find_species_inputs(species_links: Path) -> List[Tuple[Path, Path]]:
    """Return ``[(input_fasta, per_species_outdir)]`` for every species.

    ``per_species_outdir`` is ``species_links/<Species>/<FASTA stem>/``. When a
    species has multiple FASTAs (e.g. a held-out ``*_no_chr22.fna.gz`` alongside
    the full reference), files whose name contains ``_no_chr`` win.
    """
    pairs: List[Tuple[Path, Path]] = []
    for sp_dir in sorted(species_links.iterdir()):
        if not sp_dir.is_dir():
            continue
        candidates = sorted(sp_dir.glob("*.fna.gz")) + sorted(sp_dir.glob("*.fna"))
        if not candidates:
            continue
        held_out = [p for p in candidates if "_no_chr" in p.name]
        chosen = held_out[0] if held_out else candidates[0]
        pairs.append((chosen, sp_dir / _fasta_stem(chosen)))
    return pairs


def stage_one_species(input_fasta: Path, species_dir: Path, force: bool = False) -> Path:
    """Decompress ``input_fasta`` into ``species_dir/<stem>.fna`` and return that path."""
    species_dir.mkdir(parents=True, exist_ok=True)
    fna_target = species_dir / f"{_fasta_stem(input_fasta)}.fna"

    if fna_target.exists() and not force:
        return fna_target

    logger.info("Staging %s → %s", input_fasta, fna_target)
    _decompress_or_copy(input_fasta, fna_target)
    return fna_target


def stage_batch(pairs: List[Tuple[Path, Path]], force: bool = False, workers: int = 8) -> List[Path]:
    """Stage every (input_fasta, species_dir) pair in parallel."""

    def _one(pair: Tuple[Path, Path]) -> Path:
        src, sp_dir = pair
        return stage_one_species(src, sp_dir, force=force)

    staged: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, path in enumerate(ex.map(_one, pairs), start=1):
            staged.append(path)
            if i % 25 == 0 or i == len(pairs):
                logger.info("  staged %d/%d", i, len(pairs))
    return staged


def fasta_to_raw_for_species(species_dir: Path, num_worker: int, max_lines_per_file: int, force: bool = False) -> Path:
    """Convert all ``*.fna`` under ``species_dir`` into ``species_dir/raw_files/``."""
    raw_dir = species_dir / "raw_files"
    if raw_dir.exists() and any(raw_dir.glob("*.raw")) and not force:
        logger.info("[%s] raw_files already populated, skipping", species_dir.name)
        return raw_dir
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parse_fasta_to_raw_sequence(species_dir, raw_dir, num_worker, max_lines_per_file)
    return raw_dir


def train_shared_tokenizer(
    species_dirs: List[Path],
    output_root: Path,
    vocab_size: int,
    max_lines_per_file: int,
    input_sentence_size: int,
    force: bool = False,
) -> Path:
    """Train one BPE tokenizer across raw_files from every species."""
    import numpy as np
    import sentencepiece as spm

    model_path = output_root / "spm_tokenizer.model"
    if model_path.exists() and not force:
        logger.info("Shared tokenizer already exists, skipping training: %s", model_path)
        return model_path

    all_raw: List[Path] = []
    for sp_dir in species_dirs:
        all_raw.extend((sp_dir / "raw_files").glob("*.raw"))
    if not all_raw:
        raise RuntimeError("No raw files found across species — Process 2 must run first.")

    n_sample = max(1, int(input_sentence_size / max_lines_per_file) * 2)
    sampled = [all_raw[i] for i in np.random.permutation(len(all_raw))[:n_sample]]
    logger.info(
        "Training shared tokenizer: vocab=%d, sampling %d of %d raw files (input_sentence_size=%d)",
        vocab_size, len(sampled), len(all_raw), input_sentence_size,
    )

    spm.SentencePieceTrainer.train(
        input=",".join(str(f) for f in sampled),
        normalization_rule_name="identity",
        model_type="bpe",
        model_prefix=str(output_root / "spm_tokenizer"),
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=True,
        max_sentencepiece_length=50,
        split_by_whitespace=False,
        add_dummy_prefix=False,
    )
    return model_path


def tokenize_species_to_parquet(
    species_dir: Path,
    tokenizer_path: Path,
    num_proc: int | None = None,
    batch_size: int = 512,
    force: bool = False,
) -> Path:
    """Tokenise ``species_dir/raw_files/*.raw`` into ``species_dir/parquet_files``."""
    from datasets import load_dataset
    import sentencepiece as spm

    parquet_path = species_dir / "parquet_files"
    if parquet_path.exists() and not force:
        logger.info("[%s] parquet_files already exists, skipping tokenisation", species_dir.name)
        return parquet_path

    raw_dir = species_dir / "raw_files"
    if not raw_dir.exists() or not any(raw_dir.glob("*.raw")):
        raise RuntimeError(f"No raw files at {raw_dir}; cannot tokenise {species_dir.name}.")

    # Per-process hf_cache so that overlapping parallel jobs (--species-range
    # fan-out) cannot trample each other's tempfiles. Job A's cleanup used to
    # rmtree(species_dir/hf_cache) while Job B was still writing tempfiles
    # inside it, producing a FileNotFoundError on the meet-in-the-middle
    # species (jobs 18154/18155/18156 on 2026-05-25 hit this exactly).
    hf_cache = species_dir / f".hf_cache.pid{os.getpid()}"

    data = load_dataset(
        "text",
        data_dir=str(raw_dir),
        cache_dir=str(hf_cache),
        split="train",
    )

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    def batched_tokenize(batch, tokenizer):
        outputs = [tokenize_function({"text": t}, tokenizer=tokenizer) for t in batch["text"]]
        keys = outputs[0].keys()
        return {k: [o[k] for o in outputs] for k in keys}

    tokenized = data.map(
        partial(batched_tokenize, tokenizer=tokenizer),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
    )
    tokenized.to_parquet(str(parquet_path))

    # hf_cache is a transient build artefact; the parquet supersedes it. Only
    # touch our own PID-suffixed cache so we never reach across processes.
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)

    return parquet_path


def run_batch_pipeline(
    pairs: List[Tuple[Path, Path]],
    output_root: Path,
    cfg,
    stage_workers: int,
    force: bool,
) -> None:
    """Drive the per-species batch pipeline end-to-end."""
    species_dirs = [sp_dir for _, sp_dir in pairs]

    logger.info("=== Stage 1/4: decompress %d species ===", len(pairs))
    stage_batch(pairs, force=force, workers=stage_workers)

    logger.info("=== Stage 2/4: fasta_to_raw per species ===")
    for i, sp_dir in enumerate(species_dirs, start=1):
        logger.info("[%d/%d] fasta_to_raw %s", i, len(species_dirs), sp_dir.name)
        fasta_to_raw_for_species(sp_dir, cfg.num_worker, cfg.max_lines_per_file, force=force)

    logger.info("=== Stage 3/4: train shared tokenizer at %s ===", output_root)
    tokenizer_path = train_shared_tokenizer(
        species_dirs, output_root, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size, force=force,
    )

    logger.info("=== Stage 4/4: tokenise per species → parquet ===")
    num_proc = getattr(cfg, "num_proc_parquet", None) or cfg.num_worker
    batch_size = getattr(cfg, "parquet_batch_size", None) or 512
    for i, sp_dir in enumerate(species_dirs, start=1):
        logger.info("[%d/%d] tokenise %s", i, len(species_dirs), sp_dir.name)
        tokenize_species_to_parquet(sp_dir, tokenizer_path, num_proc=num_proc, batch_size=batch_size, force=force)

    logger.info("🎉 Per-species batch pipeline completed for %d species.", len(species_dirs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(
        description="Prepare genome_sequence training data from locally-staged FASTA files.",
    )
    parser.add_argument("config", help="Path to genome_sequence config yaml.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input",
        help="Path to a single local FASTA file (.fna or .fna.gz). Requires --species.",
    )
    src.add_argument(
        "--input-dir",
        help=(
            "Path to a species_links-style directory. Output is written *in place* under "
            "<input-dir>/<Species>/<FASTA stem>/ with a shared spm_tokenizer at <input-dir>/."
        ),
    )
    parser.add_argument(
        "--species",
        help="Species directory name (required with --input; ignored with --input-dir).",
    )
    parser.add_argument(
        "--group",
        choices=NGD_GROUPS,
        help="NCBI taxonomic group (required with --input; ignored with --input-dir).",
    )
    parser.add_argument(
        "--stage-workers",
        type=int,
        default=8,
        help="Parallel workers for --input-dir decompression (default: 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-stage and re-run downstream processes from scratch.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip Process 5 (single-file mode only).",
    )
    parser.add_argument(
        "--only-stage",
        action="store_true",
        help="Only stage the FASTA(s); do not run downstream processes.",
    )
    parser.add_argument(
        "--species-range",
        metavar="START:END",
        help=(
            "Python-slice subset of the discovered species list (e.g. '65:130'). Only with "
            "--input-dir. Used to fan out a long batch across multiple SLURM jobs. The shared "
            "tokenizer must already exist before parallel jobs run; otherwise each job would "
            "train a different tokenizer from its own slice."
        ),
    )
    args = parser.parse_args()

    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    setup_logging(GENOME_SEQUENCE_DIR)

    # --- Batch mode (per-species output under species_links/) ---
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.is_dir():
            raise SystemExit(f"--input-dir is not a directory: {input_dir}")
        pairs = find_species_inputs(input_dir)
        if not pairs:
            raise SystemExit(f"No species FASTAs discovered under {input_dir}")
        logger.info("Discovered %d species under %s", len(pairs), input_dir)

        if args.species_range:
            try:
                start_s, end_s = args.species_range.split(":", 1)
                start = int(start_s) if start_s else None
                end = int(end_s) if end_s else None
            except ValueError as exc:
                raise SystemExit(f"--species-range must be 'START:END' integers: {exc}")
            full_n = len(pairs)
            pairs = pairs[start:end]
            logger.info(
                "Applying --species-range %s → %d/%d species selected",
                args.species_range, len(pairs), full_n,
            )

        if args.only_stage:
            stage_batch(pairs, force=args.force, workers=args.stage_workers)
            logger.info("--only-stage given; stopping after decompression.")
            return

        run_batch_pipeline(pairs, input_dir, cfg, args.stage_workers, args.force)
        return

    # --- Single-file legacy mode (writes into LEARNING_SOURCE_DIR) ---
    if not args.species or not args.group:
        raise SystemExit("--input requires both --species and --group.")

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise SystemExit(f"--input does not point to a file: {input_path}")

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
        raise SystemExit(1)
    if not process3_train_tokenizer(
        base_dir_s, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size, args.force,
    ):
        raise SystemExit(1)
    num_proc_parquet = getattr(cfg, "num_proc_parquet", None) or cfg.num_worker
    parquet_batch_size = getattr(cfg, "parquet_batch_size", None) or 512
    if not process4_raw_to_parquet(
        base_dir_s, num_proc=num_proc_parquet, batch_size=parquet_batch_size, force=args.force,
    ):
        raise SystemExit(1)
    if not args.skip_stats:
        process5_generate_statistics(base_dir_s, cfg.vocab_size, args.force)
    logger.info("🎉 Local genome_sequence preparation completed.")


if __name__ == "__main__":
    main()
