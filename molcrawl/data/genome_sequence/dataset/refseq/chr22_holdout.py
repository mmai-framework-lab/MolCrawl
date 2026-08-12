"""chr22 hold-out for the subset flow (F2-b).

ClinVar / gnomAD variant-effect evaluation uses human chromosome 22. To keep
that evaluation uncontaminated, the human assembly must enter the pretraining
corpus with chr22 removed. The mammalian-RefSeq ``preparation_local.py`` flow
already prefers a ``*_no_chr22`` FASTA (``find_species_inputs``); the subset /
``download_by_accession`` flow had no equivalent. This module supplies it,
**keyed on the assembly accession** (not the subset name) so the substitution
fires in whichever subset the human genome is drawn into.

The chr22-excluded FASTA itself is produced by the upstream preprocessing tool
(``bin/run_preprocess_excl_chr22.sh`` / ``preprocess_fasta.py``); this module
only decides when to use it and stages it into ``extracted_files/``.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Human GRCh38.p14 RefSeq assembly. Its chr22 is what ClinVar/gnomAD eval uses,
# so this is the only accession whose chr22 leaking into pretraining would
# contaminate the held-out evaluation.
HUMAN_CHR22_ACCESSION = "GCF_000001405.40"

# GRCh38 RefSeq contig id for chromosome 22 (used by the verify gate to assert
# no chr22 window survived).
CHR22_CONTIG_IDS = frozenset({"NC_000022.11"})

# Every human chromosome the downstream evaluation expects to be unseen, not
# just the one this module can substitute a FASTA for.
#
# The two are not the same set, and the difference is worth stating plainly:
# the FASTA-substitution path below handles chr22 only, because the upstream
# preprocessing tool emits a `_no_chr22` file and nothing wider. chr21/X/Y are
# absent from the prepared corpus by a different route, so nothing here enforces
# them. Verified 2026-08-05: no window from any of the four survives in either
# the intermediate parquet or the training_ready splits of all 21 subsets.
#
# Matched by prefix rather than exact id so an assembly patch bump (the trailing
# version, e.g. NC_000022.11 -> .12) cannot silently slip past the check. These
# accessions are unique to human, so a prefix cannot collide with another
# species.
HELDOUT_CONTIG_PREFIXES = {
    "chr21": "NC_000021",
    "chr22": "NC_000022",
    "chrX": "NC_000023",
    "chrY": "NC_000024",
}

_HOLDOUT_MARKER = "_no_chr22"


def find_heldout_contigs(contig_ids):
    """Return {chromosome: [matching contig ids]} for held-out chromosomes present.

    Empty means the corpus is clean. Intended as a gate over a prepared split's
    ``contig_id`` column, which is why it takes ids rather than a dataset.
    """
    found = {}
    for name, prefix in sorted(HELDOUT_CONTIG_PREFIXES.items()):
        hits = sorted(c for c in contig_ids if str(c).startswith(prefix))
        if hits:
            found[name] = hits
    return found


def assert_no_heldout_contigs(contig_ids, where=""):
    """Raise if any held-out human chromosome appears in ``contig_ids``.

    Note this can only speak for data that carries ``contig_id`` at all: exports
    predating the contig-unit split have no such column, and there the check
    passes without having verified anything. Establish the schema first (see
    scripts/verify_genome_training_source.py) rather than relying on this alone.
    """
    found = find_heldout_contigs(contig_ids)
    if found:
        detail = "; ".join(f"{name}: {ids[:5]}" for name, ids in found.items())
        raise ValueError(f"held-out chromosome(s) present{' in ' + where if where else ''} -- {detail}")


def is_chr22_holdout_accession(accession: str) -> bool:
    """True if ``accession`` is the human assembly that needs chr22 held out."""
    return str(accession).strip() == HUMAN_CHR22_ACCESSION


def find_holdout_fasta(holdout_pool_dir: Union[str, Path]) -> Optional[Path]:
    """Return a ``*_no_chr22.fna[.gz]`` FASTA from ``holdout_pool_dir``, if any."""
    pool = Path(holdout_pool_dir)
    if not pool.exists():
        return None
    candidates = sorted(
        p
        for p in pool.rglob("*")
        if _HOLDOUT_MARKER in p.name
        and p.name.endswith((".fna", ".fna.gz", ".fa", ".fa.gz", ".fasta", ".fasta.gz"))
        and p.is_file()
    )
    return candidates[0] if candidates else None


def stage_chr22_holdout(
    extracted_dir: Union[str, Path],
    accession: str,
    holdout_pool_dir: Union[str, Path],
    force: bool = False,
) -> Optional[Path]:
    """Stage the chr22-excluded FASTA as ``extracted_dir/<accession>.fna.gz``.

    For the human accession, symlinks (or copies) the ``*_no_chr22`` FASTA over
    the accession's slot in ``extracted_files/`` so Phase 2 reads the chr22-free
    genome. No-op and returns ``None`` for any other accession or when no
    hold-out FASTA is available. Returns the staged path on success.
    """
    if not is_chr22_holdout_accession(accession):
        return None
    holdout = find_holdout_fasta(holdout_pool_dir)
    if holdout is None:
        logger.warning(
            "chr22 hold-out requested for %s but no *_no_chr22 FASTA found in %s",
            accession,
            holdout_pool_dir,
        )
        return None

    extracted_dir = Path(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".fna.gz" if str(holdout).endswith(".gz") else ".fna"
    target = extracted_dir / f"{accession}{suffix}"
    if target.exists() or target.is_symlink():
        if not force:
            return target
        target.unlink()
    # Also drop any full-genome slot for this accession so Phase 2 can't pick it.
    for other in extracted_dir.glob(f"{accession}.*"):
        if other != target:
            other.unlink()
    os.symlink(os.path.realpath(holdout), target)
    logger.info("chr22 hold-out: staged %s -> %s", holdout, target)
    return target
