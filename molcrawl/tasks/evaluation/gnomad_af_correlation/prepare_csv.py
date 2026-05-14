"""Prepare a ClinVar-shaped CSV from a gnomAD chromosome-scoped VCF.

gnomAD ships allele frequencies inside VCF INFO fields but the
evaluator expects a flat CSV with (reference_sequence, variant_sequence,
allele_frequency). This module streams through the VCF, keeps only
biallelic SNVs with a numerical AF, pulls a flank-length context around
each site from the reference FASTA, and writes one row per variant.

The output layout matches :mod:`molcrawl.tasks.evaluation.clinvar`
(``chrom``, ``pos``, ``ref``, ``alt``, ``reference_sequence``,
``variant_sequence``, ``allele_frequency``) so the gnomAD and ClinVar
pipelines share sampling / metric / prediction-log helpers downstream.

Typical invocation::

    python -m molcrawl.tasks.evaluation.gnomad_af_correlation.prepare_csv \\
        --vcf   $LSD/eval/gnomad_af_correlation/gnomad.genomes.v4.1.sites.chr22.vcf.bgz \\
        --fasta $PROJECT/dataset/GCA_000001405.28_GRCh38.p13_genomic.fna \\
        --output $LSD/eval/gnomad_af_correlation/gnomad_chr22.csv \\
        --flank 64 --max-variants 50000 --seed 42

``--max-variants`` activates reservoir sampling so the script can run in
bounded memory against whole-chromosome VCFs containing several million
records. The reservoir is a **random sample of biallelic SNVs with a
finite AF**; downstream evaluators may still subsample further.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_HEADER_RE = re.compile(
    r"^>(\S+)\s+Homo sapiens chromosome ([0-9XYMT]+), "
    r"GRCh38 reference primary assembly",
    re.IGNORECASE,
)


def build_fasta_chrom_map(fasta_path: Path) -> Dict[str, str]:
    """Return ``{chrom_label -> fasta_contig_id}`` for the primary chromosomes.

    gnomAD uses ``chr1``..``chr22``, ``chrX``, ``chrY``. The GRCh38
    primary-assembly FASTA used here stores them under GenBank
    accessions (``CM000663.2`` .. ``CM000686.2``), with the descriptor
    ``Homo sapiens chromosome N, GRCh38 reference primary assembly``.
    We scan FASTA headers once and emit a bidirectional mapping keyed by
    both ``chrN`` and ``N`` so callers with either naming style work.
    """
    mapping: Dict[str, str] = {}
    with fasta_path.open("r") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            match = _HEADER_RE.match(line)
            if not match:
                continue
            contig, chrom = match.group(1), match.group(2)
            mapping[f"chr{chrom}"] = contig
            mapping[chrom] = contig
    return mapping


def _flank_from_fasta(
    fasta, contig: str, pos_1based: int, flank: int, expected_ref: str
) -> Optional[str]:
    """Return a ``2*flank+1`` substring centred at ``pos_1based``."""
    start = pos_1based - 1 - flank
    end = pos_1based - 1 + flank + 1
    if start < 0 or end > len(fasta[contig]):
        return None
    seq = str(fasta[contig][start:end]).upper()
    if len(seq) != 2 * flank + 1:
        return None
    # Sanity: the center base in the FASTA must match the VCF REF allele;
    # if it does not (liftover mismatch / softmask drift) we drop the row.
    if seq[flank] != expected_ref.upper():
        return None
    if "N" in seq:
        return None
    return seq


def _substitute_center(ref_seq: str, flank: int, alt: str) -> str:
    return ref_seq[:flank] + alt.upper() + ref_seq[flank + 1 :]


def extract_snv_records(
    vcf_path: Path,
    fasta_path: Path,
    flank: int,
    max_variants: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    """Stream the VCF once and return a list of CSV-ready rows."""
    import pyfaidx
    import pysam

    chrom_map = build_fasta_chrom_map(fasta_path)
    if not chrom_map:
        raise RuntimeError(
            f"Could not derive a chromosome map from {fasta_path}. "
            "The FASTA may use a non-GRCh38 header format."
        )
    logger.info("Mapped %d chromosome labels from FASTA headers", len(chrom_map))

    fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True)
    vcf = pysam.VariantFile(str(vcf_path))

    rng = random.Random(seed)
    reservoir: List[Dict[str, object]] = []
    total_seen = 0
    total_kept_candidates = 0
    dropped_not_snv = 0
    dropped_no_af = 0
    dropped_flank = 0

    for rec in vcf:
        total_seen += 1
        if total_seen % 500_000 == 0:
            logger.info(
                "Scanned %d records (kept candidates=%d, reservoir=%d)",
                total_seen,
                total_kept_candidates,
                len(reservoir),
            )

        if not rec.alts:
            continue
        # Biallelic SNVs only.
        if len(rec.alts) != 1:
            continue
        ref = rec.ref or ""
        alt = rec.alts[0] or ""
        if len(ref) != 1 or len(alt) != 1 or ref.upper() == alt.upper():
            dropped_not_snv += 1
            continue
        if ref.upper() not in "ACGT" or alt.upper() not in "ACGT":
            dropped_not_snv += 1
            continue

        af_field = rec.info.get("AF")
        if af_field is None:
            dropped_no_af += 1
            continue
        if isinstance(af_field, (tuple, list)):
            af_value = af_field[0] if af_field else None
        else:
            af_value = af_field
        if af_value is None:
            dropped_no_af += 1
            continue
        try:
            af_float = float(af_value)
        except (TypeError, ValueError):
            dropped_no_af += 1
            continue
        if not (0.0 <= af_float <= 1.0):
            dropped_no_af += 1
            continue

        chrom_label = str(rec.chrom)
        contig = chrom_map.get(chrom_label) or chrom_map.get(
            chrom_label.replace("chr", "")
        )
        if contig is None:
            continue

        ref_seq = _flank_from_fasta(fasta, contig, rec.pos, flank, ref)
        if ref_seq is None:
            dropped_flank += 1
            continue
        var_seq = _substitute_center(ref_seq, flank, alt)

        row = {
            "chrom": chrom_label,
            "pos": int(rec.pos),
            "ref": ref.upper(),
            "alt": alt.upper(),
            "reference_sequence": ref_seq,
            "variant_sequence": var_seq,
            "allele_frequency": af_float,
        }
        total_kept_candidates += 1

        if max_variants is None:
            reservoir.append(row)
        else:
            # Classic reservoir sampling (Vitter's Algorithm R).
            if len(reservoir) < max_variants:
                reservoir.append(row)
            else:
                j = rng.randint(0, total_kept_candidates - 1)
                if j < max_variants:
                    reservoir[j] = row

    logger.info(
        "VCF scan complete: total=%d, candidates=%d, kept=%d "
        "(dropped: not_snv=%d, no_af=%d, flank_mismatch=%d)",
        total_seen,
        total_kept_candidates,
        len(reservoir),
        dropped_not_snv,
        dropped_no_af,
        dropped_flank,
    )
    return reservoir


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "reference_sequence",
        "variant_sequence",
        "allele_frequency",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %d rows -> %s", len(rows), output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a gnomAD VCF to a ClinVar-shaped CSV "
        "(reference_sequence, variant_sequence, allele_frequency)."
    )
    parser.add_argument("--vcf", required=True, type=Path, help="Input .vcf.bgz")
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Reference genome FASTA (GRCh38 primary assembly, CM-accession headers)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV")
    parser.add_argument(
        "--flank",
        type=int,
        default=64,
        help=(
            "Half-window around the variant site; output sequences are "
            "(2*flank + 1) bp long (default 64 → 129 bp like ClinVar)."
        ),
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help=(
            "Activate reservoir sampling to keep at most N SNV rows. "
            "Set None (omit) to keep all biallelic SNVs with AF."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Reservoir seed")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    rows = extract_snv_records(
        vcf_path=args.vcf,
        fasta_path=args.fasta,
        flank=args.flank,
        max_variants=args.max_variants,
        seed=args.seed,
    )
    write_csv(rows, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
