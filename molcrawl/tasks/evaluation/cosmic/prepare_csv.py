"""Build the cosmic evaluator's input CSV from the v100+ CMC TSV.

The cosmic evaluator (``molcrawl.tasks.evaluation.cosmic``) expects a
CSV with columns::

    reference_sequence, variant_sequence, FATHMM_PREDICTION

That schema dates from the COSMIC v95 era when ``CosmicMutantExport.tsv.gz``
shipped FATHMM predictions inline.  Sanger has since (a) retired bulk Basic-
Auth downloads — see ``workflows/data/eval-data-cosmic.sh`` for the new
NextAuth + presigned-URL flow — and (b) folded FATHMM scoring into the
Cancer Mutation Census (CMC) ``alldata-cmc`` product, where the equivalent
label is ``MUTATION_SIGNIFICANCE_TIER`` (tier 1/2 = high-confidence driver,
tier 3 = passenger / low-confidence).

This module reads
``$LSD/eval/cosmic/alldata-cmc/CancerMutationCensus_AllData_v100_GRCh37.tsv.gz``,
class-balances driver vs passenger rows, fetches a ``±FLANK`` bp window
around each variant from the Ensembl GRCh37 REST API, applies the variant
allele, and writes the legacy schema CSV the evaluator already understands.

Why Ensembl REST instead of a local FASTA: the GRCh37 reference is ~3 GB
to download and pyfaidx-index, but Ensembl serves ``/sequence/region/...``
free with a 55,000 req/hr limit, which is plenty for the 100s-of-rows
subsample used by the evaluator's 足固め mode.  If the user wants a much
larger run, a local FASTA path is straightforward to bolt on.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)

ENSEMBL_GRCh37 = "https://grch37.rest.ensembl.org"

LEGACY_FATHMM_LABELS = {
    "1": "DAMAGING",  # high-confidence driver
    "2": "DAMAGING",  # medium-confidence driver
    "3": "NEUTRAL",   # passenger
}

POSITION_RE = re.compile(r"^(\w+):(\d+)(?:-(\d+))?$")


def _parse_genome_position(raw: str) -> Optional[Tuple[str, int, int]]:
    """Parse strings like ``7:140453136-140453136`` or ``7:140453136``."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    m = POSITION_RE.match(s)
    if not m:
        return None
    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return chrom, start, end


def _fetch_flank(chrom: str, pos: int, flank: int, session: "requests.Session") -> Optional[str]:
    """Fetch a ``[pos-flank, pos+flank]`` window from Ensembl GRCh37.

    Returns the raw upper-case sequence string (length ``2*flank + 1``) or
    ``None`` on failure.  The middle base (index ``flank``) is the reference
    allele at ``pos``; callers substitute it to build the variant copy.
    """
    start = pos - flank
    end = pos + flank
    url = f"{ENSEMBL_GRCh37}/sequence/region/human/{chrom}:{start}..{end}"
    try:
        resp = session.get(url, headers={"Accept": "text/plain"}, timeout=20)
    except Exception as e:
        logger.warning("ensembl request failed for %s:%d: %s", chrom, pos, e)
        return None
    if resp.status_code == 429:
        retry = float(resp.headers.get("Retry-After", "1"))
        logger.info("ensembl rate-limited, sleeping %.1fs", retry)
        time.sleep(retry + 0.1)
        return _fetch_flank(chrom, pos, flank, session)
    if resp.status_code != 200 or not resp.text:
        logger.warning("ensembl %d for %s:%d", resp.status_code, chrom, pos)
        return None
    return resp.text.strip().upper()


def _stratified_subsample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    parts: List[pd.DataFrame] = []
    for _tier_label, sub in df.groupby("MUTATION_SIGNIFICANCE_TIER"):
        if len(sub) <= per_class:
            parts.append(sub)
        else:
            keep_idx = rng.sample(range(len(sub)), per_class)
            parts.append(sub.iloc[keep_idx])
    return pd.concat(parts, ignore_index=True)


def build_csv(
    cmc_tsv: Path,
    out_csv: Path,
    flank: int = 256,
    per_class: int = 100,
    seed: int = 42,
    sleep_seconds: float = 0.07,
) -> int:
    """Read CMC TSV, build the legacy-schema CSV, return number of rows written."""
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests is required for ensembl REST lookups") from e

    logger.info("loading %s ...", cmc_tsv)
    cols = [
        "GENOMIC_WT_ALLELE_SEQ",
        "GENOMIC_MUT_ALLELE_SEQ",
        "Mutation genome position GRCh37",
        "MUTATION_SIGNIFICANCE_TIER",
        "GENE_NAME",
    ]
    # CMC has ~5M rows so we read in chunks and only keep tier 1/2/3 rows.
    keep_chunks: List[pd.DataFrame] = []
    with gzip.open(cmc_tsv, "rt") as fh:
        reader = pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, chunksize=200_000)
        for chunk in reader:
            mask = chunk["MUTATION_SIGNIFICANCE_TIER"].isin(["1", "2", "3"])
            kept = chunk.loc[mask].copy()
            if not kept.empty:
                keep_chunks.append(kept)
    df = pd.concat(keep_chunks, ignore_index=True)
    logger.info(
        "kept %d tiered rows (1/2/3); tier dist: %s",
        len(df),
        df["MUTATION_SIGNIFICANCE_TIER"].value_counts().to_dict(),
    )

    # Restrict to single-nucleotide variants — flank substitution only makes
    # sense for SNVs, and CMC reports the ALT base in GENOMIC_MUT_ALLELE_SEQ.
    snv_mask = (
        df["GENOMIC_WT_ALLELE_SEQ"].str.len() == 1
    ) & (df["GENOMIC_MUT_ALLELE_SEQ"].str.len() == 1)
    df = df.loc[snv_mask].reset_index(drop=True)
    logger.info("%d SNV rows after WT/MUT length filter", len(df))

    df = _stratified_subsample(df, per_class=per_class, seed=seed)
    logger.info(
        "subsampled to %d rows; tier dist: %s",
        len(df),
        df["MUTATION_SIGNIFICANCE_TIER"].value_counts().to_dict(),
    )

    session = requests.Session()
    out_rows: List[dict] = []
    for idx, row in df.iterrows():
        parsed = _parse_genome_position(str(row["Mutation genome position GRCh37"]))
        if parsed is None:
            continue
        chrom, start, end = parsed
        if start != end:
            continue  # skip multi-base events
        wt = (row["GENOMIC_WT_ALLELE_SEQ"] or "").strip().upper()
        mut = (row["GENOMIC_MUT_ALLELE_SEQ"] or "").strip().upper()
        if len(wt) != 1 or len(mut) != 1 or wt == mut:
            continue
        seq = _fetch_flank(chrom, start, flank, session)
        if seq is None or len(seq) != 2 * flank + 1:
            continue
        # Build reference and variant sequences from the same window so the
        # only difference is the variant base at the middle position.
        center = flank
        if seq[center] != wt:
            # Strand mismatch / annotation drift — skip rather than emit a
            # synthetic ref that doesn't match the genome.
            continue
        ref_seq = seq
        var_seq = seq[:center] + mut + seq[center + 1:]
        tier = str(row["MUTATION_SIGNIFICANCE_TIER"])
        out_rows.append({
            "reference_sequence": ref_seq,
            "variant_sequence": var_seq,
            "FATHMM_PREDICTION": LEGACY_FATHMM_LABELS[tier],
            "GENE_NAME": row["GENE_NAME"],
            "MUTATION_SIGNIFICANCE_TIER": tier,
            "chrom": chrom,
            "pos": start,
            "ref": wt,
            "alt": mut,
        })
        time.sleep(sleep_seconds)
        if (idx + 1) % 50 == 0:
            logger.info("ensembl flank: %d/%d done (kept %d)", idx + 1, len(df), len(out_rows))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    logger.info("wrote %d rows to %s", len(out_rows), out_csv)
    return len(out_rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmc-tsv", required=True, type=Path,
                        help="Path to CancerMutationCensus_AllData_v100_GRCh37.tsv.gz")
    parser.add_argument("--out-csv", required=True, type=Path,
                        help="Output CSV consumed by the cosmic evaluator")
    parser.add_argument("--flank", type=int, default=256)
    parser.add_argument("--per-class", type=int, default=100,
                        help="Rows kept per MUTATION_SIGNIFICANCE_TIER")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=float, default=0.07,
                        help="Throttle between Ensembl REST calls")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rows = build_csv(
        cmc_tsv=args.cmc_tsv,
        out_csv=args.out_csv,
        flank=args.flank,
        per_class=args.per_class,
        seed=args.seed,
        sleep_seconds=args.sleep_seconds,
    )
    return 0 if rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
