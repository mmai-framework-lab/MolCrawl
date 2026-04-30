#!/usr/bin/env python3
"""
Download ChEMBL database (SQLite) and extract canonical SMILES strings.

Downloads ChEMBL 36 SQLite archive from the EBI FTP server, unpacks it, then
queries the `compound_structures` table for canonical SMILES with valid,
non-null values and writes them as one SMILES string per line into a text file
ready for the subsequent `prepare_chembl.py` tokenisation step.

Output layout under *output_dir*
─────────────────────────────────
  chembl_db/
    chembl_36_sqlite.tar.gz      ← downloaded archive (kept for resumability)
    chembl_36.db                 ← unpacked SQLite database
    smiles.txt                   ← one canonical SMILES per line
    download_complete.marker     ← written when this script finishes cleanly
"""

import logging
import os
import sqlite3
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ChEMBL 36 SQLite archive (EBI FTP, ~4.4 GB)
CHEMBL_VERSION = "36"
CHEMBL_ARCHIVE_URL = (
    f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/"
    f"releases/chembl_{CHEMBL_VERSION}/"
    f"chembl_{CHEMBL_VERSION}_sqlite.tar.gz"
)
CHEMBL_ARCHIVE_NAME = f"chembl_{CHEMBL_VERSION}_sqlite.tar.gz"
CHEMBL_DB_NAME = f"chembl_{CHEMBL_VERSION}.db"
SMILES_FILE_NAME = "smiles.txt"


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing a simple progress indicator."""

    def _report(block_count, block_size, total_size):
        if total_size > 0:
            downloaded = block_count * block_size
            pct = min(100, downloaded * 100 // total_size)
            if block_count % 200 == 0:
                logger.info(f"  Downloading … {pct}% ({downloaded // 1_000_000} MB / {total_size // 1_000_000} MB)")

    logger.info(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    logger.info("Download complete.")


def _extract_db(archive_path: Path, output_dir: Path) -> Path:
    """Extract the SQLite file from the tar.gz archive."""
    logger.info(f"Extracting {archive_path} …")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".db")]
        if not members:
            raise RuntimeError("No .db file found inside the archive.")
        db_member = members[0]
        db_member.name = os.path.basename(db_member.name)
        tar.extract(db_member, path=str(output_dir))
    db_path = output_dir / db_member.name
    # Rename to the canonical name if needed
    canonical = output_dir / CHEMBL_DB_NAME
    if db_path != canonical:
        db_path.rename(canonical)
        db_path = canonical
    logger.info(f"SQLite database extracted to {db_path}")
    return db_path


def _extract_smiles(db_path: Path, smiles_path: Path) -> int:
    """Query the SQLite DB and write canonical SMILES to *smiles_path*.

    Filters applied:
    - ``canonical_smiles`` IS NOT NULL and NOT empty
    - Excludes structures flagged as mixtures (containing '.') to keep the
      tokeniser vocabulary focused on single-compound SMILES.

    Returns the number of SMILES written.
    """
    logger.info(f"Querying {db_path} for canonical SMILES …")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    query = """
        SELECT canonical_smiles
        FROM compound_structures
        WHERE canonical_smiles IS NOT NULL
          AND canonical_smiles != ''
          AND canonical_smiles NOT LIKE '%.%'
        ORDER BY molregno
    """
    cursor.execute(query)

    count = 0
    with smiles_path.open("w", encoding="utf-8") as fh:
        for (smiles,) in cursor:
            fh.write(smiles.strip() + "\n")
            count += 1
            if count % 500_000 == 0:
                logger.info(f"  … {count:,} SMILES written")

    conn.close()
    logger.info(f"Wrote {count:,} SMILES to {smiles_path}")
    return count


def download_chembl(output_dir: str, force: bool = False) -> bool:
    """Download ChEMBL SQLite and extract canonical SMILES.

    Args:
        output_dir: Root directory for ChEMBL data (e.g. ``CHEMBL_SOURCE_DIR``).
        force: Re-run all steps even if marker file exists.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    marker = out / "download_complete.marker"
    if not force and marker.exists():
        logger.info("ChEMBL download already completed. Skipping. (use force=True to re-run)")
        return True

    archive_path = out / CHEMBL_ARCHIVE_NAME
    db_path = out / CHEMBL_DB_NAME
    smiles_path = out / SMILES_FILE_NAME

    try:
        # Step 1 – download archive
        if not archive_path.exists():
            _download_with_progress(CHEMBL_ARCHIVE_URL, archive_path)
        else:
            logger.info(f"Archive already present at {archive_path}, skipping download.")

        # Step 2 – extract SQLite DB
        if not db_path.exists():
            db_path = _extract_db(archive_path, out)
        else:
            logger.info(f"SQLite DB already present at {db_path}, skipping extraction.")

        # Step 3 – extract canonical SMILES
        _extract_smiles(db_path, smiles_path)

        marker.touch()
        logger.info(f"ChEMBL download pipeline complete. Marker written to {marker}")
        return True

    except Exception as exc:
        logger.error(f"ChEMBL download failed: {exc}", exc_info=True)
        return False


if __name__ == "__main__":
    import sys

    from molcrawl.core.paths import CHEMBL_SOURCE_DIR
    from molcrawl.core.base import setup_logging

    setup_logging(CHEMBL_SOURCE_DIR)
    force_flag = "--force" in sys.argv
    success = download_chembl(CHEMBL_SOURCE_DIR, force=force_flag)
    sys.exit(0 if success else 1)
