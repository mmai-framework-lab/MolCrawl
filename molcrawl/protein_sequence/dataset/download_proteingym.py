"""
Download ProteinGym v1.3 DMS substitution data for fine-tuning.

The downloaded CSV files are saved to *save_path* so that
prepare_proteingym.py can read them directly.

Usage:
    from molcrawl.protein_sequence.dataset.download_proteingym import download_proteingym
    download_proteingym("/path/to/proteingym_v1.3")
"""

import logging
import zipfile
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Only the DMS substitution split is used for fine-tuning language models.
# (indels are a much smaller set and rarely used for LM pre-training baselines)
_SUBSTITUTIONS_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/" "DMS_ProteinGym_substitutions.zip"
_REFERENCE_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/" "DMS_substitutions.csv"


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, skipping if already present."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return
    logger.info("Downloading %s → %s", url, dest)

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0 and block_num % 500 == 0:
            pct = min(100.0, block_num * block_size / total_size * 100)
            logger.info("  %.1f%%", pct)

    urlretrieve(url, str(dest), reporthook=_reporthook)
    logger.info("Download complete: %s", dest)


def download_proteingym(save_path: Union[str, Path]) -> Path:
    """
    Download ProteinGym v1.3 DMS substitution CSV files into *save_path*.

    After this function returns, *save_path* will contain:
        DMS_ProteinGym_substitutions/   ← one CSV per assay
        DMS_substitutions.csv           ← reference metadata (one row per assay)

    Args:
        save_path: Destination directory (created if missing).

    Returns:
        Path to the directory containing individual assay CSV files.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- substitutions zip --------------------------------------------------
    zip_path = save_path / "DMS_ProteinGym_substitutions.zip"
    _download_with_progress(_SUBSTITUTIONS_URL, zip_path)

    csv_dir = save_path / "DMS_ProteinGym_substitutions"
    if not csv_dir.exists():
        logger.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(save_path)
        logger.info("Extracted to %s", csv_dir)
    else:
        logger.info("Already extracted: %s", csv_dir)

    # --- reference CSV ------------------------------------------------------
    ref_path = save_path / "DMS_substitutions.csv"
    _download_with_progress(_REFERENCE_URL, ref_path)

    return csv_dir
