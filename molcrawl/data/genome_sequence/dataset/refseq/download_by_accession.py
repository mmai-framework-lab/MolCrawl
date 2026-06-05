"""Download exact RefSeq/GenBank assemblies listed in a subset CSV.

Unlike :mod:`download_refseq` (which resolves *species names* to "the latest"
assembly via ``ncbi_genome_download``), this module downloads the **exact**
assembly identified by ``assembly_accession`` / ``ftp_path`` in a subset CSV.

This matters when the species list is derived from an external corpus (e.g. the
OpenGenome2 / Evo 2 species set): we must fetch the same assembly version that
the source used, not whatever NCBI currently considers "latest" for that
species name.

Input CSV columns used:
  - ``assembly_accession`` : e.g. ``GCA_029851265.1`` (used for the output name)
  - ``ftp_path``           : NCBI assembly directory URL, e.g.
        ``https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/029/851/265/GCA_029851265.1_ASM2985126v1/``
  - ``species_name``       : optional, for logging only

The genomic FASTA URL is derived from ``ftp_path`` following the standard NCBI
layout ``<ftp_path>/<asm_dir>_genomic.fna.gz``.

Output:
  ``<base_dir>/extracted_files/<assembly_accession>.fna.gz``

Re-runs skip assemblies that already downloaded cleanly, so the same command
can be used to fill gaps after partial failures.
"""

import argparse
import csv
import gzip
import hashlib
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default per-file download timeout in seconds (socket-level).
DEFAULT_DOWNLOAD_TIMEOUT = 10 * 60
# Maximum retries per assembly before giving up.
DEFAULT_MAX_RETRIES = 3
# Streaming read chunk size.
_CHUNK = 1 << 20  # 1 MiB
_GZIP_MAGIC = b"\x1f\x8b"


def build_fasta_url(ftp_path: str) -> str:
    """Construct the genomic FASTA URL from an NCBI assembly ``ftp_path``.

    ``ftp_path`` points at the assembly directory; the FASTA file inside is
    named ``<asm_dir>_genomic.fna.gz`` where ``asm_dir`` is the final path
    component (``GCA_029851265.1_ASM2985126v1``). ``ftp://`` is upgraded to
    ``https://`` because the FTP endpoint is unreliable for scripted access.
    """
    ftp = ftp_path.strip().rstrip("/")
    if ftp.startswith("ftp://"):
        ftp = "https://" + ftp[len("ftp://"):]
    asm_dir = ftp.rsplit("/", 1)[-1]
    return f"{ftp}/{asm_dir}_genomic.fna.gz"


def read_subset_csv(csv_path: str) -> list[dict]:
    """Read a subset CSV, returning rows with assembly_accession + ftp_path."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Subset CSV is empty: {csv_path}")
    for col in ("assembly_accession", "ftp_path"):
        if col not in rows[0]:
            raise ValueError(
                f"Subset CSV {csv_path} is missing required column '{col}'. "
                f"Available columns: {list(rows[0].keys())}"
            )
    return rows


def _is_valid_gzip(path: Path, deep: bool = False) -> bool:
    """Validity check for an existing download.

    With ``deep=False`` (default) this is a cheap structural check: the file
    exists, is non-empty, and starts with the gzip magic bytes. This is enough
    to resume after a normal run, because downloads are written atomically
    (``.part`` → rename) only after a Content-Length match, so truncated files
    never become final under normal operation.

    With ``deep=True`` the whole stream is decompressed to detect truncation or
    corruption (e.g. disk faults, or servers that omitted Content-Length). This
    is O(file size) and is opt-in via ``--recheck-existing``.
    """
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "rb") as fh:
            if fh.read(2) != _GZIP_MAGIC:
                return False
        if deep:
            with gzip.open(path, "rb") as gz:
                while gz.read(_CHUNK):
                    pass
        return True
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _fetch_expected_md5(ftp_path: str, fasta_filename: str, timeout: int) -> Optional[str]:
    """Fetch the md5 for ``fasta_filename`` from the assembly's md5checksums.txt.

    Returns the hex digest, or ``None`` if it cannot be determined.
    """
    ftp = ftp_path.strip().rstrip("/")
    if ftp.startswith("ftp://"):
        ftp = "https://" + ftp[len("ftp://"):]
    url = f"{ftp}/md5checksums.txt"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        logger.warning(f"Could not fetch md5checksums.txt from {url}: {e}")
        return None
    # Lines look like: "<md5>  ./<filename>"
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("./").endswith(fasta_filename):
            return parts[0]
    return None


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    row: dict,
    extracted_dir: Path,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verify_md5: bool = False,
    recheck_existing: bool = False,
) -> dict:
    """Download a single assembly's genomic FASTA.

    Returns a dict with ``status`` in {"ok", "skipped", "error"} plus details.
    Writes to a ``.part`` file first and atomically renames on success.
    """
    accession = row["assembly_accession"].strip()
    ftp_path = row["ftp_path"].strip()
    species = row.get("species_name", "").strip()
    url = build_fasta_url(ftp_path)
    dest = extracted_dir / f"{accession}.fna.gz"
    tmp = extracted_dir / f"{accession}.fna.gz.part"

    if _is_valid_gzip(dest, deep=recheck_existing):
        return {"status": "skipped", "accession": accession}

    last_err = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            n_bytes = 0
            expected_len = None
            req = urllib.request.Request(url, headers={"User-Agent": "molcrawl-genome-fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                expected_len = int(cl) if cl is not None else None
                with open(tmp, "wb") as out:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        n_bytes += len(chunk)

            # Truncation check via Content-Length.
            if expected_len is not None and n_bytes != expected_len:
                raise IOError(f"size mismatch: got {n_bytes}, expected {expected_len}")
            # Structural check.
            with open(tmp, "rb") as fh:
                if fh.read(2) != _GZIP_MAGIC:
                    raise IOError("downloaded file is not gzip")

            if verify_md5:
                fasta_filename = url.rsplit("/", 1)[-1]
                expected_md5 = _fetch_expected_md5(ftp_path, fasta_filename, timeout)
                if expected_md5 is not None:
                    actual = _md5_of_file(tmp)
                    if actual != expected_md5:
                        raise IOError(f"md5 mismatch: got {actual}, expected {expected_md5}")

            os.replace(tmp, dest)
            return {"status": "ok", "accession": accession, "bytes": n_bytes}

        except (urllib.error.URLError, OSError) as e:
            last_err = str(e)
            logger.warning(
                f"[{accession}] download attempt {attempt}/{max_retries} failed: {last_err}"
            )
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(min(30, 2 ** attempt))

    return {
        "status": "error",
        "accession": accession,
        "species_name": species,
        "url": url,
        "error": last_err,
    }


def download_subset_from_csv(
    csv_path: str,
    base_dir: str,
    num_worker: int = 8,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verify_md5: bool = False,
    recheck_existing: bool = False,
    limit: Optional[int] = None,
) -> bool:
    """Download every assembly in ``csv_path`` into ``base_dir/extracted_files``.

    Returns ``True`` only if all (non-skipped) assemblies downloaded cleanly.
    Failures are recorded in ``base_dir/download_failures.csv`` and a
    ``download_complete.marker`` is written only when there are zero failures,
    so a re-run picks up exactly the gaps.
    """
    rows = read_subset_csv(csv_path)
    if limit is not None:
        rows = rows[:limit]

    extracted_dir = Path(base_dir) / "extracted_files"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading {len(rows)} assemblies from {Path(csv_path).name} "
        f"→ {extracted_dir} (num_worker={num_worker}, verify_md5={verify_md5})"
    )

    n_ok = n_skip = 0
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=num_worker) as ex:
        futures = {
            ex.submit(
                download_one, row, extracted_dir, timeout, max_retries, verify_md5, recheck_existing
            ): row
            for row in rows
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["status"] == "ok":
                n_ok += 1
            elif res["status"] == "skipped":
                n_skip += 1
            else:
                failures.append(res)
            if done % 50 == 0 or done == total:
                logger.info(f"  progress {done}/{total} (ok={n_ok} skip={n_skip} fail={len(failures)})")

    logger.info("=" * 60)
    logger.info("Accession download summary")
    logger.info(f"  CSV          : {csv_path}")
    logger.info(f"  Total        : {len(rows)}")
    logger.info(f"  Downloaded   : {n_ok}")
    logger.info(f"  Skipped(have): {n_skip}")
    logger.info(f"  Failed       : {len(failures)}")
    logger.info("=" * 60)

    if failures:
        fail_path = Path(base_dir) / "download_failures.csv"
        with open(fail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["accession", "species_name", "url", "error"])
            w.writeheader()
            for fr in failures:
                w.writerow({k: fr.get(k, "") for k in w.fieldnames})
        logger.warning(f"{len(failures)} failures written to {fail_path}; re-run to retry gaps.")
        return False

    (Path(base_dir) / "download_complete.marker").touch()
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download exact NCBI assemblies listed in a subset CSV (by accession/ftp_path).",
    )
    p.add_argument("--csv", required=True, help="Subset CSV (needs assembly_accession, ftp_path columns).")
    p.add_argument("--output", "-o", required=True, help="Base output dir; FASTA goes to <output>/extracted_files/.")
    p.add_argument("--num-worker", type=int, default=8)
    p.add_argument("--timeout", type=int, default=DEFAULT_DOWNLOAD_TIMEOUT, help="Per-file timeout (s).")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--verify-md5", action="store_true", help="Verify each file against NCBI md5checksums.txt.")
    p.add_argument(
        "--recheck-existing",
        action="store_true",
        help="Deep-verify already-present files (full gzip decompress) instead of a magic-byte check.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only download the first N rows (smoke test).")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    ok = download_subset_from_csv(
        csv_path=args.csv,
        base_dir=args.output,
        num_worker=args.num_worker,
        timeout=args.timeout,
        max_retries=args.max_retries,
        verify_md5=args.verify_md5,
        recheck_existing=args.recheck_existing,
        limit=args.limit,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
