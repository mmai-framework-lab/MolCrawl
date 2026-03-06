import concurrent.futures
import gzip
import json
import logging
import os
import re
import shutil
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

from molcrawl.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Per-species download timeout in seconds (default: 30 min)
DEFAULT_SPECIES_TIMEOUT = 30 * 60
# Maximum number of retries per species before giving up
DEFAULT_MAX_RETRIES = 2


def to_snake_case(string):
    return re.sub(r"(?<=[a-z])(?=[A-Z])|[^a-zA-Z]", "_", string).strip("_").lower()


def get_species(path_species):
    import ncbi_genome_download as ngd

    group_species_map = {}
    for group in ngd.SUPPORTED_TAXONOMIC_GROUPS:
        group_path = Path(path_species) / f"{group}.txt"

        if group_path.exists():
            with open(Path(path_species) / f"{group}.txt", "r") as file:
                species = file.readlines()
                group_species_map[group] = [sp.strip() for sp in species if sp.strip() != ""]
    return group_species_map


def _is_species_downloaded(sp_dir: Path) -> bool:
    """Check if a species directory already contains downloaded .fna.gz files."""
    if not sp_dir.exists():
        return False
    fna_files = list(sp_dir.glob("*genomic.fna.gz"))
    return len(fna_files) > 0


def _download_single_species_worker(queue: Queue, genera: str, group: str, sp_dir: str, num_worker: int):
    """Worker function executed in a separate process to download a single species.

    Communicates the result back to the parent via a multiprocessing Queue.
    Running in a separate process ensures that even if ncbi_genome_download
    hangs internally, the parent can terminate it via process.kill().
    """
    import ncbi_genome_download as ngd

    try:
        ret = ngd.download(
            genera=genera,
            groups=group,
            output=sp_dir,
            flat_output=True,
            progress_bar=True,
            file_formats="fasta",
            parallel=num_worker,
        )
        queue.put({"status": "ok", "return_code": ret})
    except Exception as e:
        queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def _download_single_species(
    genera: str,
    group: str,
    sp_dir: str,
    num_worker: int,
    timeout: int = DEFAULT_SPECIES_TIMEOUT,
) -> dict:
    """Download a single species with timeout protection.

    Spawns a child process for the actual download so that if ncbi_genome_download
    hangs (e.g. infinite loop, stuck HTTP connection), we can forcefully terminate it.

    Returns:
        dict with keys 'status' ('ok', 'timeout', 'error') and optional details.
    """
    queue: Queue = Queue()
    proc = Process(
        target=_download_single_species_worker,
        args=(queue, genera, group, sp_dir, num_worker),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Download exceeded the timeout – kill the child process
        logger.warning(f"Download of '{genera}' in '{group}' timed out after {timeout}s. Killing child process...")
        proc.kill()
        proc.join(timeout=10)
        return {"status": "timeout", "error": f"Timed out after {timeout}s"}

    if not queue.empty():
        return queue.get()
    else:
        exit_code = proc.exitcode
        return {"status": "error", "error": f"Worker exited with code {exit_code} and no result"}


def download_species_refseq(
    output_dir,
    path_species,
    num_worker,
    species_timeout=DEFAULT_SPECIES_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
):
    """Download RefSeq genome data for all species in the species list.

    Key safety features:
    - **Timeout**: Each species download runs in a separate process with a timeout.
      If ncbi_genome_download hangs, the child process is forcefully killed.
    - **Resume**: Species that already have downloaded files are automatically skipped.
    - **Retry**: Failed/timed-out species are retried up to ``max_retries`` times.
    - **Reporting**: A ``failed_species.json`` is written listing all species that
      could not be downloaded even after retries.
    """
    from rich.progress import Progress

    download_dir = Path(output_dir) / "download_dir"

    group_species_map = get_species(path_species)

    total_species = sum(len(v) for v in group_species_map.values())
    failed_species: list[dict] = []
    skipped_count = 0

    with Progress() as progress_bar:
        task = progress_bar.add_task("Processing ...", total=total_species)

        for group, species in group_species_map.items():
            download_group_dir = download_dir / group
            for sp in species:
                sp_name = sp.strip()
                sp_dir = download_group_dir / to_snake_case(sp_name)

                # ── Resume: skip already-downloaded species ──
                if _is_species_downloaded(sp_dir):
                    logger.info(f"Skipping '{sp_name}' in {group} (already downloaded)")
                    skipped_count += 1
                    progress_bar.update(task, advance=1)
                    continue

                progress_bar.update(task, description=f"Downloading refseq for species {sp_name} in {group}...")
                logger.info(f"Downloading refseq for species {sp_name} in {group}")

                # ── Retry loop ──
                success = False
                last_result = {}
                for attempt in range(1, max_retries + 1):
                    t0 = time.time()
                    result = _download_single_species(
                        genera=sp_name,
                        group=group,
                        sp_dir=str(sp_dir),
                        num_worker=num_worker,
                        timeout=species_timeout,
                    )
                    elapsed = time.time() - t0
                    last_result = result

                    if result["status"] == "ok":
                        logger.info(f"Successfully downloaded '{sp_name}' in {group} (attempt {attempt}, {elapsed:.1f}s)")
                        success = True
                        break
                    else:
                        logger.warning(
                            f"Download of '{sp_name}' in {group} failed "
                            f"(attempt {attempt}/{max_retries}, {elapsed:.1f}s): "
                            f"{result.get('error', 'unknown error')}"
                        )

                if not success:
                    error_info = {
                        "species": sp_name,
                        "group": group,
                        "error": last_result.get("error", "unknown"),
                        "status": last_result.get("status", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    }
                    failed_species.append(error_info)
                    logger.error(
                        f"Giving up on '{sp_name}' in {group} after {max_retries} attempts. Continuing with next species..."
                    )

                progress_bar.update(task, advance=1)

    # ── Summary report ──
    total_attempted = total_species - skipped_count
    total_failed = len(failed_species)
    total_succeeded = total_attempted - total_failed

    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info(f"  Total species     : {total_species}")
    logger.info(f"  Skipped (cached)  : {skipped_count}")
    logger.info(f"  Attempted         : {total_attempted}")
    logger.info(f"  Succeeded         : {total_succeeded}")
    logger.info(f"  Failed            : {total_failed}")
    logger.info("=" * 60)

    if failed_species:
        failed_path = Path(output_dir) / "failed_species.json"
        with open(failed_path, "w") as f:
            json.dump(failed_species, f, indent=2, ensure_ascii=False)
        logger.warning(f"Failed species list saved to {failed_path}")
        for fs in failed_species:
            logger.warning(f"  - {fs['species']} ({fs['group']}): {fs['error']}")


def extract_file(
    archive_path: str,
    try_count: int = 0,
    max_try: int = 3,
):
    # pass .sdf.gz to .sdf
    sdf_file_path = Path(archive_path.replace("download_dir", "extracted_files")).with_suffix("")
    if sdf_file_path.exists():
        logger.info(f"Skipping extraction of {sdf_file_path}, already exist")
        return
    sdf_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path} to {sdf_file_path}")
    if os.path.exists(archive_path):
        try:
            # Decompress the .gz file and save the result as .sdf
            with gzip.open(archive_path, "rb") as f_in:
                with open(sdf_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            os.remove(sdf_file_path)
            msg = str(e) + "\n" + "".join(traceback.format_exception(None, e, e.__traceback__))
            logging.error(f"[Try: {try_count + 1}]  File {archive_path} created an error : \n{msg}")
            if try_count < max_try:
                return extract_file(archive_path, try_count + 1)
    else:
        logger.error(f"File {archive_path} does not exist skipping")


def extract_refseq(output_dir, num_worker):
    from rich.progress import track

    download_dir = Path(output_dir) / "download_dir"

    archive_paths = [str(p) for p in download_dir.rglob("*genomic.fna.gz")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        list(
            track(
                executor.map(extract_file, archive_paths),
                total=len(archive_paths),
                description="Extracting...",
            )
        )


def download_refseq(
    output_dir, path_species, num_worker, species_timeout=DEFAULT_SPECIES_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES
):
    download_species_refseq(output_dir, path_species, num_worker, species_timeout=species_timeout, max_retries=max_retries)
    extract_refseq(output_dir, num_worker)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    download_refseq(
        cfg.output_dir,
        cfg.path_species,
        cfg.num_worker,
        species_timeout=getattr(cfg, "species_timeout", DEFAULT_SPECIES_TIMEOUT),
        max_retries=getattr(cfg, "max_retries", DEFAULT_MAX_RETRIES),
    )
