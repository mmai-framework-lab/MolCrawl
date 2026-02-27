import os
from argparse import ArgumentParser
from pathlib import Path
import logging
import shutil
import gzip
import concurrent.futures
import traceback
import re

from molcrawl.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)


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


def download_species_refseq(output_dir, path_species, num_worker):
    """To check a species name and refseq presence: https://www.ncbi.nlm.nih.gov/datasets/genome/"""
    import ncbi_genome_download as ngd
    from rich.progress import Progress

    download_dir = Path(output_dir) / "download_dir"

    group_species_map = get_species(path_species)

    total_species = sum([len(v) for v in group_species_map.values()])
    with Progress() as progress_bar:
        task = progress_bar.add_task("Processing ...", total=total_species)

        for group, species in group_species_map.items():
            download_group_dir = download_dir / group
            for sp in species:
                progress_bar.update(task, description=f"Downloading refseq for species {sp} in {group}...")
                logging.info(f"Downloading refseq for species {sp} in {group}")
                sp_dir = download_group_dir / to_snake_case(sp.strip())
                ngd.download(
                    genera=sp.strip(),
                    groups=group,
                    output=str(sp_dir),
                    flat_output=True,
                    progress_bar=True,
                    file_formats="fasta",
                    parallel=num_worker,
                )
                progress_bar.update(task, advance=1)


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


def download_refseq(output_dir, path_species, num_worker):
    download_species_refseq(output_dir, path_species, num_worker)
    extract_refseq(output_dir, num_worker)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    download_refseq(cfg.output_dir, cfg.path_species, cfg.num_worker)
