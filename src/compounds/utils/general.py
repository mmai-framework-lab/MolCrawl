from compounds.dataset.organix13.zinc.download_and_convert_to_parquet import (
    download_zinc_files,
    convert_zinc_to_parquet,
)
from compounds.dataset.organix13.combine_all import combine_all
from compounds.dataset.organix13.opv.prepare_opv import OPV
from compounds.dataset.organix13.download import download_datasets_from_repo


def download_datasets(raw_data_dir: str, output_dir: str):
    download_zinc_files()
    convert_zinc_to_parquet(raw_data_dir)
    OPV(raw_data_dir)
    download_datasets_from_repo(raw_data_dir)
    combine_all(raw_data_dir, output_dir)
