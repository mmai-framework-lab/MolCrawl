import logging
import os

import requests

logger = logging.getLogger(__name__)


def download(output_dir: str, url: str, name: str):
    dataset_filename = os.path.join(output_dir, name)

    response = requests.get(url)
    downloaded_data = response.content

    with open(dataset_filename, "wb") as f:
        f.write(downloaded_data)

    logger.info(msg=f"Combined dataframe saved to '{dataset_filename}' as Parquet file.")
