import concurrent.futures
import subprocess

import os.path as osp
import os
import dask.dataframe as dd
import shutil

import logging
import logging.config

logger = logging.getLogger(__name__)


def download_zinc_files(num_parallel : int = 1):
    shell_file: str = "src/compounds/dataset/organix13/zinc/zinc_complete/download_zinc.sh"
    directory = "src/compounds/dataset/organix13/zinc/zinc_complete"

    def execute_command(command, retrycount=5):
        logger.info(msg="Running: {}".format(command))
        n = 0
        while n < retrycount:
            n += 1
            try:
                err = subprocess.run(command, shell=True)
                if err.returncode == 0:
                    break
            except Exception as e:
                logger.error(msg="Error in command: {}".format(command))
                logger.error(msg="Error: {}".format(e))
                if n == retrycount:
                    raise e

    commands = []
    with open(shell_file, "r") as file:
        for line in file:
            line = line.strip()

            command_pieces = line.split(" ")
            command_pieces[2] = str(osp.join(directory, command_pieces[2]))
            command_pieces[-1] = str(osp.join(directory, command_pieces[-1]))

            if os.path.exists(command_pieces[-1]):
                if os.path.getsize(command_pieces[-1]) > 0:
                    continue

            line = " ".join(command_pieces)
            if line.startswith("mkdir") and "wget" in line:
                commands.append(line)

    for command in commands:
        execute_command(command)

    logger.info(msg="ZINC downloads completed")


def convert_zinc_to_parquet(save_path: str):
    directory = "src/compounds/dataset/organix13/zinc"
    zinc_path = os.path.join(directory, "zinc_complete")
    all_dirs = [
        osp.join(zinc_path, f)
        for f in os.listdir(zinc_path)
        if osp.isdir(osp.join(zinc_path, f))
    ]

    logger.info(msg="Number of dirs: {}".format(len(all_dirs)))
    all_dfs = []
    for d in all_dirs:
        logger.info(msg="Read: {}".format(d))
        try:
            df = dd.read_csv(
                f"{d}/*.txt",
                sep="\t",
                usecols=["smiles"],
            )
        except Exception as e:
            logger.error(msg=f"Likely a file is empty due to an error during download, try running the script again. Trace: {e}")
            continue
        all_dfs.append(df)

    concatenated_df = dd.concat(all_dfs)

    logger.info(msg="Writing")
    concatenated_df = concatenated_df.repartition(npartitions=1)
    concatenated_df = concatenated_df.reset_index(drop=True)
    concatenated_df.to_parquet(
        os.path.join(directory, "zinc_processed"),
    )
    logger.info(msg="Done Writing")
    shutil.copy(
        os.path.join(directory, "zinc_processed", "part.0.parquet"),
        os.path.join(save_path, "zinc_processed.parquet"),
    )
