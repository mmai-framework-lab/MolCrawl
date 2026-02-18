from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from rna.utils.config import RnaConfig


def h5ad_to_loom(output_dir):
    import loompy as lp
    import scanpy as sc

    h5ad_dir = Path(output_dir) / "download_dir"
    loom_outdir = Path(output_dir) / "loom_dir"

    loom_outdir.mkdir(exist_ok=True, parents=True)
    paths = list(h5ad_dir.iterdir())

    for path in tqdm(paths):
        loom_path = loom_outdir / path.with_suffix(".loom").name
        try:
            anndata = sc.read_h5ad(path)
            anndata.write_loom(loom_path)

            with lp.connect(loom_path, mode="r+") as ds:
                # Add "ensembl_id" row attribute
                # if "ensembl_id" not in ds.ra:
                ds.ra["ensembl_id"] = ds.ra["feature_id"]

                # Add "n_counts" column attribute
                ds.ca["n_counts"] = ds.ca["raw_sum"]

                del ds.ca["obs_names"]
        except Exception as e:
            print(f"Error with {path}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    h5ad_to_loom(cfg.output_dir)
