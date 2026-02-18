from pathlib import Path
from argparse import ArgumentParser

from rich.progress import Progress

from rna.utils.config import RnaConfig


def get_tissue_list(census):
    summary_table = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
    summary_table = summary_table.query("organism == 'Homo sapiens' & category == 'tissue_general'")

    return summary_table["label"].unique().tolist()


def save_tissue_var(target_tissue, census, dir_path: Path) -> None:
    tissue_var = census["census_data"]["homo_sapiens"].ms["RNA"].var.read().concat().to_pandas()
    filename = dir_path / f"{target_tissue}.var.tsv"
    tissue_var.to_csv(filename, sep="\t")


def save_tissue_obs(target_tissue, census, dir_path: Path) -> None:
    tissue_obs = (
        census["census_data"]["homo_sapiens"]
        .obs.read(value_filter="tissue_general == '" + target_tissue + "' and is_primary_data == True")
        .concat()
        .to_pandas()
    )
    filename = dir_path / f"{target_tissue}.obs_id.tsv"
    with open(filename, "w") as fp:
        fp.writelines([f"{_id}\n" for _id in tissue_obs["soma_joinid"]])


def build_list(output_directory, version):
    import cellxgene_census

    output_directory = Path(output_directory)
    data_directory = output_directory / "metadata_preparation_dir"
    data_directory.mkdir(exist_ok=True, parents=True)

    census = cellxgene_census.open_soma(census_version=version)
    tissue_list = get_tissue_list(census)
    with open(data_directory.parent / "tissue_list.tsv", "w") as fp:
        fp.writelines(tissue_list)

    with Progress() as progress_bar:
        task = progress_bar.add_task("Processing ...", total=len(tissue_list))
        for target_tissue in tissue_list:
            progress_bar.update(task, advance=1, description=f"Processing {target_tissue}...")

            save_tissue_var(target_tissue, census, data_directory)
            save_tissue_obs(target_tissue, census, data_directory)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    build_list(cfg.output_dir, cfg.census_version)
