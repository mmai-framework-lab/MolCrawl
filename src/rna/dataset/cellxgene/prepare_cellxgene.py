from script.build_list import build_list
from script.download import download
from script.conv import convert


if __name__ == "__main__":
    size_workload = 10000
    num_worker = 8
    output_dir = "/nasa/datasets/riken/projects/fundamental_models_202407/cellxgene"

    build_list(output_dir)
    download(output_dir, num_worker, size_workload)
    convert(output_dir, num_worker)
