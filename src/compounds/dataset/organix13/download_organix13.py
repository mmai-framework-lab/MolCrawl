from opv.prepare_opv import OPV
from qm9_zinc250k_cep.convert_to_parquet import download_convert_zinc

if __name__ == "__main__":

    output_dir = "./organix13"

    OPV(output_dir)

    download_convert_zinc(output_dir)
