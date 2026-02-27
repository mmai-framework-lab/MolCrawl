from dataclasses import dataclass, field
from typing import Any, Dict, Union, Optional

from molcrawl.config.paths import GENOME_SEQUENCE_DIR
from molcrawl.core.config import Config


@dataclass
class RefSeqPreparationConfig:
    # Output directory where the preparation will be made
    output_dir: str = GENOME_SEQUENCE_DIR
    # Path to a directory containing one file per species to download from refseq (see assets/genome_species_list/species for example)
    # Possible groups are archaea, bacteria, fungi, invertebrate, metagenomes, plant, protozoa, vertebrate_mammalian, vertebrate_other, viral.
    path_species: str = "assets/genome_species_list/filtered_species_refseq"

    # Num of parallel worker to use, note that for download the worker are capped to 3
    num_worker: int = 16

    max_lines_per_file: int = 10000

    # Size of the vocabulary of the BPE tokenizer
    vocab_size: int = 4096

    # Number of genome sequence to use to train the BPE tokenizer.
    # We will sample input_sentence_size randomly from input_sentence_size * 2 number of sequence.
    # So input_sentence_size * 2 / max_lines_per_file will be randomly selected for the BPE training.
    input_sentence_size: int = 700000

    # 追加: 高速化用オプション（任意）
    num_proc_parquet: Optional[int] = None
    parquet_batch_size: Optional[int] = None
    local_base_dir: Optional[str] = None


@dataclass
class GenomeSequenceConfig(Config):
    data_preparation: Union[RefSeqPreparationConfig, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.data_preparation, dict):
            self.data_preparation = RefSeqPreparationConfig(**self.data_preparation)
