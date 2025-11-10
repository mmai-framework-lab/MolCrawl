from dataclasses import dataclass, field
from core.config import Config

from config.paths import GENOME_SEQUENCE_DIR


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


@dataclass
class GenomeSequenceConfig(Config):
    data_preparation: RefSeqPreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = RefSeqPreparationConfig(**self.data_preparation)
