from dataclasses import dataclass, field
from utils.config import Config


@dataclass
class Organix13PreparationConfig:
    pass


@dataclass
class CompoundConfig(Config):

    data_preparation: Organix13PreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = Organix13PreparationConfig(**self.data_preparation)
