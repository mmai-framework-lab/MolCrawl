import json
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

import yaml

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        """
        Only json and YAML format are supported, config are expected to
        have the correct suffix.
        """

        def get_dict_from_file(file_path):
            if Path(file_path).suffix == ".json":
                with open(file_path) as read_handle:
                    cfg = json.load(read_handle)
            elif Path(file_path).suffix in [".yaml", ".yml"]:
                with open(file_path) as read_handle:
                    cfg = yaml.load(read_handle, Loader=yaml.FullLoader)
            else:
                raise ValueError("The config file should be a json or yaml with a correct suffix")
            return cfg

        cfg = get_dict_from_file(file_path)
        return cls(**cfg)
