from typing import List, Union
from argparse import ArgumentParser
import logging
from pathlib import Path
import concurrent.futures
from functools import partial
import pickle

import rich.progress
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import pandas as pd
import rich

from protein_sequence.utils.configs import ProteinSequenceConfig

logger = logging.getLogger(__name__)

# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

MASK_STR_SHORT = "_"


class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    """
    Constructs an ESM tokenizer.
    """

    model_input_names = ["sequence_tokens", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # a character-level tokenizer is the same as BPE with no token merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]

        tokenizer.add_special_tokens(
            special_tokens,
        )

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    # These are a footgun, we never use the `bos` token anywhere so we're just overriding it here.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids


def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    add_special_tokens: bool = True,
) -> List[int]:
    sequence = sequence.replace(MASK_STR_SHORT, sequence_tokenizer.mask_token)
    return sequence_tokenizer.encode(sequence, add_special_tokens=add_special_tokens)


def process_raw(path_raw, path_parquet, tokenizer):
    tokenized_sequences = []
    with open(path_raw, "r") as raw_file:
        for line in raw_file:
            tokens = tokenize_sequence(line.strip(), tokenizer)
            tokenized_sequences.append((tokens, len(tokens)))
    df = pd.DataFrame(tokenized_sequences, columns=["token", "token_count"])
    df.to_parquet(path_parquet, index=False)
    return df["token_count"].to_list()


def get_parquet_paths(raw_paths: List[Path], parquet_dir: Union[Path, str]):
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    return [parquet_dir / path.with_suffix(".parquet").name for path in raw_paths]


def generate_parquet_from_raw(raw_dir: Path, parquet_dir: Path, num_worker=5):
    raw_paths = [path for path in Path(raw_dir).iterdir() if path.suffix == ".raw"]
    parquet_paths = get_parquet_paths(raw_paths, parquet_dir)

    tokenizer = EsmSequenceTokenizer()

    token_counts = []
    with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
        func = partial(process_raw, tokenizer=tokenizer)
        for result in rich.progress.track(
            executor.map(func, raw_paths, parquet_paths), "Tokenizing to parquet", total=len(raw_paths)
        ):
            token_counts += result

    with open(raw_dir.parent / "token_counts.pkl", "wb") as file:
        pickle.dump(token_counts, file)


def tokenize_to_parquet(output_dir: Union[str, Path], num_worker):
    raw_dir = Path(output_dir) / "raw_files"
    parquet_dir = Path(output_dir) / "parquet_files"
    generate_parquet_from_raw(raw_dir, parquet_dir, num_worker)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation

    tokenize_to_parquet(cfg.output_dir, cfg.num_worker)
