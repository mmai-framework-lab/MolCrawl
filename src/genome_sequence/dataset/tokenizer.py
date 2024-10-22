from pathlib import Path
from argparse import ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split

from genome_sequence.utils.config import GenomeSequenceConfig


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    path_dir = Path(cfg.output_dir) / "raw_files"
    files = [str(p) for p in path_dir.glob("*.raw")]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size, show_progress=True, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    pattern = r"[^A-Z]"
    tokenizer.pre_tokenizer = Split(pattern, behavior="removed")

    tokenizer.train(files, trainer)
    tokenizer.save(str(Path(cfg.output_dir) / "tokenizer.json"))
