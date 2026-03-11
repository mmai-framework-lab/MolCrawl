"""
Mol-Instructions Dataset Preparation Script for GPT-2 / BERT Training

Converts zjunlp/Mol-Instructions (Molecule-oriented Instructions) into the
training_ready_hf_dataset format that is compatible with the existing
molecule_nat_lang training pipeline.

Key differences from the SMolInstruct preparation (prepare_gpt2.py):
  - Mol-Instructions uses SELFIES notation (e.g. [C][C@H1]...) instead of SMILES
  - Fields are: instruction + input + output  (no "task" field in the same form)
  - Splits in the HF dataset are task names, not train/validation/test; this
    script merges all tasks and performs an 80/10/10 random split.
  - The instruction text is prepended to the input to form the final "input"
    in the format expected by MoleculeNatLangTokenizer.tokenize_dict(), i.e.
    {"input": "<full prompt>", "output": "<target text>"}.
  - Molecules in SELFIES are wrapped in <MOLECULE>...</MOLECULE> tags so the
    tokenizer does not try to canonicalise them as SMILES.

Usage (standalone):
    LEARNING_SOURCE_DIR=learning_source_20260311 \\
    python -m molcrawl.molecule_nat_lang.dataset.prepare_mol_instructions \\
        assets/configs/molecule_nat_lang_config.yaml

    The output is saved to:
        $LEARNING_SOURCE_DIR/molecule_nat_lang/mol_instructions/training_ready_hf_dataset/
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# ── Task split names in zjunlp/Mol-Instructions ─────────────────────────────
MOL_INSTRUCTIONS_SPLITS = (
    "description_guided_molecule_design",
    "forward_reaction_prediction",
    "molecular_description_generation",
    "property_prediction",
    "reagent_prediction",
    "retrosynthesis",
)

# SELFIES molecule tag (distinct from <SMILES> so the tokenizer skips
# the RDKit canonicalisation step)
MOLECULE_TAG_LEFT = "<MOLECULE>"
MOLECULE_TAG_RIGHT = "</MOLECULE>"


# ── Helpers shared with prepare_gpt2.py ─────────────────────────────────────


def concatenate_texts(examples, eos_token_id):
    """Concatenate input_ids + output_ids for each sample into a flat sequence."""
    concatenated_ids = []
    for input_ids, output_ids in zip(examples["input_ids"], examples["output_ids"]):
        concatenated_ids.extend(input_ids + output_ids)
    return {"input_ids": concatenated_ids}


def create_chunks(examples, context_length):
    """Split a flat token sequence into fixed-length chunks."""
    concatenated_ids = examples["input_ids"]
    total_length = (len(concatenated_ids) // context_length) * context_length
    concatenated_ids = concatenated_ids[:total_length]
    input_ids = [concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)]
    return {"input_ids": input_ids}


# ── Mol-Instructions specific helpers ────────────────────────────────────────


def _wrap_molecule(text: str) -> str:
    """
    Wrap a molecule string (SELFIES or SMILES) in <MOLECULE>...</MOLECULE> tags
    if it does not already contain them.

    Mol-Instructions stores bare SELFIES/SMILES in the input/output fields with
    no surrounding tags.  We add tags to make the format consistent with the
    SMolInstruct-style data and to prevent the tokenizer from treating the
    string as SMILES and attempting canonicalisation.
    """
    text = text.strip()
    if MOLECULE_TAG_LEFT in text:
        return text
    return f"{MOLECULE_TAG_LEFT} {text} {MOLECULE_TAG_RIGHT}"


def _sample_is_molecule_field(text: str) -> bool:
    """
    Heuristically detect whether a field contains a bare molecule string
    (SELFIES or SMILES) rather than natural language.

    SELFIES strings start with '['.  SMILES strings typically start with a
    letter, digit or special character and contain no spaces.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # SELFIES: always starts with '['
    if stripped.startswith("["):
        return True
    # SMILES: no whitespace and contains chemistry-like characters
    if " " not in stripped and any(c in stripped for c in ("=", "#", "(", ")", ">>", "@")):
        return True
    return False


def convert_sample(sample: dict) -> dict:
    """
    Convert a Mol-Instructions sample into the
    {"input": str, "output": str} format expected by
    MoleculeNatLangTokenizer.tokenize_dict().

    The instruction is prepended to the input so the model sees the full
    task description.  Molecule strings in both input and output are wrapped
    with <MOLECULE> tags.
    """
    instruction = sample.get("instruction", "").strip()
    raw_input = sample.get("input", "").strip()
    raw_output = sample.get("output", "").strip()

    # Wrap molecule strings in tags
    wrapped_input = _wrap_molecule(raw_input) if _sample_is_molecule_field(raw_input) else raw_input
    wrapped_output = _wrap_molecule(raw_output) if _sample_is_molecule_field(raw_output) else raw_output

    # Combine instruction and input (mirror the SMolInstruct format where the
    # instruction is already embedded in the "input" field)
    if instruction:
        combined_input = f"{instruction}\n{wrapped_input}" if wrapped_input else instruction
    else:
        combined_input = wrapped_input

    return {"input": combined_input, "output": wrapped_output}


def validate_and_tokenize(example, tokenizer):
    """
    Tokenize a single converted Mol-Instructions sample.

    Returns a dict with input_ids / attention_mask / labels / output_ids or a
    default skeleton with valid_sample=False on failure.
    """

    def _default():
        return {
            "valid_sample": False,
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "output_ids": [],
            "input_text": example.get("input", ""),
            "real_input_text": "",
            "input_too_long": False,
            "task_type": example.get("task_type", "unknown"),
        }

    try:
        result = tokenizer.tokenize_dict(
            example,
            # Disable SMILES canonicalisation — content is SELFIES not SMILES
            canonicalize_smiles=False,
        )
        if not isinstance(result, dict):
            return _default()
        result.setdefault("valid_sample", True)
        result.setdefault("task_type", example.get("task_type", "unknown"))
        return result
    except Exception as exc:
        logger.debug("Tokenisation failed for sample: %s", exc)
        return _default()


# ── Main preparation function ─────────────────────────────────────────────────


def prepare_mol_instructions(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    context_length: int = 1024,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_proc: int = 4,
):
    """
    Load Mol-Instructions from *source_dir* (a HF DatasetDict saved to disk),
    convert all task-splits into a standard train/valid/test split, tokenize,
    chunk to *context_length*, and save to *output_dir*.

    Args:
        source_dir: Path to the directory produced by download_mol_instructions().
        output_dir: Where to write the training_ready_hf_dataset/.
        context_length: Token sequence length for GPT-2 training.
        train_ratio: Fraction used for training (the rest is split 50/50 between
            validation and test).
        seed: Random seed for reproducible splits.
        num_proc: Number of parallel workers for Dataset.map().
    """
    import numpy as np
    from datasets import Dataset, DatasetDict, load_from_disk

    from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Mol-Instructions source directory not found: {source_dir}\n"
            "Run the download step first:\n"
            "  python -m molcrawl.preparation.preparation_script_molecule_related_nat_lang "
            "assets/configs/molecule_nat_lang_config.yaml --datasets mol_instructions --download-only"
        )

    logger.info("Loading Mol-Instructions from %s", source_dir)
    raw = load_from_disk(str(source_dir))
    logger.info("Available task splits: %s", list(raw.keys()))

    # ── 1. Merge all task splits and add task_type column ────────────────────
    all_rows = []
    for task_split in raw.keys():
        for row in raw[task_split]:
            converted = convert_sample(row)
            converted["task_type"] = task_split
            all_rows.append(converted)

    logger.info("Total samples after merging all tasks: %d", len(all_rows))

    # ── 2. Random 80/10/10 split ─────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_rows))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * (1 - train_ratio) / 2)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _make_split(idx_array):
        rows = [all_rows[i] for i in idx_array]
        return Dataset.from_list(rows)

    raw_split = DatasetDict(
        {
            "train": _make_split(train_idx),
            "valid": _make_split(val_idx),
            "test": _make_split(test_idx),
        }
    )
    logger.info(
        "Split sizes — train: %d, valid: %d, test: %d",
        len(raw_split["train"]),
        len(raw_split["valid"]),
        len(raw_split["test"]),
    )

    # ── 3. Tokenise ───────────────────────────────────────────────────────────
    logger.info("Initialising tokenizer...")
    tokenizer = MoleculeNatLangTokenizer()

    logger.info("Tokenising dataset (this may take a while)...")
    # num_proc=1: the tokenizer object is not picklable (it may use a fallback
    # BasicTokenizer internally), so multiprocessing cannot serialise the
    # closure.  Concatenation and chunking use plain top-level functions and
    # run in parallel further below.
    tokenized = raw_split.map(
        lambda ex: validate_and_tokenize(ex, tokenizer),
        batched=False,
        num_proc=1,
        desc="Tokenising",
    )

    # Keep only valid samples (same reason: lambda captures tokenizer scope)
    for split_name in tokenized:
        before = len(tokenized[split_name])
        tokenized[split_name] = tokenized[split_name].filter(lambda ex: ex["valid_sample"], num_proc=1)
        after = len(tokenized[split_name])
        logger.info("  %s: %d → %d samples (removed %d invalid)", split_name, before, after, before - after)

    # ── 4. Concatenate input_ids + output_ids, then chunk ────────────────────
    logger.info("Concatenating and chunking sequences to length %d...", context_length)
    eos_token_id = tokenizer.tokenizer.eos_token_id

    concatenated = tokenized.map(
        partial(concatenate_texts, eos_token_id=eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=tokenized["train"].column_names,
        desc="Concatenating",
    )

    chunked = concatenated.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
        desc="Chunking",
    )

    # ── 5. Save ───────────────────────────────────────────────────────────────
    output_path = output_dir / "training_ready_hf_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving prepared dataset to %s", output_path)
    chunked.save_to_disk(str(output_path))

    logger.info("Done! Dataset statistics:")
    for split_name in chunked:
        logger.info("  %s: %d chunks of %d tokens", split_name, len(chunked[split_name]), context_length)

    return str(output_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging.config

    from molcrawl.core.base import setup_logging
    from molcrawl.molecule_nat_lang.utils.config import MoleculeNLConfig
    from molcrawl.utils.environment_check import check_learning_source_dir

    parser = ArgumentParser(description="Prepare Mol-Instructions for GPT-2/BERT training")
    parser.add_argument("config", help="Path to molecule_nat_lang YAML config")
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Token sequence chunk length (default: 1024)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel workers for tokenisation (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid/test split (default: 42)",
    )
    args = parser.parse_args()

    learning_source_dir = check_learning_source_dir()
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation

    mol_instructions_dir = Path(learning_source_dir) / "molecule_nat_lang" / "mol_instructions" / "zjunlp_Mol-Instructions"
    output_dir = Path(learning_source_dir) / "molecule_nat_lang" / "mol_instructions"
    log_dir = output_dir / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_dir))

    logger.info("Source dir : %s", mol_instructions_dir)
    logger.info("Output dir : %s", output_dir)

    prepare_mol_instructions(
        source_dir=str(mol_instructions_dir),
        output_dir=str(output_dir),
        context_length=args.context_length,
        num_proc=args.num_proc,
        seed=args.seed,
    )
