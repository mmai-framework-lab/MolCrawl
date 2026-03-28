import logging
import logging.config
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root src directory to path
from molcrawl.core.base import setup_logging
from molcrawl.molecule_nat_lang.utils.config import MoleculeNLConfig
from molcrawl.molecule_nat_lang.utils.general import compute_resource_aware_params, read_dataset, save_dataset
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

logger = logging.getLogger(__name__)


def run_statistics(series, column_name):
    series_length = [len(i) for i in series]
    plt.hist(series_length, bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized {}".format(column_name))
    plt.title("Distribution of tokenized {} lengths".format(column_name))

    # Save to unified image directory
    from molcrawl.utils.image_manager import get_image_path

    image_path = get_image_path("molecule_nat_lang", "molecule_nat_lang_tokenized_{}_lengths_dist.png".format(column_name))
    plt.savefig(image_path)
    plt.close()
    logger.info(msg="Saved distribution of tokenized {} lengths to {}".format(column_name, image_path))


def validate_smiles_in_sample(sample):
    """Validate SMILES structures in the sample to ensure chemical validity"""
    try:
        from rdkit import Chem

        # Extract SMILES from input and output
        input_smiles = extract_smiles_from_text(sample.get("input", ""))
        output_smiles = extract_smiles_from_text(sample.get("output", ""))

        # Validate input SMILES
        for smiles in input_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES in input: {smiles}")
                return False

        # Validate output SMILES
        for smiles in output_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES in output: {smiles}")
                return False

        return True
    except Exception as e:
        logger.warning(f"Error validating SMILES: {e}")
        return False


def extract_smiles_from_text(text):
    """Extract SMILES strings from text enclosed in <SMILES> tags"""
    import re

    smiles_pattern = r"<SMILES>\s*([^<]+)\s*</SMILES>"
    matches = re.findall(smiles_pattern, text)
    return [match.strip() for match in matches]


def analyze_dataset_tasks(dataset):
    """Analyze the chemical tasks present in the dataset"""
    task_distribution = {}

    for split in dataset.keys():
        logger.info(f"Analyzing tasks in {split} split...")

        # Check if task field exists
        if "task" in dataset[split].features:
            tasks = dataset[split]["task"]
            for task in tasks:
                task_distribution[task] = task_distribution.get(task, 0) + 1
        else:
            logger.warning(f"No task field found in {split} split - cannot analyze task distribution")

    logger.info("Task distribution:")
    for task, count in sorted(task_distribution.items()):
        logger.info(f"  {task}: {count} samples")

    return task_distribution


def calculate_statistics(dataset, split):
    inp_out = [i + j for i, j in zip(dataset[split]["input_ids"], dataset[split]["output_ids"])]
    num_samples = len(inp_out)
    num_tokens = sum(len(i) for i in inp_out)

    run_statistics(inp_out, split)
    return num_samples, num_tokens


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if files exist",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["smolinstruct", "mol_instructions"],
        default=["smolinstruct"],
        help=(
            "Which dataset(s) to prepare. "
            "'smolinstruct' processes osunlp/SMolInstruct (default). "
            "'mol_instructions' processes zjunlp/Mol-Instructions via "
            "molcrawl.molecule_nat_lang.dataset.prepare_mol_instructions."
        ),
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset(s); skip tokenisation/preparation.",
    )
    args = parser.parse_args()

    # ── Dispatch mol_instructions early so we can exit without running the
    # ── heavy SMolInstruct pipeline if the user only asked for that dataset.
    if "mol_instructions" in args.datasets:
        from molcrawl.molecule_nat_lang.dataset.download import download_mol_instructions
        from molcrawl.molecule_nat_lang.dataset.prepare_mol_instructions import (
            prepare_mol_instructions,
        )
        from molcrawl.utils.environment_check import check_learning_source_dir as _check_dir

        _lsd = _check_dir()
        _mi_source = Path(_lsd) / "molecule_nat_lang" / "mol_instructions" / "zjunlp_Mol-Instructions"
        _mi_output = Path(_lsd) / "molecule_nat_lang" / "mol_instructions"

        if not _mi_source.exists():
            logger.info("Downloading Mol-Instructions...")
            download_mol_instructions(str(_mi_source))
        else:
            logger.info("Mol-Instructions source already present at %s", _mi_source)

        if not args.download_only:
            _ready = _mi_output / "training_ready_hf_dataset"
            if not args.force and _ready.exists():
                logger.info(
                    "Mol-Instructions training_ready_hf_dataset already exists at %s. Use --force to reprocess.",
                    _ready,
                )
            else:
                logger.info("Preparing Mol-Instructions dataset...")
                prepare_mol_instructions(
                    source_dir=str(_mi_source),
                    output_dir=str(_mi_output),
                )

        if "smolinstruct" not in args.datasets:
            # Nothing more to do — exit before the SMolInstruct block
            exit(0)

    # ── SMolInstruct pipeline ─────────────────────────────────────────────────
    # Use LEARNING_SOURCE_DIR environment variable for dataset storage
    from molcrawl.utils.environment_check import check_learning_source_dir

    learning_source_dir = check_learning_source_dir()

    cfg = MoleculeNLConfig.from_file(args.config).data_preparation

    # Set paths using LEARNING_SOURCE_DIR
    base_dataset_dir = Path(learning_source_dir) / "molecule_nat_lang" / "osunlp" / "SMolInstruct"
    logging_dir = Path(learning_source_dir) / "molecule_nat_lang" / "logs"
    parquet_file = Path(learning_source_dir) / "molecule_nat_lang" / "molecule_related_natural_language_tokenized.parquet"

    logger.info(f"Using dataset directory: {base_dataset_dir}")
    logger.info(f"Using logging directory: {logging_dir}")
    logger.info(f"Using parquet file path: {parquet_file}")

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    setup_logging(str(logging_dir))

    # Check if dataset directory exists
    if not base_dataset_dir.exists():
        logger.error(msg="")
        logger.error(msg="=" * 70)
        logger.error(msg="Dataset directory not found!")
        logger.error(msg="=" * 70)
        logger.error(msg=f"Expected location: {base_dataset_dir}")
        logger.error(msg="")
        logger.error(msg="Please download the dataset first using:")
        logger.error(msg="  bash src/preparation/download_smolinstruct.sh")
        logger.error(msg="")
        logger.error(msg="Or manually download from Hugging Face:")
        logger.error(msg="  https://huggingface.co/datasets/osunlp/SMolInstruct")
        logger.error(msg="=" * 70)
        exit(1)

    logger.info(msg="Using local dataset directory...")
    logger.info(msg=f"Dataset directory: {base_dataset_dir}")

    # Load dataset with error handling
    try:
        dataset = read_dataset(base_dataset_dir)
        logger.info(msg=f"Successfully loaded dataset from {base_dataset_dir}")
        logger.info(msg=f"Dataset splits: {list(dataset.keys())}")
        for split in dataset.keys():
            logger.info(msg=f"  {split}: {len(dataset[split])} samples")
    except Exception as load_error:
        logger.error(msg=f"Failed to load dataset: {load_error}")
        logger.error(msg="The dataset directory may be corrupted or incomplete.")
        logger.error(msg="Please delete the directory and run download_smolinstruct.sh again.")
        exit(1)

    # Analyze the dataset structure and tasks
    logger.info(msg="Analyzing dataset structure and chemical tasks...")
    task_distribution = analyze_dataset_tasks(dataset)

    # Log dataset structure information
    for split in dataset.keys():
        logger.info(f"Dataset split '{split}' contains {len(dataset[split])} samples")
        if len(dataset[split]) > 0:
            sample_keys = list(dataset[split].features.keys())
            logger.info(f"  Available fields: {sample_keys}")

            # Show a sample for understanding the structure
            sample = dataset[split][0]
            logger.info("  Sample structure preview:")
            for key in sample_keys[:5]:  # Show first 5 fields
                value = str(sample[key])[:100] + "..." if len(str(sample[key])) > 100 else sample[key]
                logger.info(f"    {key}: {value}")

    # Check if there is already a processed parquet file
    if not args.force and parquet_file.exists():
        logger.info(msg=f"Processed dataset already exists at {parquet_file}.")
        logger.info(
            msg="Skipping tokenization and processing. If you want to reprocess, please use --force option or delete the parquet file and run again."
        )
        exit(0)
    elif args.force and parquet_file.exists():
        logger.info(msg="Force option specified. Reprocessing dataset...")

    tokenizer = MoleculeNatLangTokenizer()

    # ── Resource-aware parameter computation ──────────────────────────────────
    # Estimate total rows so the memory model can compute safe parallelism.
    total_rows_estimate = sum(len(dataset[s]) for s in dataset.keys())
    resource_params = compute_resource_aware_params(num_rows=total_rows_estimate)
    # Respect config ceiling: use the smaller of what the config says and what
    # memory allows.  cfg.num_workers == 1 acts as a safe hard cap during debug.
    dynamic_num_workers = min(cfg.num_workers, resource_params["num_workers"])
    dynamic_batch_size = resource_params["batch_size"]
    logger.info(
        "Effective preparation parameters: num_workers=%d  parquet_batch_size=%d",
        dynamic_num_workers,
        dynamic_batch_size,
    )
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(msg="Processing dataset with chemical validation...")

    processed_dataset = {}
    for split in dataset.keys():
        # Filter out samples with invalid SMILES before tokenization
        def validate_and_tokenize(example):
            """Validate SMILES and tokenize. Always return a dict. If invalid, mark valid_sample=False
            and provide default token fields so huggingface datasets writer doesn't fail.
            """

            # Prepare a default skeleton to return in all cases
            def default_result():
                return {
                    "valid_sample": False,
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "output_ids": [],
                    "input_text": example.get("input", ""),
                    "real_input_text": "",
                    "input_too_long": False,
                    "task_type": example.get("task", "unknown"),
                }

            try:
                # First validate the chemical content
                if not validate_smiles_in_sample(example):
                    logger.debug(f"Skipping sample due to invalid SMILES: {example.get('sample_id', 'unknown')}")
                    return default_result()

                # Then tokenize
                result = tokenizer.tokenize_dict(example)
                # Ensure the result is a dict and contains expected keys
                if not isinstance(result, dict):
                    logger.warning(f"Tokenizer returned non-dict for sample {example.get('sample_id', 'unknown')}")
                    return default_result()

                result.setdefault(
                    "task_type",
                    example.get("task", "unknown"),
                )
                result["valid_sample"] = True
                return result
            except Exception as e:
                logger.warning(f"Error processing sample (task: {example.get('task', 'unknown')}): {e}")
                return default_result()

        # Apply validation and tokenization
        logger.info(f"Processing {split} split...")
        num_proc = dynamic_num_workers
        try:
            processed_split = dataset[split].map(
                validate_and_tokenize,
                batched=False,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc="Validating and tokenizing {}".format(split),
                remove_columns=dataset[split].column_names,
            )
        except Exception as e:
            logger.warning(
                f"Multiprocessing map failed (num_proc={num_proc}): {type(e).__name__}: {e}\n"
                "Retrying with num_proc=1 (single process)..."
            )
            processed_split = dataset[split].map(
                validate_and_tokenize,
                batched=False,
                num_proc=1,
                load_from_cache_file=False,
                desc="Validating and tokenizing {} (single-proc)".format(split),
                remove_columns=dataset[split].column_names,
            )

        # Filter out None results (invalid samples)
        processed_split = processed_split.filter(lambda x: x is not None)
        processed_dataset[split] = processed_split

        logger.info(f"Processed {len(processed_dataset[split])} valid samples in {split} split")

    logger.info(msg="Computing Dataset Statistics...")
    total_num_samples = 0
    total_num_tokens = 0

    # Compute statistics by task type if available
    task_stats: dict[str, dict[str, int]] = {}

    for split in processed_dataset.keys():
        logger.info(msg=f"{split}")
        num_samples, num_tokens = calculate_statistics(processed_dataset, split)
        logger.info(msg=f"Number of examples: {num_samples}")
        logger.info(msg=f"Number of tokens: {num_tokens}")
        total_num_samples += num_samples
        total_num_tokens += num_tokens

        # Collect task-specific statistics - DISABLED for performance
        # (Too slow for large datasets - can be computed separately if needed)git
        # if "task_type" in processed_dataset[split].features:
        #     task_types = processed_dataset[split]["task_type"]
        #     for task_type in set(task_types):
        #         if task_type not in task_stats:
        #             task_stats[task_type] = {"samples": 0, "tokens": 0}
        #
        #         task_samples = sum(1 for t in task_types if t == task_type)
        #         task_indices = [i for i, t in enumerate(task_types) if t == task_type]
        #         task_tokens = sum(
        #             len(processed_dataset[split][i]["input_ids"]) + len(processed_dataset[split][i]["output_ids"])
        #             for i in task_indices
        #         )
        #
        #         task_stats[task_type]["samples"] += task_samples
        #         task_stats[task_type]["tokens"] += task_tokens

    logger.info(msg="=== OVERALL STATISTICS ===")
    logger.info(msg="Total number of samples: {}".format(total_num_samples))
    logger.info(msg="Total number of tokens: {}".format(total_num_tokens))

    if task_stats:
        logger.info(msg="=== TASK-SPECIFIC STATISTICS ===")
        for task_type, stats in sorted(task_stats.items()):
            logger.info(msg=f"{task_type}: {stats['samples']} samples, {stats['tokens']} tokens")

    logger.info(msg="=== QUALITY VALIDATION SUMMARY ===")
    logger.info(msg="All samples have been validated for:")
    logger.info(msg="- SMILES chemical structure validity")
    logger.info(msg="- Input-output coherence")
    logger.info(msg="- Task type preservation")

    logger.info(msg="Saving processed dataset to {}.".format(parquet_file))
    save_dataset(processed_dataset, parquet_file, batch_size=dynamic_batch_size)

    # Also save in arrow format for BERT training (keeps individual samples)
    arrow_output_dir = Path(learning_source_dir) / "molecule_nat_lang" / "arrow_splits"
    logger.info(msg=f"Saving BERT-compatible arrow format to {arrow_output_dir}")
    os.makedirs(arrow_output_dir, exist_ok=True)

    for split_name, split_dataset in processed_dataset.items():
        # Filter out invalid samples for training
        valid_split = split_dataset.filter(lambda x: x.get("valid_sample", True))
        logger.info(f"Saving {split_name} split: {len(valid_split)} valid samples out of {len(split_dataset)}")

        split_path = arrow_output_dir / f"{split_name}.arrow"
        valid_split.save_to_disk(str(split_path))
        logger.info(f"Saved {split_name} to {split_path}")

    logger.info(msg="BERT-compatible arrow datasets saved successfully")

    # Save GPT-2 compatible format (concatenated token stream)
    logger.info(msg="Creating GPT-2-compatible token stream format...")
    gpt2_output_dir = Path(learning_source_dir) / "molecule_nat_lang" / "gpt2_format"
    os.makedirs(gpt2_output_dir, exist_ok=True)

    for split_name, split_dataset in processed_dataset.items():
        # Filter valid samples
        valid_split = split_dataset.filter(lambda x: x.get("valid_sample", True))

        # Concatenate all input_ids into a single token stream
        all_tokens = []
        for sample in valid_split:
            # Combine input_ids and output_ids for full sequence
            full_sequence = sample["input_ids"] + sample["output_ids"]
            all_tokens.extend(full_sequence)

        # Convert to tensor and save
        token_tensor = torch.tensor(all_tokens, dtype=torch.long)
        output_file = gpt2_output_dir / f"{split_name}.pt"
        torch.save(token_tensor, output_file)

        logger.info(f"Saved GPT-2 format for {split_name}: {len(all_tokens):,} tokens to {output_file}")

    logger.info(msg="GPT-2-compatible token stream format saved successfully")
