#!/usr/bin/env python3
"""
Output directory management utility for AI model evaluation scripts

Generate a structured assessment report directory based on the LEARNING_SOURCE_DIR environment variable.
"""

import logging
from datetime import datetime
from pathlib import Path

from molcrawl.core.utils.environment_check import check_learning_source_dir

logger = logging.getLogger(__name__)


def get_evaluation_output_dir(model_type, evaluation_type, model_name=None, timestamp=None):
    """
    Generate output directory path for assessment report

        Args:
    model_type (str): Model type ('genome_sequence', 'protein_sequence', etc.)
    evaluation_type (str): Evaluation type ('proteingym', 'clinvar', 'protein_classification', etc.)
    model_name (str, optional): Model name (automatically generated if not specified)
    timestamp (str, optional): timestamp (current time if not specified)

        Returns:
    Path: Path of evaluation result output directory

        Example:
            get_evaluation_output_dir('genome_sequence', 'clinvar')
            -> {LEARNING_SOURCE_DIR}/genome_sequence/report/clinvar_20241015_143022

            get_evaluation_output_dir('protein_sequence', 'proteingym', 'bert_medium')
            -> {LEARNING_SOURCE_DIR}/protein_sequence/report/proteingym_bert_medium_20241015_143022
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    learning_source_dir = Path(check_learning_source_dir())

    # Model type directory (genome_sequence, protein_sequence, etc.)
    model_type_dir = learning_source_dir / model_type

    # report directory
    report_dir = model_type_dir / "report"

    # Generate directory name including evaluation type and model name
    if model_name:
        dir_name = f"{evaluation_type}_{model_name}_{timestamp}"
    else:
        dir_name = f"{evaluation_type}_{timestamp}"

    output_dir = report_dir / dir_name

    # create directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created evaluation output directory: {output_dir}")

    return output_dir


def get_model_type_from_path(model_path):
    """
    Infer model type from model path

        Args:
    model_path (str): model path

        Returns:
    str: estimated model type
    """
    model_path_str = str(model_path).lower()

    if "genome" in model_path_str:
        return "genome_sequence"
    elif "protein" in model_path_str:
        return "protein_sequence"
    elif "compound" in model_path_str:
        return "compounds"
    elif "rna" in model_path_str:
        return "rna"
    elif "molecule" in model_path_str:
        return "molecule_nat_lang"
    else:
        return "general"


def get_model_name_from_path(model_path):
    """
    Infer model name from model path

        Args:
    model_path (str): model path

        Returns:
    str: Estimated model name
    """
    model_path = Path(model_path)

    # Infer model name from the last part of the path
    if model_path.is_dir():
        model_name = model_path.name
    else:
        model_name = model_path.stem

    # remove common prefixes/suffixes
    model_name = model_name.replace("runs_train_", "")
    model_name = model_name.replace("bert_", "")
    model_name = model_name.replace("gpt2_", "")

    return model_name


def setup_evaluation_logging(output_dir, script_name):
    """
    Log settings for evaluation scripts

        Args:
    output_dir (Path): Output directory
    script_name (str): Script name

        Returns:
    logging.Logger: configured logger
    """
    log_file = output_dir / f"{script_name}.log"

    # log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # logger settings
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_evaluation_summary(output_dir, evaluation_info):
    """
    Create evaluation summary file

        Args:
    output_dir (Path): Output directory
    evaluation_info (dict): Evaluation information
    """
    summary_file = output_dir / "evaluation_summary.json"

    import json

    with open(summary_file, "w") as f:
        json.dump(evaluation_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation summary saved to: {summary_file}")


if __name__ == "__main__":
    # test execution
    print("Testing evaluation output directory generation...")

    # test 1: genome_sequence + clinvar
    output_dir1 = get_evaluation_output_dir("genome_sequence", "clinvar")
    print(f"Test 1: {output_dir1}")

    # Test 2: protein_sequence + proteingym + model name
    output_dir2 = get_evaluation_output_dir("protein_sequence", "proteingym", "bert_medium")
    print(f"Test 2: {output_dir2}")

    # Test 3: Model type estimation
    model_type = get_model_type_from_path("runs_train_bert_genome_sequence")
    print(f"Test 3: {model_type}")

    # Test 4: Model name estimation
    model_name = get_model_name_from_path("runs_train_bert_protein_sequence_medium")
    print(f"Test 4: {model_name}")
