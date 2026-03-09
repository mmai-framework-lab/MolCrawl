#!/usr/bin/env python3
"""
Define constants for path settings used throughout the project
"""

import os

# import common module
from molcrawl.utils.environment_check import check_learning_source_dir

# datasetDefining constants for the destination directory
LEARNING_SOURCE_DIR = check_learning_source_dir()

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_refseq_tokenizer_path():
    return os.path.join(PROJECT_ROOT, GENOME_SEQUENCE_DIR, "spm_tokenizer.model")


def get_genome_tokenizer_path():
    """
    Get tokenizer path for genome sequence

    Returns:
        str: Genome sequence tokenizer path (string)
    """
    # Use tokenizer for RefSeq genome sequences
    return get_refseq_tokenizer_path()


# Base path of each dataset
def get_dataset_path(dataset_type, relative_path=""):
    """
    Function to get dataset path

    Args:
        dataset_type (str): dataset type ('uniprot', 'refseq', 'cellxgene', 'molecule_nat_lang')
        relative_path (str): relative path within the dataset

    Returns:
        str: complete path
    """
    if dataset_type == "molecule_nat_lang":
        base_path = os.path.join(PROJECT_ROOT, MOLECULE_NAT_LANG_DATASET_DIR)
    else:
        base_path = os.path.join(PROJECT_ROOT, GENOME_SEQUENCE_DIR, dataset_type)

    if relative_path:
        return os.path.join(base_path, relative_path)
    return base_path


# Commonly used path constants
PROTEIN_SEQUENCE_DIR = LEARNING_SOURCE_DIR + "/protein_sequence"
GENOME_SEQUENCE_DIR = LEARNING_SOURCE_DIR + "/genome_sequence"
RNA_DATASET_DIR = LEARNING_SOURCE_DIR + "/rna"
MOLECULE_NAT_LANG_DIR = LEARNING_SOURCE_DIR + "/molecule_nat_lang"
COMPOUNDS_DIR = LEARNING_SOURCE_DIR + "/compounds"
UNIPROT_DATASET_DIR = PROTEIN_SEQUENCE_DIR + "/training_ready_hf_dataset"
REFSEQ_DATASET_DIR = GENOME_SEQUENCE_DIR + "/training_ready_hf_dataset"
CELLXGENE_DATASET_DIR = RNA_DATASET_DIR + "/training_ready_hf_dataset"
MOLECULE_NAT_LANG_DATASET_DIR = MOLECULE_NAT_LANG_DIR + "/training_ready_hf_dataset"
COMPOUNDS_DATASET_DIR = COMPOUNDS_DIR + "/organix13/compounds/training_ready_hf_dataset"

# Absolute path version (used in web applications and APIs)
ABSOLUTE_LEARNING_SOURCE_PATH = os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR)

# Basic path of GPT-2 model output directory
GPT2_OUTPUT_BASE_DIR = "gpt2-output"
BERT_OUTPUT_BASE_DIR = "bert-output"


def get_gpt2_output_path(domain, model_size):
    """
    Function to get output path of GPT-2 model

    Args:
        domain (str): domain name ('protein_sequence', 'genome_sequence', 'rna', 'compounds', 'molecule_nat_lang')
        model_size (str): Model size ('small', 'medium', 'large', 'xl', 'ex-large')

    Returns:
        str: GPT-2 output directory path
    """
    # Standardize model_size
    if model_size == "xl":
        size_suffix = "ex-large"
    else:
        size_suffix = model_size

    return os.path.join(LEARNING_SOURCE_DIR, domain, GPT2_OUTPUT_BASE_DIR, f"{domain}-{size_suffix}")


# Commonly used GPT-2 output path constants
def get_gpt2_tensorboard_path(domain, model_size):
    """Get GPT-2 TensorBoard output path"""
    return get_gpt2_output_path(domain, model_size)


def get_gpt2_model_output_path(domain, model_size):
    """Get GPT-2 model output path"""
    return get_gpt2_output_path(domain, model_size)


def get_bert_output_path(domain, model_size):
    """
    Function to get output path of BERT model

    Args:
        domain (str): domain name ('protein_sequence', 'genome_sequence', 'rna', 'compounds', 'molecule_nat_lang')
        model_size (str): Model size ('small', 'medium', 'large')

    Returns:
        str: BERT output directory path
    """
    return os.path.join(LEARNING_SOURCE_DIR, domain, BERT_OUTPUT_BASE_DIR, f"{domain}-{model_size}")


def get_bert_tensorboard_path(domain, model_size):
    """Get BERT TensorBoard output path"""
    return get_bert_output_path(domain, model_size)


def get_bert_model_output_path(domain, model_size):
    """Get BERT model output path"""
    return get_bert_output_path(domain, model_size)


def get_custom_tokenizer_path(domain, model_type="bert"):
    """
    Get path for custom tokenizer output

    Args:
        domain (str): domain name ('genome_sequence', 'rna', etc.)
        model_type (str): model type ('bert', 'rnaformer', 'dnabert2')

    Returns:
        str: custom tokenizer directory path
    """
    return os.path.join(LEARNING_SOURCE_DIR, domain, f"custom_tokenizer_{model_type}")
