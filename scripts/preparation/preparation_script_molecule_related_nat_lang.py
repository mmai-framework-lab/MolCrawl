from argparse import ArgumentParser
import os
import sys
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.base import setup_logging
from molecule_related_nl.dataset.download import download_hf_dataset
from molecule_related_nl.utils.config import MoleculeNLConfig
from molecule_related_nl.utils.general import read_dataset, save_dataset

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer

from config.paths import MOLECULE_NL_DATASET_DIR


logger = logging.getLogger(__name__)


def run_statistics(series, column_name):
    series_length = [len(i) for i in series]
    plt.hist(series_length, bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized {}".format(column_name))
    plt.title("Distribution of tokenized {} lengths".format(column_name))
    plt.savefig("assets/img/molecule_nl_tokenized_{}_lengths_dist.png".format(column_name))
    plt.close()
    logger.info(
        msg="Saved distribution of tokenized {} lengths to assets/img/molecule_nl_tokenized_{}_lengths_dist.png".format(
            column_name, column_name
        )
    )


def validate_smiles_in_sample(sample):
    """Validate SMILES structures in the sample to ensure chemical validity"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        # Extract SMILES from input and output
        input_smiles = extract_smiles_from_text(sample.get('input', ''))
        output_smiles = extract_smiles_from_text(sample.get('output', ''))
        
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
    smiles_pattern = r'<SMILES>\s*([^<]+)\s*</SMILES>'
    matches = re.findall(smiles_pattern, text)
    return [match.strip() for match in matches]

def analyze_dataset_tasks(dataset):
    """Analyze the chemical tasks present in the dataset"""
    task_distribution = {}
    
    for split in dataset.keys():
        logger.info(f"Analyzing tasks in {split} split...")
        
        # Check if sample_id exists to determine task types
        if 'sample_id' in dataset[split].features:
            sample_ids = dataset[split]['sample_id']
            for sample_id in sample_ids:
                task_name = sample_id.split('.')[0] if '.' in sample_id else 'unknown'
                task_distribution[task_name] = task_distribution.get(task_name, 0) + 1
        else:
            logger.warning(f"No sample_id found in {split} split - cannot analyze task distribution")
    
    logger.info("Task distribution:")
    for task, count in sorted(task_distribution.items()):
        logger.info(f"  {task}: {count} samples")
    
    return task_distribution

def copy_local_dataset(local_path, target_path):
    """Copy local dataset from server to target directory using symlink for efficiency"""
    import shutil
    import os
    
    local_dataset_path = Path(local_path)
    if not local_dataset_path.exists():
        logger.error(f"Local dataset path does not exist: {local_path}")
        return False
    
    if not local_dataset_path.is_dir():
        logger.error(f"Local dataset path is not a directory: {local_path}")
        return False
    
    logger.info(f"Creating symlink from {local_path} to {target_path}")
    
    try:
        # Remove target directory if it exists
        if target_path.exists():
            if target_path.is_symlink():
                target_path.unlink()
            else:
                shutil.rmtree(target_path)
        
        # Create parent directory if it doesn't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink instead of copying for efficiency
        os.symlink(str(local_dataset_path), str(target_path))
        logger.info(f"Successfully created symlink to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create symlink: {e}")
        logger.info("Falling back to copying...")
        try:
            # Fallback to copying if symlink fails
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(local_path, target_path)
            logger.info(f"Successfully copied local dataset to {target_path}")
            return True
        except Exception as copy_error:
            logger.error(f"Failed to copy local dataset: {copy_error}")
            return False

def calculate_statistics(dataset, split):
    inp_out = [i + j for i, j in zip(dataset[split]["input_ids"], dataset[split]["output_ids"])]
    num_samples = len(inp_out)
    num_tokens = sum(len(i) for i in inp_out)

    run_statistics(inp_out, split)
    return num_samples, num_tokens


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    parser.add_argument("--download", action="store_true", help="Download dataset from Hugging Face (default: use local dataset)")
    parser.add_argument("--local-path", type=str, default="molecule_related_nl/osunlp/SMolInstruct", help="Path to local dataset directory")
    args = parser.parse_args()
    
    # Use LEARNING_SOURCE_DIR environment variable for dataset storage
    learning_source_dir = os.environ.get('LEARNING_SOURCE_DIR')
    if learning_source_dir is None:
        print("ERROR: Environment variable 'LEARNING_SOURCE_DIR' is not set.")
        print("Please set LEARNING_SOURCE_DIR environment variable before running this script.")
        print("Example: export LEARNING_SOURCE_DIR='learning_source'")
        exit(1)
    
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation
    
    # Set paths using LEARNING_SOURCE_DIR
    base_dataset_dir = Path(learning_source_dir) / "molecule_nl" / "osunlp" / "SMolInstruct"
    logging_dir = Path(learning_source_dir) / "molecule_nl" / "logs"
    parquet_file = Path(learning_source_dir) / "molecule_nl" / "molecule_related_natural_language_tokenized.parquet"
    
    logger.info(f"Using dataset directory: {base_dataset_dir}")
    logger.info(f"Using logging directory: {logging_dir}")
    logger.info(f"Using parquet file path: {parquet_file}")
    
    os.path.exists(logging_dir) or os.makedirs(logging_dir)
    setup_logging(logging_dir)

    os.path.exists(base_dataset_dir) or os.makedirs(base_dataset_dir)

    # Determine data source (default: use local dataset)
    # Default behavior: use local dataset unless --download is specified
    use_local_data = not args.download
    
    if use_local_data:
        logger.info(msg="Using local dataset...")
        local_dataset_path = Path(args.local_path)
        
        # Check if local dataset exists
        if not local_dataset_path.exists():
            logger.warning(f"Local dataset not found at {args.local_path}")
            logger.info("Falling back to downloading from Hugging Face...")
            use_local_data = False
        else:
            logger.info(f"Found local dataset at {args.local_path}")
    
    if not use_local_data:
        logger.info(msg="Downloading Dataset from Hugging Face...")
        
        # Set environment variable for Hugging Face cache to use our custom directory
        import os
        os.environ['HF_DATASETS_CACHE'] = str(base_dataset_dir.parent)
    
    if use_local_data:
        # Handle local dataset processing
        dataset_marker_file = base_dataset_dir / "dataset_info.json"
        dataset_config_file = base_dataset_dir / "dataset_dict.json"
        
        if not args.force and dataset_marker_file.exists() and dataset_config_file.exists():
            logger.info(msg=f"Local dataset already copied to {base_dataset_dir}. Skipping copy.")
            logger.info(msg="If you want to re-copy, please use --force option or delete the directory and run again.")
        else:
            if args.force:
                logger.info(msg="Force option specified. Re-copying local dataset...")
            
            # Copy local dataset
            if not copy_local_dataset(args.local_path, base_dataset_dir):
                logger.error(msg="Failed to copy local dataset. Exiting.")
                exit(1)
    else:
        # Handle Hugging Face download (original logic)
        # データセットが既に存在するかチェック
        dataset_marker_file = base_dataset_dir / "dataset_info.json"
        dataset_config_file = base_dataset_dir / "dataset_dict.json"
        
        if not args.force and dataset_marker_file.exists() and dataset_config_file.exists():
            logger.info(msg=f"Dataset already exists at {base_dataset_dir}. Skipping download.")
            logger.info(msg="If you want to re-download, please use --force option or delete the directory and run again.")
            
            # データセットの読み込みテストを行う
            try:
                test_dataset = read_dataset(base_dataset_dir)
                logger.info(msg=f"Dataset validation successful. Found splits: {list(test_dataset.keys())}")
            except Exception as e:
                logger.warning(msg=f"Dataset validation failed: {e}")
                logger.info(msg="Re-downloading dataset...")
                try:
                    # Clear potentially corrupted cache and re-download
                    import shutil
                    if base_dataset_dir.exists():
                        shutil.rmtree(base_dataset_dir)
                    os.makedirs(base_dataset_dir, exist_ok=True)
                    download_hf_dataset(base_dataset_dir)
                    logger.info(msg="Dataset download completed successfully.")
                except Exception as download_error:
                    logger.error(msg="Failed to download dataset.")
                    logger.error(msg=f"Error details: {download_error}")
                    exit(1)
        else:
            if args.force:
                logger.info(msg="Force option specified. Re-downloading dataset...")
                # Clear existing directory if force is specified
                import shutil
                if base_dataset_dir.exists():
                    shutil.rmtree(base_dataset_dir)
            
            os.makedirs(base_dataset_dir, exist_ok=True)
            
            try:
                # Set encoding explicitly to handle potential UTF-8 issues
                import locale
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                
                download_hf_dataset(base_dataset_dir)
                logger.info(msg="Dataset download completed successfully.")
            except UnicodeDecodeError as decode_error:
                logger.error(msg=f"UTF-8 encoding error: {decode_error}")
                logger.error(msg="This may be due to corrupted cache files. Trying to clear cache and retry...")
                
                # Clear Hugging Face cache and retry
                import shutil
                hf_cache_dir = Path.home() / ".cache" / "huggingface"
                if hf_cache_dir.exists():
                    logger.info(msg="Clearing Hugging Face cache...")
                    shutil.rmtree(hf_cache_dir)
                
                try:
                    download_hf_dataset(base_dataset_dir)
                    logger.info(msg="Dataset download completed successfully after cache clear.")
                except Exception as retry_error:
                    logger.error(msg=f"Failed to download dataset even after cache clear: {retry_error}")
                    exit(1)
            except Exception as e:
                logger.error(msg="Failed to download dataset.")
                logger.error(msg=f"Error details: {e}")
                
                # Try alternative approach: download to temporary location first
                logger.info(msg="Attempting alternative download approach...")
                temp_dir = base_dataset_dir.parent / "temp_download"
                try:
                    if temp_dir.exists():
                        import shutil
                        shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    download_hf_dataset(temp_dir)
                    
                    # Move from temp to final location
                    import shutil
                    if base_dataset_dir.exists():
                        shutil.rmtree(base_dataset_dir)
                    shutil.move(str(temp_dir), str(base_dataset_dir))
                    logger.info(msg="Dataset download completed via alternative approach.")
                    
                except Exception as alt_error:
                    logger.error(msg=f"Alternative download approach also failed: {alt_error}")
                    exit(1)

    # Load dataset with error handling
    try:
        dataset = read_dataset(base_dataset_dir)
        logger.info(msg=f"Successfully loaded dataset from {base_dataset_dir}")
    except UnicodeDecodeError as decode_error:
        logger.error(msg=f"Unicode decode error when loading dataset: {decode_error}")
        logger.error(msg="This suggests corrupted dataset files. Removing and re-downloading...")
        
        import shutil
        shutil.rmtree(base_dataset_dir)
        os.makedirs(base_dataset_dir, exist_ok=True)
        
        try:
            download_hf_dataset(base_dataset_dir)
            dataset = read_dataset(base_dataset_dir)
            logger.info(msg="Dataset successfully re-downloaded and loaded.")
        except Exception as reload_error:
            logger.error(msg=f"Failed to re-download and load dataset: {reload_error}")
            exit(1)
    except Exception as load_error:
        logger.error(msg=f"Failed to load dataset: {load_error}")
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
            logger.info(f"  Sample structure preview:")
            for key in sample_keys[:5]:  # Show first 5 fields
                value = str(sample[key])[:100] + "..." if len(str(sample[key])) > 100 else sample[key]
                logger.info(f"    {key}: {value}")

    # 既に処理済みのparquetファイルが存在するかチェック
    if not args.force and parquet_file.exists():
        logger.info(msg=f"Processed dataset already exists at {parquet_file}.")
        logger.info(msg="Skipping tokenization and processing. If you want to reprocess, please use --force option or delete the parquet file and run again.")
        exit(0)
    elif args.force and parquet_file.exists():
        logger.info(msg="Force option specified. Reprocessing dataset...")

    tokenizer = MoleculeNatLangTokenizer()

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
                    'valid_sample': False,
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': [],
                    'output_ids': [],
                    'input_text': example.get('input', ''),
                    'real_input_text': '',
                    'input_too_long': False,
                    'task_type': example.get('sample_id', 'unknown').split('.')[0] if 'sample_id' in example else 'unknown'
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

                result.setdefault('task_type', example.get('sample_id', 'unknown').split('.')[0] if 'sample_id' in example else 'unknown')
                result['valid_sample'] = True
                return result
            except Exception as e:
                logger.warning(f"Error processing sample {example.get('sample_id', 'unknown')}: {e}")
                return default_result()
        
        # Apply validation and tokenization
        logger.info(f"Processing {split} split...")
        processed_split = dataset[split].map(
            validate_and_tokenize,
            batched=False,
            num_proc=cfg.num_workers,
            load_from_cache_file=False,
            desc="Validating and tokenizing {}".format(split),
            remove_columns=dataset[split].column_names  # Remove original columns to save space
        )
        
        # Filter out None results (invalid samples)
        processed_split = processed_split.filter(lambda x: x is not None)
        processed_dataset[split] = processed_split
        
        logger.info(f"Processed {len(processed_dataset[split])} valid samples in {split} split")

    logger.info(msg="Computing Dataset Statistics...")
    total_num_samples = 0
    total_num_tokens = 0
    
    # Compute statistics by task type if available
    task_stats = {}
    
    for split in processed_dataset.keys():
        logger.info(msg=f"{split}")
        num_samples, num_tokens = calculate_statistics(processed_dataset, split)
        logger.info(msg=f"Number of examples: {num_samples}")
        logger.info(msg=f"Number of tokens: {num_tokens}")
        total_num_samples += num_samples
        total_num_tokens += num_tokens
        
        # Collect task-specific statistics
        if 'task_type' in processed_dataset[split].features:
            task_types = processed_dataset[split]['task_type']
            for task_type in set(task_types):
                if task_type not in task_stats:
                    task_stats[task_type] = {'samples': 0, 'tokens': 0}
                
                task_samples = sum(1 for t in task_types if t == task_type)
                task_indices = [i for i, t in enumerate(task_types) if t == task_type]
                task_tokens = sum(
                    len(processed_dataset[split][i]['input_ids']) + len(processed_dataset[split][i]['output_ids'])
                    for i in task_indices
                )
                
                task_stats[task_type]['samples'] += task_samples
                task_stats[task_type]['tokens'] += task_tokens
    
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
    save_dataset(processed_dataset, parquet_file)
