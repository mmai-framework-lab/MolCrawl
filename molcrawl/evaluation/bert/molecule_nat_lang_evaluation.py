#!/usr/bin/env python3
"""
BERT Molecule Natural Language Model - Evaluation Script

Accuracy verification of molecular-related natural language tasks using trained BERT models
Uses the same evaluation criteria and output format as the GPT-2 version.

Main evaluation methods:
1. Predicted probability by Masked Language Modeling (MLM)
2. Perplexity calculation (MLM based)
3. Analysis of array length/token length
4. Same visualization format as GPT-2 version
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM
from molcrawl.core.utils.environment_check import check_learning_source_dir

# add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from molcrawl.core.utils.evaluation_output import (  # noqa: E402
    setup_evaluation_logging,
)
from molcrawl.core.utils.model_evaluator import ModelEvaluator  # noqa: E402

# Molecule NL tokenizer
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer  # noqa: E402

# Log settingslatersetup_evaluation_loggingdo it with
logger = logging.getLogger(__name__)


class BERTMoleculeNLEvaluator(ModelEvaluator):
    """BERT model evaluation class using Molecule NL data"""

    def __init__(self, model_path, tokenizer_path="", device="cuda", max_length=512):
        """
        initialization

        Args:
            model_path (str): Path of trained BERT model
            tokenizer_path (str): unused (MoleculeNatLangTokenizer is initialized internally)
            device (str): Device used
            max_length (int): Maximum input length
        """
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer_path = "molecule_nat_lang_internal"  # dummy path
        self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # initialize tokenizer
        self.tokenizer = self._init_tokenizer()

        # set vocabulary size(Required before loading the model)
        self.vocab_size = getattr(self.tokenizer, "vocab_size", 32024)

        # initialize model
        self.model = self._init_model()

        # Get special token ID
        self.mask_token_id = self._get_mask_token_id()
        self.cls_token_id = self._get_cls_token_id()
        self.sep_token_id = self._get_sep_token_id()
        self.pad_token_id = self._get_pad_token_id()

    def _init_tokenizer(self):
        """Tokenizer initialization (abstract method implementation)"""
        logger.info("Loading MoleculeNatLangTokenizer")
        try:
            tokenizer = MoleculeNatLangTokenizer()
            logger.info(f"✅ MoleculeNatLangTokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Failed to load MoleculeNatLangTokenizer: {e}")
            raise

    def _init_model(self):
        """Model initialization (abstract method implementation)"""
        logger.info(f"Loading BERT model from {self.model_path}")
        return self._load_model()

    def _get_mask_token_id(self):
        """Get the ID of the MASK token"""
        # Try to get from MoleculeNatLangTokenizer
        if hasattr(self.tokenizer, "mask_token_id"):
            logger.info(f"Using MASK token ID: {self.tokenizer.mask_token_id}")
            return self.tokenizer.mask_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "mask_token_id"):
            logger.info(f"Using MASK token ID: {self.tokenizer.tokenizer.mask_token_id}")
            return self.tokenizer.tokenizer.mask_token_id
        else:
            # fallback
            logger.warning("MASK token not found, using ID 103 (BERT default)")
            return 103

    def _get_cls_token_id(self):
        """Get ID of CLS token"""
        if hasattr(self.tokenizer, "cls_token_id"):
            return self.tokenizer.cls_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "cls_token_id"):
            return self.tokenizer.tokenizer.cls_token_id
        else:
            logger.warning("CLS token not found, using ID 101 (BERT default)")
            return 101

    def _get_sep_token_id(self):
        """Get SEP token ID"""
        if hasattr(self.tokenizer, "sep_token_id"):
            return self.tokenizer.sep_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "sep_token_id"):
            return self.tokenizer.tokenizer.sep_token_id
        else:
            logger.warning("SEP token not found, using ID 102 (BERT default)")
            return 102

    def _get_pad_token_id(self):
        """Get PAD token ID"""
        if hasattr(self.tokenizer, "pad_token_id"):
            return self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "pad_token_id"):
            return self.tokenizer.tokenizer.pad_token_id
        else:
            logger.warning("PAD token not found, using ID 0")
            return 0

    def _load_model(self):
        """Loading a trained BERT model (safetensors compatible)"""
        try:
            logger.info(f"Loading trained BERT model from: {self.model_path}")

            # Load in Hugging Face transformers format
            config = BertConfig.from_pretrained(self.model_path)
            logger.info(f"Model config loaded: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

            # Check if it matches the tokenizer size
            if config.vocab_size != self.vocab_size:
                logger.warning(f"Vocab size mismatch: model={config.vocab_size}, tokenizer={self.vocab_size}")
                logger.info("Using model's original vocab size for compatibility")
                original_vocab_size = config.vocab_size
                self.vocab_size = original_vocab_size  # Adjust tokenizer size

            # Load trained model from safetensors file
            model = BertForMaskedLM.from_pretrained(
                self.model_path,
                config=config,
                local_files_only=True,  # only use local files
                use_safetensors=True,  # use safetensors format
                ignore_mismatched_sizes=False,  # Strictly check for size mismatches
            )

            logger.info("✅ Successfully loaded trained BERT model with safetensors")

        except Exception as e:
            logger.error(f"❌ Failed to load trained model: {e}")
            logger.info("🔄 Creating new untrained model as fallback")

            # fallback: new untrained model
            config = BertConfig(
                vocab_size=self.vocab_size,
                max_position_embeddings=self.max_length,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            model = BertForMaskedLM(config)
            logger.warning("⚠️  Using untrained model - results may not be meaningful")

        model.to(self.device)
        model.eval()

        # Display model statistics
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("📊 Model Statistics:")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Hidden size: {model.config.hidden_size}")
        logger.info(f"   - Number of layers: {model.config.num_hidden_layers}")
        logger.info(f"   - Attention heads: {model.config.num_attention_heads}")
        logger.info(f"   - Max sequence length: {model.config.max_position_embeddings}")

        return model

    def encode_text(self, text):
        """Encode text to token ID"""
        try:
            # Use MoleculeNatLangTokenizer
            if hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(text, max_length=self.max_length)
                return torch.tensor(tokens, dtype=torch.long)
            else:
                # For HuggingFace tokenizer format
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                return encoding["input_ids"][0]
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            return torch.tensor([self.cls_token_id, self.sep_token_id], dtype=torch.long)

    def encode_sequence(self, sequence):
        """
        Encode array to token ID (implementation of abstract method)

        Same processing as encode_text in molecule_nat_lang
        """
        return self.encode_text(sequence)

    def calculate_perplexity(self, text_or_tokens):
        """
        Calculate perplexity of text or token IDs (BERT MLM based)

        BERT uses Masked Language Modeling, so each token is masked in turn and
        Calculate the predicted probabilities and calculate the perplexity from the average.

        Args:
            text_or_tokens: text string or list/array of token IDs
        """
        with torch.no_grad():
            try:
                # Determine whether the input is token IDs or text
                if isinstance(text_or_tokens, (list, tuple)) or (
                    hasattr(text_or_tokens, "__iter__") and not isinstance(text_or_tokens, str)
                ):
                    # If already tokenized
                    if hasattr(text_or_tokens, "tolist"):
                        tokens = torch.tensor(text_or_tokens.tolist(), dtype=torch.long)
                    else:
                        tokens = torch.tensor(list(text_or_tokens), dtype=torch.long)
                else:
                    # encode if text
                    tokens = self.encode_text(text_or_tokens)

                if len(tokens) < 2:
                    logger.debug(f"Sequence too short for perplexity calculation: {len(tokens)} tokens")
                    return float("inf")

                # Add batch dimension and transfer to device
                tokens = tokens.unsqueeze(0).to(self.device)

                # Calculate predicted probability by masking each token in turn
                total_log_prob = 0.0
                num_predictions = 0

                for i in range(tokens.shape[1]):
                    # Skip special tokens
                    if tokens[0, i].item() in [
                        self.cls_token_id,
                        self.sep_token_id,
                        self.pad_token_id,
                    ]:
                        continue

                    # mask token
                    masked_tokens = tokens.clone()
                    original_token = masked_tokens[0, i].item()
                    masked_tokens[0, i] = self.mask_token_id

                    # prediction
                    outputs = self.model(masked_tokens)
                    logits = outputs.logits

                    # Get the predicted probability of mask position
                    masked_token_logits = logits[0, i, :]
                    probs = F.softmax(masked_token_logits, dim=-1)

                    # Predicted probability of original token
                    token_prob = probs[original_token].item()

                    # If the probability is 0, replace it with a smaller value
                    if token_prob <= 0:
                        token_prob = 1e-10

                    # Accumulate log probability
                    total_log_prob += math.log(token_prob)
                    num_predictions += 1

                if num_predictions == 0:
                    logger.debug("No valid tokens for perplexity calculation")
                    return float("inf")

                # Calculate perplexity from average log probability
                avg_log_prob = total_log_prob / num_predictions
                perplexity = math.exp(-avg_log_prob)

                return perplexity

            except Exception as e:
                logger.debug(f"Error in perplexity calculation: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                return float("inf")

    def evaluate_dataset(
        self,
        dataset_path,
        output_dir="./reports/bert_molecule_nat_lang_evaluation",
        sample_size=None,
    ):
        """
        Evaluation of the entire Molecule NL dataset（GPT-2(same format as the edition)

        Args:
            dataset_path (str): dataset path
            output_dir (str): Output directory
            sample_size (int): Sample size (None=all data)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🔬 Starting BERT Molecule NL Model Evaluation")
        logger.info("=" * 60)

        # Load dataset
        logger.info("📚 Loading Molecule NL dataset...")
        try:
            dataset = load_from_disk(dataset_path)

            # For DatasetDict, select appropriate split
            if hasattr(dataset, "keys"):
                # For DatasetDict, use test or validation preferentially
                if "test" in dataset:
                    dataset_split = dataset["test"]
                    logger.info("Using 'test' split from dataset")
                elif "validation" in dataset:
                    dataset_split = dataset["validation"]
                    logger.info("Using 'validation' split from dataset")
                elif "train" in dataset:
                    dataset_split = dataset["train"]
                    logger.info("Using 'train' split from dataset")
                else:
                    # use first split
                    split_name = list(dataset.keys())[0]
                    dataset_split = dataset[split_name]
                    logger.info(f"Using '{split_name}' split from dataset")
            else:
                # For a single Dataset
                dataset_split = dataset

            # Convert HuggingFace Dataset to pandas DataFrame
            df = dataset_split.to_pandas()
            logger.info(f"✅ Dataset loaded successfully: {len(df)} samples")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"📊 Using sample of {len(df)} texts")
        else:
            logger.info(f"📊 Evaluating all {len(df)} texts")

        # dataset statistics
        logger.info(f"   - Total samples: {len(df)}")
        logger.info(f"   - Available columns: {list(df.columns)}")
        if "input_ids" in df.columns:
            # Length calculation when input_ids is list type
            try:
                avg_length = df["input_ids"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0).mean()
                logger.info(f"   - Average sequence length: {avg_length:.1f}")
            except Exception as e:
                logger.warning(f"   - Could not calculate average sequence length: {e}")

        results = []
        perplexities = []
        processing_errors = 0

        logger.info("🧬 Processing texts...")
        logger.info("   Note: Using 'input_text' field and re-tokenizing with BERT tokenizer")

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                avg_perplexity = np.mean(perplexities) if perplexities else float("inf")
                logger.info(f"   Progress: {idx}/{len(df)} texts processed, avg perplexity: {avg_perplexity:.2f}")

            try:
                # Re-tokenize with BERT tokenizer using input_text
                # (Do not use the existing input_ids as they are for GPT-2)
                if "input_text" in row and row["input_text"]:
                    text = row["input_text"]

                    # Debugging: Check with first sample
                    if idx == 0:
                        logger.info("First sample analysis:")
                        logger.info(f"  input_text: {text[:100]}...")
                        logger.info(f"  Re-tokenizing with BERT tokenizer (vocab_size={self.vocab_size})")

                    # Encode text with BERT tokenizer
                    # Use encode_text method (MoleculeNatLangTokenizer is used internally)
                    try:
                        tokens = self.encode_text(text)

                        if idx == 0:
                            logger.info(f"  BERT tokenized length: {len(tokens)}")
                            logger.info(
                                f"  BERT token sample: {tokens[:20].tolist() if hasattr(tokens, 'tolist') else tokens[:20]}"
                            )
                            if hasattr(tokens, "__iter__"):
                                token_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
                                if token_list:
                                    logger.info(f"  BERT token range: {min(token_list)} - {max(token_list)}")

                        if len(tokens) == 0:
                            logger.warning(f"Sample {idx}: Empty tokens after encoding")
                            continue

                    except Exception as e:
                        logger.warning(f"Sample {idx}: Failed to encode text: {e}")
                        continue

                    # Perplexity calculation
                    if idx < 5 or idx % 500 == 0:  # log the first 5 and every 500
                        logger.info(f"Sample {idx}: Calculating perplexity for {len(tokens)} tokens...")

                    perplexity = self.calculate_perplexity(tokens)

                    if idx < 5 or idx % 500 == 0:
                        logger.info(f"Sample {idx}: Perplexity = {perplexity:.4f}")

                    # for text preview
                    text_preview = text[:100]

                else:
                    logger.warning(f"Sample {idx}: No input_text found in row")
                    continue

                # record the result（GPT-2(same format as the version)
                result = {
                    "index": idx,
                    "text_length": len(text_preview),
                    "token_length": len(tokens),
                    "perplexity": perplexity,
                    "log_perplexity": (math.log(perplexity) if perplexity > 0 and perplexity != float("inf") else float("inf")),
                    "text_preview": text_preview,
                }

                results.append(result)
                if perplexity != float("inf"):
                    perplexities.append(perplexity)

            except Exception as e:
                processing_errors += 1
                logger.warning(f"Error processing sample {idx}: {e}")
                import traceback

                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        logger.info("✅ Processing completed!")
        logger.info(f"   - Successfully processed: {len(results)} texts")
        logger.info(f"   - Processing errors: {processing_errors}")

        # What to do if the result is empty
        if not results:
            logger.error("❌ No samples were successfully processed!")
            logger.error("   Please check:")
            logger.error("   1. Dataset format (should have 'input_ids' or 'text' field)")
            logger.error("   2. Tokenizer compatibility")
            logger.error("   3. Model checkpoint path")

            # Create an empty DataFrame (columns are defined)
            results_df = pd.DataFrame(
                columns=[
                    "index",
                    "text_length",
                    "token_length",
                    "perplexity",
                    "log_perplexity",
                    "text_preview",
                ]
            )

            # return empty metrics
            metrics = {
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": 0.0,
                "total_samples": 0,
                "valid_samples": 0,
                "processing_errors": processing_errors,
            }

            return metrics, results_df

        # Save resultsand analysis (GPT-2(same format as the version)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv"), index=False)

        logger.info(f"📊 Results DataFrame shape: {results_df.shape}")
        logger.info(f"   Columns: {list(results_df.columns)}")
        if len(results_df) > 0:
            logger.info(f"   Sample data:\n{results_df.head()}")

        # Calculating performance indicators
        metrics = self._calculate_metrics(perplexities, results_df)

        # Save results
        with open(os.path.join(output_dir, "molecule_nat_lang_evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Visualization (GPT-2(uses the same class as version)
        self._create_visualizations_with_separate_class(results_df, output_dir)

        # generate report
        self._generate_report(metrics, results_df, output_dir)

        logger.info(f"📁 Results saved to: {output_dir}")
        return metrics, results_df

    def _create_visualizations_with_separate_class(self, results_df, output_dir):
        """Generate visualization using a separate visualization class (same as GPT-2 version)"""
        try:
            # Generate path for CSV result file
            csv_file = os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv")

            # import visualization class（GPT-2(uses the same class as version)
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            # initialize the visualizer
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # generate visualization
            logger.info("📊 Generating visualizations...")
            visualizer.generate_all_visualizations()
            logger.info("✅ Visualization completed successfully!")

        except ImportError as e:
            logger.warning(f"Could not import visualization module: {e}. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _calculate_metrics(self, perplexities, results_df):
        """Performance index calculation (same format as GPT-2 version)"""
        if not perplexities:
            return {
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": 0.0,
                "min_perplexity": float("inf"),
                "max_perplexity": float("inf"),
                "total_samples": len(results_df),
                "valid_samples": 0,
                "invalid_samples": len(results_df),
            }

        valid_perplexities = [p for p in perplexities if p != float("inf")]

        metrics = {
            "mean_perplexity": float(np.mean(valid_perplexities)),
            "median_perplexity": float(np.median(valid_perplexities)),
            "std_perplexity": float(np.std(valid_perplexities)),
            "min_perplexity": float(np.min(valid_perplexities)),
            "max_perplexity": float(np.max(valid_perplexities)),
            "total_samples": len(results_df),
            "valid_samples": len(valid_perplexities),
            "invalid_samples": len(results_df) - len(valid_perplexities),
        }

        logger.info("📊 Evaluation Metrics:")
        logger.info(f"   - Mean Perplexity: {metrics['mean_perplexity']:.3f}")
        logger.info(f"   - Median Perplexity: {metrics['median_perplexity']:.3f}")
        logger.info(f"   - Std Perplexity: {metrics['std_perplexity']:.3f}")
        logger.info(f"   - Min/Max Perplexity: {metrics['min_perplexity']:.3f} / {metrics['max_perplexity']:.3f}")
        logger.info(f"   - Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")

        return metrics

    def _generate_report(self, metrics, results_df, output_dir):
        """Text report generation (same format as GPT-2 version)"""
        report_path = os.path.join(output_dir, "molecule_nat_lang_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BERT Molecule Natural Language Model - Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("Model Information:\n")
            f.write(f"  Model Path: {self.model_path}\n")
            f.write(f"  Vocabulary Size: {self.vocab_size}\n")
            f.write(f"  Max Sequence Length: {self.max_length}\n")
            f.write(f"  Device: {self.device}\n\n")

            f.write("Overall Performance Metrics:\n")
            f.write(f"  Mean Perplexity: {metrics['mean_perplexity']:.3f}\n")
            f.write(f"  Median Perplexity: {metrics['median_perplexity']:.3f}\n")
            f.write(f"  Std Perplexity: {metrics['std_perplexity']:.3f}\n")
            f.write(f"  Min Perplexity: {metrics['min_perplexity']:.3f}\n")
            f.write(f"  Max Perplexity: {metrics['max_perplexity']:.3f}\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"  Total Samples: {metrics['total_samples']}\n")
            f.write(f"  Valid Samples: {metrics['valid_samples']}\n")
            f.write(f"  Invalid Samples: {metrics['invalid_samples']}\n\n")

            if len(results_df) > 0:
                f.write("Token Length Statistics:\n")
                f.write(f"  Mean Token Length: {results_df['token_length'].mean():.1f}\n")
                f.write(f"  Median Token Length: {results_df['token_length'].median():.1f}\n")
                f.write(f"  Min/Max Token Length: {results_df['token_length'].min()} / {results_df['token_length'].max()}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"📝 Report saved to: {report_path}")


def main():
    """Main evaluation flow"""
    parser = argparse.ArgumentParser(description="Evaluate BERT Molecule NL model on dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained BERT model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to Molecule NL dataset (parquet or HuggingFace format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing results (skip evaluation)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory containing existing results for visualization-only mode",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        # Use LEARNING_SOURCE_DIR environment variable
        learning_source_dir = check_learning_source_dir()
        args.output_dir = os.path.join(
            learning_source_dir,
            "molecule_nat_lang",
            "report",
            f"bert_molecule_nat_lang_ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # Convert the output directory to Path type and create a directory
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Log settings（Pathpass the type)
    setup_evaluation_logging(output_dir_path, "bert_molecule_nat_lang_evaluation")
    logger.info("Starting BERT Molecule NL model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Visualization-only mode
    if args.visualize_only:
        logger.info("📊 Visualization-only mode enabled")

        if not args.result_dir:
            logger.error("--result-dir is required in visualization-only mode")
            return

        if not os.path.exists(args.result_dir):
            logger.error(f"Result directory not found: {args.result_dir}")
            return

        # Check existing result file
        csv_file = os.path.join(args.result_dir, "molecule_nat_lang_detailed_results.csv")
        json_file = os.path.join(args.result_dir, "molecule_nat_lang_evaluation_results.json")

        if not os.path.exists(csv_file):
            logger.error(f"CSV results file not found: {csv_file}")
            return

        # Generate visualization only
        try:
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=args.result_dir)

            logger.info("📊 Generating visualizations from existing results...")
            visualizer.generate_all_visualizations()
            logger.info("✅ Visualization completed successfully!")

            # show metrics
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                    logger.info(f"📊 Mean Perplexity: {metrics.get('mean_perplexity', 'N/A'):.3f}")
                    logger.info(
                        f"📈 Valid samples: {metrics.get('valid_samples', 'N/A')}/{metrics.get('total_samples', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"Error in visualization-only mode: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return

    # Normal evaluation mode
    try:
        # Initialize the evaluator
        evaluator = BERTMoleculeNLEvaluator(
            model_path=args.model_path,
            tokenizer_path="",  # not used
            device=args.device,
            max_length=args.max_length,
        )

        # Dataset evaluation
        metrics, results_df = evaluator.evaluate_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )

        logger.info("Evaluation completed successfully!")
        logger.info(f"Mean Perplexity: {metrics['mean_perplexity']:.3f}")
        logger.info(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
