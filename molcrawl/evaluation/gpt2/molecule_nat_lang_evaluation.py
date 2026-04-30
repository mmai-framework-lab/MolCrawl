#!/usr/bin/env python3
"""
Molecule Natural Language model evaluation script

This script uses the trained GPT-2 molecule_nat_lang model to
Verifying the performance of molecular-related natural language tasks.
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk

# add project root

from molcrawl.core.utils.environment_check import check_learning_source_dir

from molcrawl.gpt2.model import GPT, GPTConfig

from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
from molcrawl.core.utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from molcrawl.core.utils.model_evaluator import ModelEvaluator

# Log settingslatersetup_evaluation_loggingdo it with
logger = logging.getLogger(__name__)


class GPT2MoleculeNLEvaluator(ModelEvaluator):
    """Model evaluation class using Molecule NL data"""

    def __init__(self, model_path, tokenizer_path="", device="cuda", max_length=1024):
        """
        initialization

        Args:
            model_path (str): path of the trained model
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

        # Initialize tokenizer and model
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()

        # set vocabulary size
        self.vocab_size = getattr(self.tokenizer, "vocab_size", 32024)

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
        logger.info(f"Loading GPT2 model from {self.model_path}")
        return self._load_gpt2_model()

    def _load_gpt2_model(self):
        """Loading the trained GPT2 model"""
        try:
            # Read checkpoint
            checkpoint = torch.load(self.model_path, map_location="cpu")
            logger.info("✅ Checkpoint loaded successfully")

            # Get model settings
            if "config" in checkpoint:
                # New format: settings are saved
                saved_config = checkpoint["config"]
                logger.info("📝 Using saved model configuration")

                # Extract only parameters available in GPTConfig
                valid_params = {
                    "block_size",
                    "vocab_size",
                    "n_layer",
                    "n_head",
                    "n_embd",
                    "dropout",
                    "bias",
                }
                model_args = {k: v for k, v in saved_config.items() if k in valid_params}

                # If the vocabulary size is not in the settings, infer it from the weights
                if "vocab_size" not in model_args:
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                        if "transformer.wte.weight" in state_dict:
                            model_args["vocab_size"] = state_dict["transformer.wte.weight"].shape[0]
                            logger.info(f"📊 Detected vocab_size from weights: {model_args['vocab_size']}")

                logger.info(f"   - Filtered config keys: {list(model_args.keys())}")
                if "vocab_size" in model_args:
                    logger.info(f"   - Model vocab_size: {model_args['vocab_size']}")
                if self.tokenizer:
                    logger.info(f"   - Tokenizer vocab_size: {self.tokenizer.vocab_size}")
            else:
                # Old style: infer vocabulary size from checkpoints
                logger.warning("⚠️  No saved config found, using checkpoint weights for config")

                # Estimate vocabulary size from checkpoints
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif isinstance(checkpoint, dict) and "transformer.wte.weight" in checkpoint:
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Get vocabulary size from embedding layer
                vocab_size = 32024  # Default value (verified from checkpoint)
                if "transformer.wte.weight" in state_dict:
                    vocab_size = state_dict["transformer.wte.weight"].shape[0]
                    logger.info(f"📊 Detected vocab_size from checkpoint: {vocab_size}")
                elif "wte.weight" in state_dict:
                    vocab_size = state_dict["wte.weight"].shape[0]
                    logger.info(f"📊 Detected vocab_size from checkpoint: {vocab_size}")
                else:
                    logger.info(f"📊 Using default vocab_size: {vocab_size}")

                model_args = {
                    "block_size": 1024,
                    "vocab_size": vocab_size,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                    "dropout": 0.0,
                    "bias": True,
                }

            # Create a model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

            # Load weights
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            # Display model statistics
            total_params = sum(p.numel() for p in model.parameters())
            logger.info("📊 Model Statistics:")
            logger.info(f"   - Total parameters: {total_params:,}")
            logger.info(f"   - Vocabulary size: {model_args.get('vocab_size', 'unknown')}")
            logger.info(f"   - Block size: {model_args.get('block_size', 'unknown')}")
            logger.info(f"   - Number of layers: {model_args.get('n_layer', 'unknown')}")
            logger.info(f"   - Embedding dimension: {model_args.get('n_embd', 'unknown')}")

            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode sequence (implementation of abstract method)"""
        return self.encode_text(sequence)

    def encode_text(self, text, add_special_tokens=True):
        """Encode text to token ID"""
        try:
            # Use MoleculeNatLangTokenizerand encode
            tokenized_result = self.tokenizer.tokenize_text(text)
            tokens = tokenized_result["input_ids"]

            # adjust to maximum length
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
                logger.debug(f"Text truncated to {len(tokens)} tokens")

            if not tokens:
                logger.warning(f"Empty tokenization for text: {text[:50]}...")
                # Use padding token ID
                tokens = [self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0]

            return torch.tensor(tokens, dtype=torch.long)

        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            # fallback：Basic tokenization
            try:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"][0]
                return tokens
            except Exception as e2:
                logger.error(f"Fallback tokenization also failed: {e2}")
                return torch.tensor([0], dtype=torch.long)

    def calculate_perplexity(self, text_or_tokens):
        """Calculate perplexity of text or token IDs

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
                    logger.info(f"Sequence too short for perplexity calculation: {len(tokens)} tokens")
                    return float("inf")

                # Add batch dimension and transfer to device
                tokens = tokens.unsqueeze(0).to(self.device)

                # Shift inputs and goals just like during training
                # x = tokens[:, :-1] (excluding the last token)
                # y = tokens[:, 1:] (excluding the first token)
                x = tokens[:, :-1]
                y = tokens[:, 1:]

                # Model prediction
                logits, loss = self.model(x, targets=y)

                if loss is None:
                    logger.info("Model returned None for loss")
                    return float("inf")

                # Perplexity is an index of loss
                perplexity = torch.exp(loss).item()

                return perplexity

            except Exception as e:
                logger.info(f"Error in perplexity calculation: {e}")
                import traceback

                logger.info(f"Traceback: {traceback.format_exc()}")
                return float("inf")

    def generate_text(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
        """Text generation"""
        self.model.eval()
        with torch.no_grad():
            tokens = self.encode_text(prompt)
            tokens = tokens.unsqueeze(0).to(self.device)

            for _ in range(max_new_tokens):
                if tokens.shape[1] >= self.max_length:
                    break

                # prediction
                logits, _ = self.model(tokens)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, top_k_indices, top_k_logits)

                # sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # add token
                tokens = torch.cat([tokens, next_token], dim=1)

            # decode
            generated_tokens = tokens[0].cpu().numpy().tolist()
            try:
                generated_text = self.tokenizer.decode(generated_tokens)
            except Exception as e:
                logger.warning(f"Decode failed in text generation: {e}")
                generated_text = f"[GENERATED TOKENS: {generated_tokens[:20]}...]"

            return generated_text

    def evaluate_dataset(
        self,
        dataset_path,
        output_dir="./reports/molecule_nat_lang_evaluation",
        sample_size=None,
    ):
        """
        Evaluation of the entire Molecule NL dataset

        Args:
            dataset_path (str): dataset path
            output_dir (str): Output directory
            sample_size (int): Sample size (None=all data)
        """
        print("DEBUG: evaluate_dataset method called!", flush=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🔬 Starting Molecule NL Model Evaluation")
        print("DEBUG: After first logger.info", flush=True)
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

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                avg_perplexity = np.mean(perplexities) if perplexities else float("inf")
                logger.info(f"   Progress: {idx}/{len(df)} texts processed, avg perplexity: {avg_perplexity:.2f}")

            try:
                # use input_ids directly (as they are already tokenized)
                if "input_ids" in row:
                    input_ids = row["input_ids"]

                    # Debugging: Check the type and contents of input_ids
                    if idx == 0:  # Check details with first sample
                        logger.info("First sample analysis:")
                        logger.info(f"  input_ids type: {type(input_ids)}")
                        logger.info(f"  input_ids length: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")
                        if hasattr(input_ids, "__iter__"):
                            sample_ids = list(input_ids)[:10] if len(input_ids) > 10 else list(input_ids)
                            logger.info(f"  input_ids sample: {sample_ids}")
                            logger.info(f"  Min/Max IDs: {min(input_ids)}/{max(input_ids)}")

                    # Supports numpy arrays, lists, and tuples
                    if hasattr(input_ids, "__len__") and len(input_ids) > 0:
                        # Perplexity calculation (pass input_ids directly)
                        logger.info(f"Sample {idx}: Calculating perplexity for {len(input_ids)} tokens...")
                        perplexity = self.calculate_perplexity(input_ids)
                        logger.info(f"Sample {idx}: Perplexity = {perplexity}")

                        # For text preview (display only)
                        text_preview = row.get("input_text", f"[{len(input_ids)} tokens]")[:100]
                    else:
                        logger.warning(f"Sample {idx}: Empty or invalid input_ids")
                        continue
                else:
                    logger.warning(f"Sample {idx}: No input_ids found in row")
                    continue

                # record the result
                result = {
                    "index": idx,
                    "text_length": len(text_preview),
                    "token_length": len(input_ids),
                    "perplexity": perplexity,
                    "log_perplexity": math.log(perplexity) if perplexity > 0 and perplexity != float("inf") else float("inf"),
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

        # Save resultsand analysis
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

        # Visualization (processed in a separate class)
        self._create_visualizations_with_separate_class(results_df, output_dir)

        # generate report
        self._generate_report(metrics, results_df, output_dir)

        logger.info(f"📁 Results saved to: {output_dir}")
        return metrics, results_df

    def _create_visualizations_with_separate_class(self, results_df, output_dir):
        """Generate visualizations using separate visualization classes"""
        try:
            # Generate path for CSV result file
            csv_file = os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv")

            # import visualization class
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            # initialize the visualizer
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # generate all visualizations
            visualizer.generate_all_visualizations()

            # Also generate HTML report
            visualizer.create_html_report()

            logger.info("✅ Visualizations created using separate visualization class")

        except Exception as e:
            logger.warning(f"⚠️  Visualization generation failed: {e}")
            logger.info("📊 Falling back to basic visualization")
            # fallback：Perform basic visualization only
            self._create_basic_visualization(results_df, output_dir)
        """Generate visualizations using separate visualization classes"""
        try:
            # Generate path for CSV result file
            csv_file = os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv")

            # import visualization class
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            # initialize the visualizer
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # generate all visualizations
            visualizer.generate_all_visualizations()

            # Also generate HTML report
            visualizer.create_html_report()

            logger.info("✅ Visualizations created using separate visualization class")

        except Exception as e:
            logger.warning(f"⚠️  Visualization generation failed: {e}")
            logger.info("📊 Falling back to basic visualization")
            # fallback：Perform basic visualization only
            self._create_basic_visualization(results_df, output_dir)

    def _create_basic_visualization(self, results_df, output_dir):
        """⚠️ DEPRECATED: Please use molecule_nat_lang_visualization.py for visualization"""
        logger.warning("⚠️  Inline visualization is deprecated.")
        logger.info("Please use: python scripts/evaluation/gpt2/molecule_nat_lang_visualization.py --result-dir <output_dir>")
        logger.info("Skipping inline visualization.")

    def _calculate_metrics(self, perplexities, results_df):
        """Calculating performance indicators"""
        if not perplexities:
            logger.warning("No valid perplexities found")
            return {
                "total_samples": len(results_df),
                "valid_samples": 0,
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": float("inf"),
                "min_perplexity": float("inf"),
                "max_perplexity": float("inf"),
            }

        perplexities = np.array(perplexities)

        metrics = {
            "total_samples": len(results_df),
            "valid_samples": len(perplexities),
            "mean_perplexity": float(np.mean(perplexities)),
            "median_perplexity": float(np.median(perplexities)),
            "std_perplexity": float(np.std(perplexities)),
            "min_perplexity": float(np.min(perplexities)),
            "max_perplexity": float(np.max(perplexities)),
            "percentile_25": float(np.percentile(perplexities, 25)),
            "percentile_75": float(np.percentile(perplexities, 75)),
            "mean_text_length": float(results_df["text_length"].mean()) if "text_length" in results_df.columns else 0.0,
            "mean_token_length": float(results_df["token_length"].mean()) if "token_length" in results_df.columns else 0.0,
        }

        return metrics

    def _generate_report(self, metrics, results_df, output_dir):
        """Generate evaluation report"""
        report_path = os.path.join(output_dir, "molecule_nat_lang_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("Molecule Natural Language Model - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"🕐 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("🧬 Model Type: GPT-2 for Molecule Natural Language\n")
            f.write("📊 Evaluation Method: Perplexity-based language modeling assessment\n\n")

            f.write("📈 Dataset Summary:\n")
            f.write(f"   • Total samples evaluated: {metrics['total_samples']}\n")
            f.write(f"   • Valid samples: {metrics['valid_samples']}\n")
            f.write(f"   • Processing success rate: {metrics['valid_samples'] / metrics['total_samples']:.1%}\n\n")

            f.write("🎯 Performance Metrics:\n")
            f.write(f"   • Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Median Perplexity: {metrics.get('median_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Std Perplexity: {metrics.get('std_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Min Perplexity: {metrics.get('min_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Max Perplexity: {metrics.get('max_perplexity', float('inf')):.3f}\n\n")

            f.write("📊 Text Statistics:\n")
            f.write(f"   • Mean Text Length: {metrics.get('mean_text_length', 0):.1f} characters\n")
            f.write(f"   • Mean Token Length: {metrics.get('mean_token_length', 0):.1f} tokens\n")
            f.write(f"   • 25th Percentile Perplexity: {metrics.get('percentile_25', float('inf')):.3f}\n")
            f.write(f"   • 75th Percentile Perplexity: {metrics.get('percentile_75', float('inf')):.3f}\n\n")

            # performance interpretation
            mean_ppl = metrics.get("mean_perplexity", float("inf"))
            if mean_ppl < 10:
                performance_interpretation = "🟢 Excellent language modeling performance"
            elif mean_ppl < 50:
                performance_interpretation = "🟡 Good language modeling performance"
            elif mean_ppl < 200:
                performance_interpretation = "🟠 Moderate language modeling performance"
            else:
                performance_interpretation = "🔴 Poor language modeling performance"

            f.write(f"📊 Overall Assessment: {performance_interpretation}\n")
            f.write(f"   Mean Perplexity: {mean_ppl:.3f}\n\n")

            f.write("💡 Key Insights:\n")
            f.write("   • Lower perplexity indicates better language modeling capability\n")
            f.write("   • This model specializes in molecule-related natural language\n")
            f.write("   • Perplexity reflects the model's uncertainty in predicting next tokens\n")
            f.write("   • Results should be compared with domain-specific baselines\n")


def main():
    parser = argparse.ArgumentParser(description="Molecule NL model evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt2-output/molecule_nat_lang-small/ckpt.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to test dataset directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for testing (default: use all data)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Image regeneration option
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing results (skip evaluation)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory containing existing evaluation results (CSV files) for visualization",
    )

    args = parser.parse_args()

    # Setting LEARNING_SOURCE_DIR
    learning_source_dir = check_learning_source_dir()

    # Image only generation mode
    if args.visualize_only:
        if args.result_dir is None:
            print("Error: --result-dir must be specified when using --visualize-only")
            sys.exit(1)

        if not os.path.exists(args.result_dir):
            print(f"Error: Result directory not found: {args.result_dir}")
            sys.exit(1)

        # Log settings
        logger = setup_evaluation_logging(Path(args.result_dir), "molecule_nat_lang_visualization")

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        logger.info("🎨 Visualization-only mode")
        logger.info(f"📁 Loading results from: {args.result_dir}")

        try:
            # Read CSV file
            csv_file = os.path.join(args.result_dir, "molecule_nat_lang_detailed_results.csv")
            if not os.path.exists(csv_file):
                logger.error(f"❌ CSV file not found: {csv_file}")
                sys.exit(1)

            results_df = pd.read_csv(csv_file)
            logger.info(f"✅ Loaded {len(results_df)} results from CSV")

            # Load metrics JSON (if available)
            json_file = os.path.join(args.result_dir, "molecule_nat_lang_evaluation_results.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                logger.info("✅ Loaded metrics from JSON")
            else:
                # Recalculate from CSV
                logger.info("ℹ️  Metrics JSON not found, recalculating from CSV...")
                valid_perplexities = results_df[results_df["perplexity"] != float("inf")]["perplexity"].tolist()
                metrics = {
                    "total_samples": len(results_df),
                    "valid_samples": len(valid_perplexities),
                    "mean_perplexity": float(np.mean(valid_perplexities)) if valid_perplexities else float("inf"),
                    "median_perplexity": float(np.median(valid_perplexities)) if valid_perplexities else float("inf"),
                    "std_perplexity": float(np.std(valid_perplexities)) if valid_perplexities else float("inf"),
                    "min_perplexity": float(np.min(valid_perplexities)) if valid_perplexities else float("inf"),
                    "max_perplexity": float(np.max(valid_perplexities)) if valid_perplexities else float("inf"),
                }

            # Use visualization generation class
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=args.result_dir)

            # generate all visualizations
            logger.info("🎨 Generating visualizations...")
            visualizer.generate_all_visualizations()

            logger.info("✅ Visualization completed successfully!")
            logger.info(f"📊 Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}")
            logger.info(f"📈 Valid samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")

        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

        return

    # Normal evaluation mode
    # Set default path
    if args.dataset_path is None:
        args.dataset_path = f"{learning_source_dir}/molecule_nat_lang/training_ready_hf_dataset/test"

    # Check the existence of dataset path
    if not os.path.exists(args.dataset_path):
        print(f"❌ ERROR: Dataset path does not exist: {args.dataset_path}")
        print(f"Expected structure: {learning_source_dir}/molecule_nat_lang/training_ready_hf_dataset/test")
        print("")
        print("Please verify that:")
        print(f"1. LEARNING_SOURCE_DIR='{learning_source_dir}' is correct")
        print("2. The molecule_nat_lang dataset has been processed")
        sys.exit(1)

    # Automatically generate output directory or use specified one
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "molecule_nat_lang", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Log settings
    logger = setup_evaluation_logging(Path(args.output_dir), "molecule_nat_lang_evaluation")

    # Set log level (debug mode)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    logger.info("Starting Molecule NL model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Initialize the evaluator(Tokenizer pass not required)
        evaluator = GPT2MoleculeNLEvaluator(
            model_path=args.model_path,
            tokenizer_path="",  # MoleculeNatLangTokenizer is initialized internally
            device=args.device,
            max_length=args.max_length,
        )

        # Run evaluation
        metrics, results_df = evaluator.evaluate_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )

        # Display results
        logger.info("Evaluation completed successfully!")
        logger.info(f"Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}")
        logger.info(f"Valid samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
