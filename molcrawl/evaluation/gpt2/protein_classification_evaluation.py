#!/usr/bin/env python3
"""
Protein Sequence GPT2 Classification Evaluation

This script evaluates a trained GPT2 protein sequence model using classification metrics
similar to the genome_sequence ClinVar evaluation approach.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add src and gpt2 to path for imports
from molcrawl.protein_sequence.utils.bert_tokenizer import EsmSequenceTokenizer
from molcrawl.core.utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from molcrawl.core.utils.model_evaluator import ModelEvaluator

# Log settingslatersetup_evaluation_loggingdo it with
logger = logging.getLogger(__name__)


class ProteinClassificationEvaluator(ModelEvaluator):
    """Protein sequence GPT2 model classification evaluator"""

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize the evaluator

        Args:
            model_path: Path to trained GPT2 model
            tokenizer_path: Path to tokenizer (None for EsmSequenceTokenizer)
        """
        # Initialize parent class
        super().__init__(model_path, tokenizer_path)

        # Subclass-specific initialization
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.model.to(self.device)
        self.model.eval()

    def _init_tokenizer(self):
        """Tokenizer initialization (abstract method implementation)"""
        # protein_sequence uses EsmSequenceTokenizer (no file required)
        logger.info("Using EsmSequenceTokenizer for protein_sequence")
        return EsmSequenceTokenizer(vocab_size=33)

    def _init_model(self):
        """Model initialization (abstract method implementation)"""
        logger.info(f"Loading model from {self.model_path}")
        return self._load_model()

    def _load_model(self):
        """Load the GPT2 model"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if "model" in checkpoint:
                model_state = checkpoint["model"]
                config = checkpoint.get("config", None)
            else:
                model_state = checkpoint
                config = None

            # Import GPT2 model architecture
            from molcrawl.gpt2.model import GPT, GPTConfig

            # Create model with appropriate config
            if config:
                # Filter out non-GPTConfig parameters
                valid_config_keys = [
                    "block_size",
                    "vocab_size",
                    "n_layer",
                    "n_head",
                    "n_embd",
                    "dropout",
                    "bias",
                ]
                filtered_config = {k: v for k, v in config.items() if k in valid_config_keys}

                # Override vocab_size from actual checkpoint if different
                if "transformer.wte.weight" in model_state:
                    actual_vocab_size = model_state["transformer.wte.weight"].shape[0]
                    filtered_config["vocab_size"] = actual_vocab_size

                model_config = GPTConfig(**filtered_config)
            else:
                # Default config for protein sequence
                # Try to infer vocab_size from checkpoint
                vocab_size = 33  # ESM tokenizer vocab size (default)
                if "transformer.wte.weight" in model_state:
                    vocab_size = model_state["transformer.wte.weight"].shape[0]

                model_config = GPTConfig(
                    block_size=1024,
                    vocab_size=vocab_size,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                    dropout=0.0,
                    bias=False,
                )

            model = GPT(model_config)
            model.load_state_dict(model_state, strict=False)

            logger.info(f"Model loaded successfully. Config: {model_config}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encode_sequence(self, sequence: str, **kwargs) -> List[int]:
        """
        Encode array to token ID (implementation of abstract method)

        Args:
            sequence: protein sequence
            Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(sequence)

    def calculate_fitness_score(self, sequence: str, variant_pos: int, ref_aa: str, alt_aa: str) -> float:
        """
        Calculate fitness score for a protein variant using GPT2 likelihood

        Args:
            sequence: Protein sequence
            variant_pos: Position of variant (0-indexed)
            ref_aa: Reference amino acid
            alt_aa: Alternative amino acid

        Returns:
            Fitness score (positive = beneficial, negative = deleterious)
        """
        try:
            # Create reference and variant sequences
            ref_sequence = sequence
            var_sequence = sequence[:variant_pos] + alt_aa + sequence[variant_pos + 1 :]

            # Tokenize sequences
            ref_tokens = self.tokenizer.encode(ref_sequence)
            var_tokens = self.tokenizer.encode(var_sequence)

            # Calculate log probabilities
            ref_logprob = self._calculate_sequence_logprob(ref_tokens)
            var_logprob = self._calculate_sequence_logprob(var_tokens)

            # Fitness score = log(P(variant)) - log(P(reference))
            fitness_score = var_logprob - ref_logprob

            return fitness_score

        except Exception as e:
            logger.warning(f"Error calculating fitness score: {e}")
            return 0.0

    def _calculate_sequence_logprob(self, tokens: List[int]) -> float:
        """Calculate log probability of a token sequence"""
        if len(tokens) == 0:
            return 0.0

        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_ids)
            # Handle GPT2 output format (tuple or object with logits attribute)
            if hasattr(outputs, "logits"):
                logits = outputs.logits[0]  # Remove batch dimension
            else:
                # GPT2 returns tuple (logits, past)
                logits = outputs[0][0]  # [batch_size, seq_len, vocab_size] -> [seq_len, vocab_size]

            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Sum log probabilities for the sequence (excluding first token prediction)
            total_logprob = 0.0
            seq_len = min(len(tokens) - 1, logits.shape[0])
            for i in range(seq_len):
                token_id = tokens[i + 1]
                if i < logits.shape[0] and token_id < logits.shape[1]:
                    total_logprob += log_probs[i, token_id].item()

            # Normalize by sequence length
            if len(tokens) > 1:
                total_logprob /= len(tokens) - 1

            return total_logprob

    def evaluate_dataset(self, data_path: str, threshold: float = 0.0) -> Dict:
        """
        Evaluate model on a classification dataset

        Args:
            data_path: Path to evaluation dataset CSV
            threshold: Threshold for binary classification

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Loading evaluation data from {data_path}")

        # Load dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} protein variants")

        # Check required columns
        required_columns = ["sequence", "variant_pos", "ref_aa", "alt_aa", "pathogenic"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Show distribution
        logger.info("Pathogenicity distribution:")
        logger.info(df["pathogenic"].value_counts())

        # Calculate fitness scores
        logger.info("Starting model evaluation on protein variants")

        fitness_scores = []
        true_labels = []

        for idx, row in df.iterrows():
            try:
                # Calculate fitness score
                fitness_score = self.calculate_fitness_score(
                    sequence=row["sequence"],
                    variant_pos=int(row["variant_pos"]),
                    ref_aa=row["ref_aa"],
                    alt_aa=row["alt_aa"],
                )

                fitness_scores.append(fitness_score)
                true_labels.append(int(row["pathogenic"]))

            except Exception as e:
                logger.warning(f"Error processing variant {idx}: {e}")
                # Use neutral score for failed variants
                fitness_scores.append(0.0)
                true_labels.append(int(row["pathogenic"]))

        logger.info("Model evaluation completed")

        # Convert to numpy arrays
        fitness_scores = np.array(fitness_scores)
        true_labels = np.array(true_labels)

        # Binary predictions (negative fitness = pathogenic)
        predictions = (fitness_scores < threshold).astype(int)  # type: ignore[operator]

        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, predictions, fitness_scores)

        # Log results
        logger.info("=== Evaluation Results ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return {
            "metrics": metrics,
            "true_labels": true_labels.tolist(),  # type: ignore[attr-defined]
            "predictions": predictions.tolist(),
            "fitness_scores": fitness_scores.tolist(),  # type: ignore[attr-defined]
            "threshold": threshold,
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""

        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC and PR AUC (using negative fitness scores as pathogenic probabilities)
        try:
            pathogenic_probs = 1 / (1 + np.exp(y_scores))  # Sigmoid of negative fitness
            roc_auc = roc_auc_score(y_true, pathogenic_probs)
            pr_auc = average_precision_score(y_true, pathogenic_probs)
        except ValueError:
            # Handle case where all labels are the same
            roc_auc = 0.5
            pr_auc = np.mean(y_true)

        # Confusion matrix for sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
        }


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """Create a sample protein variant dataset for evaluation"""

    # Sample protein sequences (various lengths and types)
    sample_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MGSSHHHHHHSSGLVPRGSHMKELKRLTCCKVQTCLRPPGQRQELAYFFKALPQCCNLCSPLVQNPKNCT",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEL",
        "MADEAAQGAFQPGASGSRSKELKEAEDEAEEAEEAKEAEEEAKEAEEEAKEAEEEAKEAEEEA",
        "MGKEKIFSDDVRAIKEQKMLQIKHTAMAEVFLEQLACKMYSVDANTIKDFDLQHIWWNTVEQCE",
    ]

    data = []
    np.random.seed(42)

    for i in range(num_samples):
        # Select random sequence
        sequence = np.random.choice(sample_sequences)

        # Select random position (avoid start/end)
        variant_pos = np.random.randint(5, len(sequence) - 5)
        ref_aa = sequence[variant_pos]

        # Select random alternative amino acid
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        alt_aa = np.random.choice([aa for aa in amino_acids if aa != ref_aa])

        # Assign pathogenicity based on some rules (for demonstration)
        # In reality, this would come from databases like ClinVar
        pathogenic = 0

        # Simple heuristic: certain amino acid changes more likely pathogenic
        if ref_aa in "CGHPWY" and alt_aa not in "ACDEFGHIKLMNPQRSTVWY"[:10]:
            pathogenic = 1
        elif variant_pos < len(sequence) * 0.3:  # N-terminal region
            pathogenic = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            pathogenic = np.random.choice([0, 1], p=[0.8, 0.2])

        data.append(
            {
                "variant_id": f"VAR_{i:03d}",
                "sequence": sequence,
                "variant_pos": variant_pos,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "pathogenic": pathogenic,
                "description": f"{ref_aa}{variant_pos + 1}{alt_aa}",
            }
        )

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample dataset with {len(df)} variants")
    logger.info(f"Pathogenic distribution: {df['pathogenic'].value_counts().to_dict()}")
    logger.info(f"Dataset saved to: {output_path}")

    return output_path


def get_protein_tokenizer_path():
    """
    Get tokenizer path for protein_sequence
    protein_sequence uses EsmSequenceTokenizer, so returns None

    Returns:
        None: protein_sequence does not use SentencePiece
    """
    # protein_sequence uses EsmSequenceTokenizer, so
    # No SentencePiece tokenizer required
    logger.info("protein_sequence uses EsmSequenceTokenizer, not SentencePiece")
    return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate protein sequence GPT2 model with classification metrics")

    parser.add_argument("--model_path", required=True, help="Path to trained GPT2 model checkpoint")

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: use EsmSequenceTokenizer)",
    )

    parser.add_argument("--data_path", help="Path to evaluation dataset CSV")

    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample evaluation dataset",
    )

    parser.add_argument(
        "--output_dir",
        default="./protein_classification_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binary classification",
    )

    args = parser.parse_args()

    # Automatically generate output directory or use specified one
    if hasattr(args, "output_dir") and args.output_dir != "./protein_classification_results":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        output_dir = get_evaluation_output_dir(model_type, "protein_classification", model_name)

    # Log settings
    logger = setup_evaluation_logging(output_dir, "protein_classification_evaluation")

    # Create sample dataset if requested
    if args.create_sample_data or not args.data_path:
        sample_data_path = output_dir / "sample_protein_variants.csv"
        args.data_path = create_sample_dataset(str(sample_data_path))

    # Validate inputs
    if not args.data_path or not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {args.data_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    # Get tokenizer pass
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = get_protein_tokenizer_path()

    # protein_sequence uses EsmSequenceTokenizer, sotokenizer_pathteethNoneBut it is possible
    if tokenizer_path and tokenizer_path != "None" and not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    # Not used if None
    if tokenizer_path == "None":
        tokenizer_path = None

    try:
        # Initialize evaluator
        evaluator = ProteinClassificationEvaluator(model_path=args.model_path, tokenizer_path=tokenizer_path)

        # Run evaluation
        results = evaluator.evaluate_dataset(data_path=args.data_path, threshold=args.threshold)

        # Save results
        results_file = output_dir / "protein_classification_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
