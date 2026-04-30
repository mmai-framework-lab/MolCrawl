#!/usr/bin/env python3
"""
OMIM Evaluation Script
======================

Using the OMIM (Online Mendelian Inheritance in Man) database
Script to evaluate genetic disease prediction performance of genome sequence models

Main features:
- Model evaluation with OMIM genetic disease data
- Measurement of pathogenic variant prediction accuracy
- Detailed analysis by genetic type
- Comprehensive evaluation report generation

Evaluation metrics:
- Accuracy
- Precision rate (Precision)
- Recall rate (Recall)
- F1 score
- ROC-AUC, PR-AUC
- Sensitivity/specificity
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# add project root

try:
    from molcrawl.models.gpt2.model import GPT, GPTConfig
    from molcrawl.core.paths import get_genome_tokenizer_path
    from molcrawl.core.utils.evaluation_output import (
        get_evaluation_output_dir,
        get_model_name_from_path,
        get_model_type_from_path,
        setup_evaluation_logging,
    )
    from molcrawl.core.utils.model_evaluator import ModelEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# setup_logging function uses utils.evaluation_output.setup_evaluation_logging


class OMIMEvaluator(ModelEvaluator):
    """OMIM Genetic Disease Prediction Evaluation Class"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # Initialize parent class(Auto-initialize tokenizer and model)
        super().__init__(
            model_path,
            tokenizer_path,
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # For saving evaluation results
        self.results: Dict[str, Any] = {}
        self.predictions: List[int] = []
        self.true_labels: List[int] = []
        self.prediction_scores: List[float] = []

    def _init_tokenizer(self):
        """Tokenizer initialization (abstract method implementation)"""
        self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        self.vocab_size = tokenizer.vocab_size()
        self.logger.info(f"Tokenizer loaded with vocab_size: {self.vocab_size}")
        return tokenizer

    def _init_model(self):
        """Model initialization (abstract method implementation)"""
        self.logger.info(f"Loading model from {self.model_path}")
        return self.load_model_and_tokenizer()

    def encode_sequence(self, sequence: str, **kwargs) -> List[int]:
        """
        Encode array to token ID (implementation of abstract method)

        Args:
            sequence: genome sequence

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(sequence)

    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        try:
            # load tokenizer
            self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.tokenizer_path)
            self.logger.info("Tokenizer loaded successfully")

            # Load model (create dummy model if file does not exist)
            self.logger.info(f"Loading model from {self.model_path}")

            if os.path.exists(self.model_path):
                # If the actual model file exists
                checkpoint = torch.load(self.model_path, map_location=self.device)

                model_args = checkpoint["model_args"]
                gptconf = GPTConfig(**model_args)
                self.model = GPT(gptconf)

                state_dict = checkpoint["model"]
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_orig_mod.")}
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model.to(self.device)

                self.logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                # Create a dummy model (for testing)
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.logger.info("Creating dummy model for testing purposes")

                # Create GPTConfig with dummy settings
                vocab_size = self.tokenizer.vocab_size()
                gptconf = GPTConfig(
                    block_size=512,
                    vocab_size=vocab_size,
                    n_layer=6,
                    n_head=6,
                    n_embd=384,
                    dropout=0.0,
                    bias=False,
                )

                self.model = GPT(gptconf)
                self.model.eval()
                self.model.to(self.device)

                self.logger.info("Dummy model created successfully")

        except Exception as e:
            self.logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def tokenize_sequence(self, sequence: str, max_length: int = 512) -> torch.Tensor:
        """Tokenize an array"""
        # Tokenize with spaces
        spaced_sequence = " ".join(list(sequence.upper()))
        tokens = self.tokenizer.encode(spaced_sequence, out_type=int)

        # Length adjustment
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def calculate_sequence_likelihood(self, sequence: str) -> float:
        """Calculate the likelihood of an array"""
        try:
            tokens = self.tokenize_sequence(sequence)

            with torch.no_grad():
                # model prediction
                logits, _ = self.model(tokens)

                # Log likelihood calculation
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # sum the log-likelihood at each position
                total_log_likelihood = 0.0
                for i in range(tokens.size(1) - 1):
                    target_token = tokens[0, i + 1]
                    token_log_prob = log_probs[0, i, target_token]
                    total_log_likelihood += token_log_prob.item()

                # Average log-likelihood
                avg_log_likelihood = total_log_likelihood / (tokens.size(1) - 1)

                return avg_log_likelihood

        except Exception as e:
            self.logger.warning(f"Error calculating likelihood for sequence: {e}")
            return -np.inf

    def load_omim_data(self, data_path: str) -> pd.DataFrame:
        """Load OMIM data"""
        self.logger.info(f"Loading OMIM data from {data_path}")

        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} OMIM variants")

        if "is_disease_causing" in df.columns:
            df = df.dropna(subset=["sequence", "is_disease_causing"])
            df["is_disease_causing"] = df["is_disease_causing"].astype(int)
        else:
            self.logger.error("Required column 'is_disease_causing' not found in data")
            raise ValueError("Missing required column")

        # Display genetic type distribution
        if "inheritance_pattern" in df.columns:
            self.logger.info("Inheritance pattern distribution:")
            self.logger.info(df["inheritance_pattern"].value_counts())

        # Show pathogenicity distribution
        self.logger.info("Disease causality distribution:")
        self.logger.info(df["is_disease_causing"].value_counts())

        return df

    def evaluate_variants(self, df: pd.DataFrame, batch_size: int = 16) -> Dict[str, Any]:
        """Evaluate variant"""
        self.logger.info("Starting model evaluation on OMIM data")

        # initialize list
        predictions = []
        true_labels = []
        prediction_scores = []
        inheritance_patterns = []

        total_variants = len(df)
        for i in range(0, total_variants, batch_size):
            batch_end = min(i + batch_size, total_variants)
            batch_df = df.iloc[i:batch_end]

            batch_scores = []
            for _, row in batch_df.iterrows():
                sequence = row["sequence"]
                likelihood = self.calculate_sequence_likelihood(sequence)
                batch_scores.append(likelihood)

            # add batch results
            prediction_scores.extend(batch_scores)
            true_labels.extend(batch_df["is_disease_causing"].tolist())

            if "inheritance_pattern" in batch_df.columns:
                inheritance_patterns.extend(batch_df["inheritance_pattern"].tolist())

            # Progress display
            if (batch_end) % 50 == 0 or batch_end == total_variants:
                self.logger.info(f"Processed {batch_end}/{total_variants} variants")

        # Optimize prediction threshold
        optimal_threshold = self.find_optimal_threshold(true_labels, prediction_scores)

        # Predictive label generation
        predictions = [1 if score > optimal_threshold else 0 for score in prediction_scores]

        # Save results
        self.true_labels = true_labels
        self.predictions = predictions
        self.prediction_scores = prediction_scores

        # Evaluation index calculation
        results = self.calculate_metrics(true_labels, predictions, prediction_scores, optimal_threshold)

        # Analysis by genetic type
        if inheritance_patterns:
            results["inheritance_analysis"] = self.analyze_by_inheritance_pattern(
                true_labels, predictions, inheritance_patterns
            )

        self.results = results
        self.logger.info("Model evaluation completed")

        return results

    def find_optimal_threshold(self, true_labels: List[int], scores: List[float]) -> float:
        """Find the optimal threshold"""
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]
        except (ValueError, IndexError):
            return np.median(scores)

    def calculate_metrics(
        self,
        true_labels: List[int],
        predictions: List[int],
        scores: List[float],
        threshold: float,
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics"""

        # Basic indicators
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        # ROC-AUC, PR-AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except (ValueError, RuntimeError):
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(true_labels, scores)
        except (ValueError, RuntimeError):
            pr_auc = 0.0

        # Sensitivity/specificity
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "optimal_threshold": threshold,
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            "total_samples": len(true_labels),
            "positive_samples": sum(true_labels),
            "negative_samples": len(true_labels) - sum(true_labels),
        }

        # log the results
        self.logger.info("=== Evaluation Results ===")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-score: {f1:.4f}")
        self.logger.info(f"ROC-AUC: {roc_auc:.4f}")
        self.logger.info(f"PR-AUC: {pr_auc:.4f}")
        self.logger.info(f"Sensitivity: {sensitivity:.4f}")
        self.logger.info(f"Specificity: {specificity:.4f}")

        return results

    def analyze_by_inheritance_pattern(
        self,
        true_labels: List[int],
        predictions: List[int],
        inheritance_patterns: List[str],
    ) -> Dict[str, Dict]:
        """Analysis by genetic type"""
        analysis = {}
        unique_patterns = set(inheritance_patterns)

        for pattern in unique_patterns:
            # Extract data by pattern
            pattern_indices = [i for i, p in enumerate(inheritance_patterns) if p == pattern]
            pattern_true = [true_labels[i] for i in pattern_indices]
            pattern_pred = [predictions[i] for i in pattern_indices]

            if len(pattern_true) > 0:
                # Indicator calculation by pattern
                accuracy = accuracy_score(pattern_true, pattern_pred)
                precision = precision_score(pattern_true, pattern_pred, zero_division=0)
                recall = recall_score(pattern_true, pattern_pred, zero_division=0)
                f1 = f1_score(pattern_true, pattern_pred, zero_division=0)

                analysis[pattern] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "sample_count": len(pattern_true),
                    "positive_count": sum(pattern_true),
                }

        return analysis

    def save_results(self, output_dir: str):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save in JSON format
        results_file = os.path.join(output_dir, "omim_evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save detailed report
        report_file = os.path.join(output_dir, "omim_evaluation_report.txt")
        self.generate_detailed_report(report_file)

        # Save prediction results
        predictions_file = os.path.join(output_dir, "omim_predictions.csv")
        pred_df = pd.DataFrame(
            {
                "true_label": self.true_labels,
                "prediction": self.predictions,
                "prediction_score": self.prediction_scores,
            }
        )
        pred_df.to_csv(predictions_file, index=False)

        self.logger.info(f"Results saved to {output_dir}")

    def generate_detailed_report(self, report_file: str):
        """Generate detailed report"""
        with open(report_file, "w") as f:
            f.write("OMIM Hereditary Disease Prediction Evaluation Report\n")
            f.write("=" * 55 + "\n\n")

            # Basic statistics
            f.write(f"Total samples: {self.results['total_samples']}\n")
            f.write(f"Disease-causing samples: {self.results['positive_samples']}\n")
            f.write(f"Benign samples: {self.results['negative_samples']}\n\n")

            # Performance indicators
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {self.results['accuracy']:.4f}\n")
            f.write(f"  Precision: {self.results['precision']:.4f}\n")
            f.write(f"  Recall: {self.results['recall']:.4f}\n")
            f.write(f"  F1-score: {self.results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {self.results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {self.results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {self.results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {self.results['specificity']:.4f}\n")
            f.write(f"  Optimal threshold: {self.results['optimal_threshold']:.4f}\n\n")

            # Mix rows
            cm = self.results["confusion_matrix"]
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives: {cm['tp']}\n")
            f.write(f"  False Positives: {cm['fp']}\n")
            f.write(f"  True Negatives: {cm['tn']}\n")
            f.write(f"  False Negatives: {cm['fn']}\n\n")

            # Analysis by genetic type
            if "inheritance_analysis" in self.results:
                f.write("Analysis by Inheritance Pattern:\n")
                for pattern, metrics in self.results["inheritance_analysis"].items():
                    f.write(f"\n  {pattern}:\n")
                    f.write(f"    Sample count: {metrics['sample_count']}\n")
                    f.write(f"    Positive count: {metrics['positive_count']}\n")
                    f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1-score: {metrics['f1_score']:.4f}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OMIM Evaluation for Genome Sequence Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="dummy_model",
        help="Path to the trained GPT-2 model (default: dummy_model)",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to OMIM evaluation dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to SentencePiece tokenizer model (auto-detect if not provided)",
    )

    args = parser.parse_args()

    try:
        # Automatically generate output directory or use specified one
        if args.output_dir is None:
            model_type = get_model_type_from_path(args.model_path)
            model_name = get_model_name_from_path(args.model_path)
            args.output_dir = get_evaluation_output_dir(model_type, "omim", model_name)
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        # Log settings
        logger = setup_evaluation_logging(Path(args.output_dir), "omim_evaluation")

        # Get tokenizer pass
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
            logger.info(f"Using specified tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = get_genome_tokenizer_path()
            logger.info(f"Using auto-detected tokenizer: {tokenizer_path}")

        # run evaluation
        evaluator = OMIMEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
            logger=logger,
        )

        # Data loading and evaluation
        df = evaluator.load_omim_data(args.data_path)
        evaluator.evaluate_variants(df, batch_size=args.batch_size)

        # Save results
        evaluator.save_results(args.output_dir)

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
