#!/usr/bin/env python3
"""
Genome sequence model accuracy verification script using ClinVar database

This script uses a trained GPT-2 genome sequence model to
Validate the accuracy of identifying pathogenic variants in the ClinVar database.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# add project root

from molcrawl.gpt2.model import GPT, GPTConfig

from molcrawl.core.paths import get_genome_tokenizer_path
from molcrawl.core.utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from molcrawl.core.utils.model_evaluator import ModelEvaluator

# Log settingslatersetup_evaluation_loggingdo it with
logger = logging.getLogger(__name__)


class GPT2ClinVarEvaluator(ModelEvaluator):
    """Model evaluation class using ClinVar data"""

    def __init__(self, model_path, tokenizer_path, device="cuda"):
        """
        initialization

        Args:
            model_path (str): path of the trained model
            tokenizer_path (str): SentencePiece tokenizer path
            device (str): Device used
        """
        # Initialize parent class(Auto-initialize tokenizer and model)
        super().__init__(model_path, tokenizer_path, device)

    def _init_tokenizer(self):
        """Tokenizer initialization (abstract method implementation)"""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        self.vocab_size = tokenizer.vocab_size()
        logger.info(f"Tokenizer loaded with vocab_size: {self.vocab_size}")
        return tokenizer

    def _init_model(self):
        """Model initialization (abstract method implementation)"""
        logger.info(f"Loading model from {self.model_path}")
        return self._load_model()

    def encode_sequence(self, sequence: str, **kwargs):
        """
        Encode array to token ID (implementation of abstract method)

        Args:
            sequence: genome sequence

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(sequence)

    def _load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Restore model settings
        model_args = checkpoint.get("model_args", {})
        config = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=model_args.get("block_size", 1024),
            n_layer=model_args.get("n_layer", 12),
            n_head=model_args.get("n_head", 12),
            n_embd=model_args.get("n_embd", 768),
            dropout=0.0,  # Disable dropout during evaluation
            bias=model_args.get("bias", True),
        )

        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully. Config: {config}")
        return model

    def calculate_perplexity(self, sequence, variant_pos=None):
        """Calculate the perplexity of a sequence"""
        """Encode DNA sequence into token ID"""
        # Format the array properly (convert uppercase, remove invalid characters, etc.)
        sequence = sequence.upper().replace("N", "").replace("-", "")

        # Encode with SentencePiece
        tokens = self.tokenizer.encode(sequence)

        # debug log
        logger.debug(f"Original sequence length: {len(sequence)}")
        logger.debug(f"Encoded tokens length: {len(tokens)}")
        logger.debug(f"First 10 chars: {sequence[:10]}")
        logger.debug(f"First 5 tokens: {tokens[:5] if tokens else 'Empty'}")

        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            # Return unknown token if empty
            tokens = [self.tokenizer.unk_id()] if hasattr(self.tokenizer, "unk_id") else [0]

        return torch.tensor(tokens, dtype=torch.long)

    def get_variant_probability(self, reference_seq, variant_seq, context_length=512):
        """
        Calculate the pathogenicity probability of a mutation

        Args:
            reference_seq (str): reference sequence
            variant_seq (str): variant sequence
            context_length (int): Context length used for evaluation

        Returns:
            float: Pathogenic probability of mutation (difference in relative likelihood)
        """
        with torch.no_grad():
            # encode reference and variant sequences
            ref_tokens = self.encode_sequence(reference_seq)
            var_tokens = self.encode_sequence(variant_seq)

            # Ensure minimum length (1 or more)
            if len(ref_tokens) == 0:
                logger.warning("Reference sequence tokenization resulted in empty tokens")
                return 0.0
            if len(var_tokens) == 0:
                logger.warning("Variant sequence tokenization resulted in empty tokens")
                return 0.0

            # adjust to context length
            if len(ref_tokens) > context_length:
                ref_tokens = ref_tokens[:context_length]
            if len(var_tokens) > context_length:
                var_tokens = var_tokens[:context_length]

            # Convert list to tensor then add batch dimension
            ref_tokens = torch.tensor(ref_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            var_tokens = torch.tensor(var_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

            logger.debug(f"Model input shapes - ref: {ref_tokens.shape}, var: {var_tokens.shape}")

            # Model predictionGet probability (prediction of all series)
            # Pass dummy value to targets and get logits of all series (disable optimization during inference)
            ref_dummy_targets = torch.zeros_like(ref_tokens)
            var_dummy_targets = torch.zeros_like(var_tokens)
            ref_logits, _ = self.model(ref_tokens, targets=ref_dummy_targets)
            var_logits, _ = self.model(var_tokens, targets=var_dummy_targets)

            logger.debug(f"Model output shapes - ref: {ref_logits.shape}, var: {var_logits.shape}")

            # calculate log likelihood
            ref_log_prob = F.log_softmax(ref_logits, dim=-1)
            var_log_prob = F.log_softmax(var_logits, dim=-1)

            # Compute the likelihood of each token
            ref_likelihood = self._calculate_sequence_likelihood(ref_tokens, ref_log_prob)
            var_likelihood = self._calculate_sequence_likelihood(var_tokens, var_log_prob)

            # Use relative likelihood difference as pathogenicity score
            pathogenicity_score = ref_likelihood - var_likelihood

            return pathogenicity_score.item()

    def _calculate_sequence_likelihood(self, tokens, log_probs):
        """Calculate the log-likelihood of an array"""
        # Input check
        if tokens.size(1) <= 1:
            logger.warning(f"Sequence too short for likelihood calculation: {tokens.shape}")
            return torch.tensor(0.0, device=tokens.device)

        if log_probs.size(1) == 0:
            logger.warning(f"Empty log_probs: {log_probs.shape}")
            return torch.tensor(0.0, device=tokens.device)

        # Exclude the last token (because there is no prediction target)
        target_tokens = tokens[:, 1:]
        pred_log_probs = log_probs[:, :-1, :]

        # Double check the size
        if pred_log_probs.size(1) != target_tokens.size(1):
            logger.warning(f"Size mismatch: pred_log_probs={pred_log_probs.shape}, target_tokens={target_tokens.shape}")
            return torch.tensor(0.0, device=tokens.device)

        # Get the log probability of the correct token at each position
        token_log_probs = pred_log_probs.gather(2, target_tokens.unsqueeze(2)).squeeze(2)

        # Return the average log-likelihood
        return token_log_probs.mean()

    def _calculate_likelihood_token_by_token(self, tokens):
        """Calculate the likelihood for each token (compatible with generation mode)"""
        if tokens.size(1) <= 1:
            return torch.tensor(0.0, device=tokens.device)

        total_log_prob = 0.0
        count = 0

        # Compute the conditional probability at each position
        for i in range(1, tokens.size(1)):
            context = tokens[:, :i]  # Context up to i-th
            target = tokens[:, i : i + 1]  # i+1th token (prediction target)

            with torch.no_grad():
                logits, _ = self.model(context)
                log_probs = F.log_softmax(logits[:, -1:, :], dim=-1)  # Only predict the last position

                # Log probability of correct token
                token_log_prob = log_probs.gather(2, target.unsqueeze(2)).squeeze()
                total_log_prob += token_log_prob.item()
                count += 1

        return torch.tensor(total_log_prob / count if count > 0 else 0.0, device=tokens.device)

    def load_clinvar_data(self, clinvar_file):
        """
        Loading ClinVar data

        Args:
            clinvar_file (str): ClinVar data file path

        Returns:
            pd.DataFrame: ClinVar data
        """
        logger.info(f"Loading ClinVar data from {clinvar_file}")

        # Load according to file format
        if clinvar_file.endswith(".csv"):
            df = pd.read_csv(clinvar_file)
        elif clinvar_file.endswith(".tsv"):
            df = pd.read_csv(clinvar_file, sep="\t")
        elif clinvar_file.endswith(".json"):
            df = pd.read_json(clinvar_file)
        else:
            raise ValueError(f"Unsupported file format: {clinvar_file}")

        # Check required columns
        required_columns = [
            "reference_sequence",
            "variant_sequence",
            "ClinicalSignificance",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")

        # Standardization of pathogenicity labels
        df = self._standardize_clinical_significance(df)

        logger.info(f"Loaded {len(df)} ClinVar variants")
        logger.info(f"Clinical significance distribution:\n{df['pathogenic'].value_counts()}")

        return df

    def _standardize_clinical_significance(self, df):
        """Standardize clinical significance into a binary classification of pathogenic/non-pathogenic"""
        pathogenic_terms = [
            "pathogenic",
            "likely pathogenic",
            "pathogenic/likely pathogenic",
        ]
        benign_terms = ["benign", "likely benign", "benign/likely benign"]

        def classify_pathogenicity(significance):
            if pd.isna(significance):
                return None
            significance = significance.lower()
            if any(term in significance for term in pathogenic_terms):
                return 1  # Pathogenicity
            elif any(term in significance for term in benign_terms):
                return 0  # non-pathogenic
            else:
                return None  # Unknown (excluded from evaluation)

        df["pathogenic"] = df["ClinicalSignificance"].apply(classify_pathogenicity)

        # exclude unknowns
        df = df.dropna(subset=["pathogenic"])
        df["pathogenic"] = df["pathogenic"].astype(int)

        return df

    def evaluate_model(self, clinvar_data, batch_size=32):
        """
        Run model evaluation

        Args:
            clinvar_data (pd.DataFrame): ClinVar data
            batch_size (int): Batch size

        Returns:
            dict: evaluation result
        """
        logger.info("Starting model evaluation on ClinVar data")

        predictions = []
        true_labels = []
        pathogenicity_scores = []

        for i in range(0, len(clinvar_data), batch_size):
            batch = clinvar_data.iloc[i : i + batch_size]

            for _, row in batch.iterrows():
                try:
                    # Calculate virulence score
                    score = self.get_variant_probability(row["reference_sequence"], row["variant_sequence"])

                    pathogenicity_scores.append(score)
                    true_labels.append(row["pathogenic"])

                    logger.debug(f"Processed variant {len(predictions) + 1}/{len(clinvar_data)}")

                except Exception as e:
                    logger.warning(f"Error processing variant: {e}")
                    continue

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(clinvar_data)} variants")

        # Optimize the threshold to determine the predicted label
        pathogenicity_scores = np.array(pathogenicity_scores)
        true_labels = np.array(true_labels)

        # ROCCalculate optimal threshold from curve
        optimal_threshold = self._find_optimal_threshold(pathogenicity_scores, true_labels)
        predictions = (pathogenicity_scores > optimal_threshold).astype(int)

        # Calculate evaluation metrics
        results = self._calculate_metrics(true_labels, predictions, pathogenicity_scores)
        results["optimal_threshold"] = optimal_threshold

        logger.info("Model evaluation completed")
        return results

    def _find_optimal_threshold(self, scores, labels):
        """Find the threshold that maximizes the F1 score from the ROC curve"""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find the threshold that maximizes F1 score
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            pred = (scores > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, pred, average="binary")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _calculate_metrics(self, true_labels, predictions, scores):
        """Calculate evaluation metrics"""
        from sklearn.metrics import average_precision_score, roc_auc_score

        # Basic indicators
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

        # AUCindex
        roc_auc = roc_auc_score(true_labels, scores)
        pr_auc = average_precision_score(true_labels, scores)

        # Mix rows
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        }

        return results

    def save_results(self, results, output_file):
        """Save evaluation results"""
        logger.info(f"Saving results to {output_file}")

        # Convert NumPy array to list
        serializable_results = self._make_serializable(results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def create_sample_clinvar_data(output_file):
    """
    ⚠️ DEPRECATED: Move data preparation functionality to clinvar_data_preparation.py
    Use: python scripts/evaluation/gpt2/clinvar_data_preparation.py --mode sample
    """
    logger.warning("⚠️  create_sample_clinvar_data() is deprecated.")
    logger.warning("Please use: python scripts/evaluation/gpt2/clinvar_data_preparation.py --mode sample")
    raise DeprecationWarning("Use clinvar_data_preparation.py for data preparation")


def main():
    parser = argparse.ArgumentParser(description="ClinVar evaluation for genome sequence model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--clinvar_data",
        type=str,
        required=True,
        help="Path to ClinVar data file (CSV/TSV/JSON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample ClinVar data for testing",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to SentencePiece tokenizer model (auto-detect if not provided)",
    )

    args = parser.parse_args()

    # Automatically generate output directory or use specified one
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "clinvar", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Log settings
    logger = setup_evaluation_logging(Path(args.output_dir), "clinvar_evaluation")

    # Sample data creation mode
    if args.create_sample_data:
        create_sample_clinvar_data(args.clinvar_data)
        logger.info("Sample data created. Run again without --create_sample_data to evaluate.")
        return

    try:
        # Get tokenizer pass
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
            logger.info(f"Using specified tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = get_genome_tokenizer_path()
            logger.info(f"Using auto-detected tokenizer: {tokenizer_path}")

        # Initialize the evaluator
        evaluator = GPT2ClinVarEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
        )

        # Load ClinVar data
        clinvar_data = evaluator.load_clinvar_data(args.clinvar_data)

        # model evaluationexecution of
        results = evaluator.evaluate_model(clinvar_data, batch_size=args.batch_size)

        # Display results
        logger.info("=== Evaluation Results ===")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-score: {results['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {results['pr_auc']:.4f}")
        logger.info(f"Sensitivity: {results['sensitivity']:.4f}")
        logger.info(f"Specificity: {results['specificity']:.4f}")

        # Save results
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        evaluator.save_results(results, results_file)

        # Create detailed report
        report_file = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_file, "w") as f:
            f.write("ClinVar Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Data: {args.clinvar_data}\n")
            f.write(f"Total variants evaluated: {len(clinvar_data)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1-score: {results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {results['specificity']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            cm = results["confusion_matrix"]
            f.write(f"  True Positive: {cm['true_positive']}\n")
            f.write(f"  False Positive: {cm['false_positive']}\n")
            f.write(f"  True Negative: {cm['true_negative']}\n")
            f.write(f"  False Negative: {cm['false_negative']}\n")

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
