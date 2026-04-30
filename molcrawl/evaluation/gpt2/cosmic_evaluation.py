#!/usr/bin/env python3
"""
Genome sequence model evaluation script using COSMIC data

Using cancer-associated mutation data from the COSMIC database,
Evaluate the performance of genome sequence models in predicting variant pathogenicity.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
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
    roc_curve,
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


class COSMICEvaluator(ModelEvaluator):
    """Model evaluation class using COSMIC data"""

    def __init__(self, model_path, tokenizer_path=None, device=None):
        """
        initialization

        Args:
            model_path (str): Path of trained model
            tokenizer_path (str): tokenizer path
            device (str): Device to use ('cuda' or 'cpu')
        """
        # Initialize parent class(Auto-initialize tokenizer and model)
        super().__init__(
            model_path,
            tokenizer_path or get_genome_tokenizer_path(),
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        logger.info(f"Model loaded successfully. Config: {self.model.config}")

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
        """Load model"""
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

        # Create model and load weights
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()

        return model

    def get_oncogenic_probability(self, reference_seq, variant_seq, context_length=512):
        """
        Calculate the tumorigenic probability of a mutation

        Args:
            reference_seq (str): reference sequence
            variant_seq (str): variant sequence
            context_length (int): Context length used for evaluation

        Returns:
            float: tumorigenic probability of mutation
        """
        with torch.no_grad():
            # encode reference and variant sequences
            ref_tokens = self.encode_sequence(reference_seq)
            var_tokens = self.encode_sequence(variant_seq)

            # minimum length check
            if len(ref_tokens) == 0 or len(var_tokens) == 0:
                return 0.0

            # adjust to context length
            if len(ref_tokens) > context_length:
                ref_tokens = ref_tokens[:context_length]
            if len(var_tokens) > context_length:
                var_tokens = var_tokens[:context_length]

            # Add batch dimension and transfer to device
            ref_tokens = ref_tokens.unsqueeze(0).to(self.device)
            var_tokens = var_tokens.unsqueeze(0).to(self.device)

            # calculate likelihood
            ref_likelihood = self._calculate_likelihood_token_by_token(ref_tokens)
            var_likelihood = self._calculate_likelihood_token_by_token(var_tokens)

            # Calculate the probability of carcinogenicity from the likelihood ratio
            likelihood_ratio = var_likelihood - ref_likelihood
            oncogenic_prob = torch.sigmoid(likelihood_ratio * 10).item()  # Scaling

            return oncogenic_prob

    def _calculate_likelihood_token_by_token(self, tokens):
        """Calculate likelihood for each token"""
        if tokens.size(1) <= 1:
            return torch.tensor(0.0, device=tokens.device)

        total_log_prob = 0.0
        count = 0

        # Compute the conditional probability at each position
        for i in range(1, tokens.size(1)):
            context = tokens[:, :i]
            target = tokens[:, i : i + 1]

            with torch.no_grad():
                logits, _ = self.model(context)
                log_probs = F.log_softmax(logits[:, -1:, :], dim=-1)

                # Log probability of correct token
                token_log_prob = log_probs.gather(2, target.unsqueeze(2)).squeeze()
                total_log_prob += token_log_prob.item()
                count += 1

        return torch.tensor(total_log_prob / count if count > 0 else 0.0, device=tokens.device)

    def load_cosmic_data(self, data_path):
        """
        Load COSMIC data

        Args:
            data_path (str): COSMIC data file path

        Returns:
            pd.DataFrame: Processed COSMIC data
        """
        logger.info(f"Loading COSMIC data from {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} COSMIC variants")

        # Check required columns
        required_columns = [
            "Reference_sequence",
            "Variant_sequence",
            "Cancer_significance",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {df.columns.tolist()}")

        # Standardization of carcinogenicity labels
        df = self._standardize_cancer_significance(df)

        logger.info(f"Cancer significance distribution:\n{df['oncogenic'].value_counts()}")

        return df

    def _standardize_cancer_significance(self, df):
        """Standardizing the clinical significance of carcinogenicity"""

        def classify_oncogenicity(significance):
            significance = str(significance).lower()

            # Carcinogenic pattern
            oncogenic_terms = ["pathogenic", "oncogenic", "likely_pathogenic", "driver"]
            # Non-cancerous pattern
            benign_terms = ["benign", "likely_benign", "neutral", "passenger"]

            if any(term in significance for term in oncogenic_terms):
                return 1  # carcinogenicity
            elif any(term in significance for term in benign_terms):
                return 0  # non-cancerous
            else:
                return None  # Unknown (excluded from evaluation)

        df["oncogenic"] = df["Cancer_significance"].apply(classify_oncogenicity)

        # exclude unknowns
        df = df.dropna(subset=["oncogenic"])
        df["oncogenic"] = df["oncogenic"].astype(int)

        return df

    def evaluate_model(self, cosmic_data, batch_size=16):
        """
        Run model evaluation

        Args:
            cosmic_data (pd.DataFrame): COSMIC data
            batch_size (int): Batch size

        Returns:
            dict: evaluation result
        """
        logger.info("Starting model evaluation on COSMIC data")

        oncogenicity_scores = []
        true_labels = []

        for idx, row in cosmic_data.iterrows():
            try:
                # Calculate carcinogenicity probability
                score = self.get_oncogenic_probability(row["Reference_sequence"], row["Variant_sequence"])

                oncogenicity_scores.append(score)
                true_labels.append(row["oncogenic"])

                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(cosmic_data)} variants")

            except Exception as e:
                logger.warning(f"Error processing variant {idx}: {e}")
                continue

        if not oncogenicity_scores:
            raise ValueError("No valid oncogenicity scores computed")

        # convert to array
        oncogenicity_scores = np.array(oncogenicity_scores)
        true_labels = np.array(true_labels)

        # Determine the optimal threshold
        optimal_threshold = self._find_optimal_threshold(oncogenicity_scores, true_labels)

        # Generate predicted labels
        predicted_labels = (oncogenicity_scores >= optimal_threshold).astype(int)

        # Calculate evaluation metrics
        results = self._calculate_metrics(true_labels, predicted_labels, oncogenicity_scores, optimal_threshold)

        logger.info("Model evaluation completed")

        return results

    def _find_optimal_threshold(self, scores, labels):
        """Determine the threshold that maximizes the F1 score from the ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find the threshold that maximizes the F1 score
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _calculate_metrics(self, true_labels, predicted_labels, scores, threshold):
        """Calculate evaluation metrics"""

        # Basic indicators
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except ValueError:
            roc_auc = 0.5

        # PR-AUC
        try:
            pr_auc = average_precision_score(true_labels, scores)
        except ValueError:
            pr_auc = 0.5

        # Mix rows
        cm = confusion_matrix(true_labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Sensitivity (Recall) and Specificity
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
            "confusion_matrix": cm.tolist(),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_samples": len(true_labels),
            "oncogenic_samples": int(np.sum(true_labels)),
            "non_oncogenic_samples": int(len(true_labels) - np.sum(true_labels)),
        }

        return results

    def save_results(self, results, output_dir):
        """Save results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results in JSON format
        results_file = output_dir / "cosmic_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        report_file = output_dir / "cosmic_evaluation_report.txt"
        with open(report_file, "w") as f:
            f.write("COSMIC Oncogenicity Prediction Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {results['total_samples']}\n")
            f.write(f"Oncogenic samples: {results['oncogenic_samples']}\n")
            f.write(f"Non-oncogenic samples: {results['non_oncogenic_samples']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1-score: {results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {results['specificity']:.4f}\n")
            f.write(f"  Optimal threshold: {results['optimal_threshold']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives: {results['true_positives']}\n")
            f.write(f"  False Positives: {results['false_positives']}\n")
            f.write(f"  True Negatives: {results['true_negatives']}\n")
            f.write(f"  False Negatives: {results['false_negatives']}\n")

        logger.info(f"Results saved to {output_dir}")

        return results_file, report_file


def main():
    """Main processing"""
    parser = argparse.ArgumentParser(description="COSMIC-based genome sequence model evaluation")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--cosmic_data", required=True, help="Path to COSMIC evaluation dataset")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (auto-generated if not provided)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
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

    # Automatically generate output directory or use specified one
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "cosmic", model_name)

    # Log settings
    logger = setup_evaluation_logging(Path(args.output_dir), "cosmic_evaluation")

    try:
        # Get tokenizer pass
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
            logger.info(f"Using specified tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = get_genome_tokenizer_path()
            logger.info(f"Using auto-detected tokenizer: {tokenizer_path}")

        # Initialize the evaluator
        evaluator = COSMICEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
        )

        # Load COSMIC data
        cosmic_data = evaluator.load_cosmic_data(args.cosmic_data)

        # model evaluation
        results = evaluator.evaluate_model(cosmic_data, batch_size=args.batch_size)

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
        evaluator.save_results(results, args.output_dir)

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
