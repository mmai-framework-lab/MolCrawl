#!/usr/bin/env python3
"""
Protein Classification evaluation result visualization script

Visualize the results of protein variant classification evaluation using the GPT-2 model and perform detailed analysis.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# add project root

from molcrawl.core.utils.base_visualization import BaseVisualizationGenerator

# Japanese font settings
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "Hiragino Sans",
    "Yu Gothic",
    "Meiryo",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "VL PGothic",
    "Noto Sans CJK JP",
]

# Log settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProteinClassificationVisualizer(BaseVisualizationGenerator):
    """Protein Classification evaluation result visualization class"""

    def __init__(self, results_file, output_dir="./visualization_results"):
        """
        initialization

        Args:
            results_file (str): Path of evaluation result JSON file
            output_dir (str): Output directory of visualization results
        """
        # Initialize parent class
        super().__init__(results_file, output_dir, logger)

        # Protein Classification specific validation
        required_keys = ["metrics", "true_labels", "predictions", "fitness_scores"]
        self._validate_results(required_keys)

        # data extraction
        self.metrics = self.results["metrics"]
        self.true_labels = np.array(self.results["true_labels"])
        self.predictions = np.array(self.results["predictions"])
        self.fitness_scores = np.array(self.results["fitness_scores"])
        self.threshold = self.results.get("threshold", 0.0)

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        self.logger.info("Creating confusion matrix plot")

        cm = confusion_matrix(self.true_labels, self.predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Benign", "Predicted Pathogenic"],
            yticklabels=["Actual Benign", "Actual Pathogenic"],
        )

        plt.title("Confusion Matrix - Protein Variant Classification")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        # add accuracy
        accuracy = self.metrics["Accuracy"]
        plt.figtext(0.02, 0.02, f"Accuracy: {accuracy:.3f}", fontsize=12)

        plt.tight_layout()
        self._save_plot("confusion_matrix")

    def plot_performance_metrics(self):
        """Plot bar graph of performance indicators"""
        self.logger.info("Creating performance metrics plot")

        metrics_to_plot = {
            "Accuracy": self.metrics["Accuracy"],
            "Precision": self.metrics["Precision"],
            "Recall": self.metrics["Recall"],
            "F1-Score": self.metrics["F1-score"],
            "Sensitivity": self.metrics["Sensitivity"],
            "Specificity": self.metrics["Specificity"],
        }

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            metrics_to_plot.keys(),
            metrics_to_plot.values(),
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        )

        plt.title("Model Performance Metrics on Protein Variants")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # display the value above the bar
        for bar, value in zip(bars, metrics_to_plot.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot("performance_metrics")

    def plot_auc_comparison(self):
        """AUC Metric Comparison Plot"""
        self.logger.info("Creating AUC comparison plot")

        auc_metrics = {
            "ROC-AUC": self.metrics["ROC-AUC"],
            "PR-AUC": self.metrics["PR-AUC"],
        }

        plt.figure(figsize=(8, 6))
        bars = plt.bar(auc_metrics.keys(), auc_metrics.values(), color=["#ff7f0e", "#2ca02c"])

        plt.title("Area Under Curve (AUC) Metrics")
        plt.ylabel("AUC Score")
        plt.ylim(0, 1)

        # display the value above the bar
        for bar, value in zip(bars, auc_metrics.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Add baseline for random prediction
        plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Prediction")
        plt.legend()

        plt.tight_layout()
        self._save_plot("auc_metrics")

    def plot_roc_curve(self):
        """Plot ROC curve"""
        self.logger.info("Creating ROC curve plot")

        # Calculate pathogenic probabilities (using negative fitness scores)
        pathogenic_probs = 1 / (1 + np.exp(self.fitness_scores))

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.true_labels, pathogenic_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot("roc_curve")

    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve"""
        self.logger.info("Creating Precision-Recall curve plot")

        # Calculate pathogenic probabilities
        pathogenic_probs = 1 / (1 + np.exp(self.fitness_scores))

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(self.true_labels, pathogenic_probs)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )

        # Baseline (proportion of pathogenic variants)
        baseline = np.mean(self.true_labels)
        plt.plot(
            [0, 1],
            [baseline, baseline],
            color="red",
            lw=2,
            linestyle="--",
            label=f"Random Classifier (baseline={baseline:.3f})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot("precision_recall_curve")

    def plot_fitness_score_distribution(self):
        """Plot the distribution of Fitness score"""
        self.logger.info("Creating fitness score distribution plot")

        benign_scores = self.fitness_scores[self.true_labels == 0]
        pathogenic_scores = self.fitness_scores[self.true_labels == 1]

        plt.figure(figsize=(10, 6))

        # Histogram
        plt.hist(
            benign_scores,
            bins=30,
            alpha=0.5,
            label="Benign",
            color="blue",
            edgecolor="black",
        )
        plt.hist(
            pathogenic_scores,
            bins=30,
            alpha=0.5,
            label="Pathogenic",
            color="red",
            edgecolor="black",
        )

        # threshold line
        plt.axvline(
            x=self.threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {self.threshold}",
        )

        plt.xlabel("Fitness Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Fitness Scores by Pathogenicity")
        plt.legend()
        plt.grid(alpha=0.3, axis="y")

        # Add statistics
        stats_text = (
            f"Benign: μ={np.mean(benign_scores):.3f}, σ={np.std(benign_scores):.3f}\n"
            f"Pathogenic: μ={np.mean(pathogenic_scores):.3f}, σ={np.std(pathogenic_scores):.3f}"
        )
        plt.figtext(
            0.15,
            0.95,
            stats_text,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            verticalalignment="top",
        )

        plt.tight_layout()
        self._save_plot("fitness_score_distribution")

    def plot_performance_radar_chart(self):
        """Create a radar chart of performance indicators"""
        self.logger.info("Creating performance radar chart")

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Sensitivity",
            "Specificity",
        ]
        values = [
            self.metrics["Accuracy"],
            self.metrics["Precision"],
            self.metrics["Recall"],
            self.metrics["F1-score"],
            self.metrics["Sensitivity"],
            self.metrics["Specificity"],
        ]

        # Radar chart anglecalculate
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # add first value to close
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        ax.plot(angles, values, "o-", linewidth=2, color="blue")
        ax.fill(angles, values, alpha=0.25, color="blue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True)

        plt.title("Performance Metrics Radar Chart", size=16, y=1.08)

        plt.tight_layout()
        self._save_plot("performance_radar")

    def create_summary_dashboard(self):
        """Generate summary dashboard (combine multiple plots)"""
        self.logger.info("Creating summary dashboard")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Mix rows(Top left)
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(self.true_labels, self.predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=["Benign", "Pathogenic"],
            yticklabels=["Benign", "Pathogenic"],
        )
        ax1.set_title("Confusion Matrix")
        ax1.set_ylabel("Actual")
        ax1.set_xlabel("Predicted")

        # 2. Performance indicator bar (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_subset = ["Accuracy", "Precision", "Recall", "F1-score"]
        values = [self.metrics[m] for m in metrics_subset]
        ax2.bar(range(len(metrics_subset)), values, color="steelblue")
        ax2.set_xticks(range(len(metrics_subset)))
        ax2.set_xticklabels(metrics_subset, rotation=45, ha="right")
        ax2.set_ylim(0, 1)
        ax2.set_title("Performance Metrics")
        ax2.grid(axis="y", alpha=0.3)

        # 3. AUCComparison (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        auc_names = ["ROC-AUC", "PR-AUC"]
        auc_values = [self.metrics["ROC-AUC"], self.metrics["PR-AUC"]]
        ax3.bar(auc_names, auc_values, color=["orange", "green"])
        ax3.set_ylim(0, 1)
        ax3.set_title("AUC Metrics")
        ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax3.grid(axis="y", alpha=0.3)

        # 4. ROCCurve (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        pathogenic_probs = 1 / (1 + np.exp(self.fitness_scores))
        fpr, tpr, _ = roc_curve(self.true_labels, pathogenic_probs)
        ax4.plot(
            fpr,
            tpr,
            "b-",
            linewidth=2,
            label=f"ROC (AUC={self.metrics['ROC-AUC']:.3f})",
        )
        ax4.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
        ax4.set_xlabel("False Positive Rate")
        ax4.set_ylabel("True Positive Rate")
        ax4.set_title("ROC Curve")
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. PRCurve (middle)
        ax5 = fig.add_subplot(gs[1, 1])
        precision, recall, _ = precision_recall_curve(self.true_labels, pathogenic_probs)
        ax5.plot(
            recall,
            precision,
            "g-",
            linewidth=2,
            label=f"PR (AUC={self.metrics['PR-AUC']:.3f})",
        )
        baseline = np.mean(self.true_labels)
        ax5.plot(
            [0, 1],
            [baseline, baseline],
            "r--",
            alpha=0.5,
            label=f"Baseline={baseline:.3f}",
        )
        ax5.set_xlabel("Recall")
        ax5.set_ylabel("Precision")
        ax5.set_title("Precision-Recall Curve")
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Fitness Score distribution (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        benign_scores = self.fitness_scores[self.true_labels == 0]
        pathogenic_scores = self.fitness_scores[self.true_labels == 1]
        ax6.hist(benign_scores, bins=20, alpha=0.5, label="Benign", color="blue")
        ax6.hist(pathogenic_scores, bins=20, alpha=0.5, label="Pathogenic", color="red")
        ax6.axvline(x=self.threshold, color="green", linestyle="--", label="Threshold")
        ax6.set_xlabel("Fitness Score")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Score Distribution")
        ax6.legend()

        # 7-9. Statistical information text (lower row)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("off")

        stats_text = f"""
        Dataset Statistics:
        • Total Variants: {len(self.true_labels)}
        • Benign: {np.sum(self.true_labels == 0)} ({np.mean(self.true_labels == 0) * 100:.1f}%)
        • Pathogenic: {np.sum(self.true_labels == 1)} ({np.mean(self.true_labels == 1) * 100:.1f}%)

        Performance Summary:
        • Accuracy: {self.metrics["Accuracy"]:.4f}  • Precision: {self.metrics["Precision"]:.4f}  • Recall: {self.metrics["Recall"]:.4f}
        • F1-Score: {self.metrics["F1-score"]:.4f}  • ROC-AUC: {self.metrics["ROC-AUC"]:.4f}  • PR-AUC: {self.metrics["PR-AUC"]:.4f}
        • Sensitivity: {self.metrics["Sensitivity"]:.4f}  • Specificity: {self.metrics["Specificity"]:.4f}
        """

        ax7.text(
            0.1,
            0.5,
            stats_text,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.suptitle("Protein Variant Classification - Summary Dashboard", fontsize=16, y=0.995)

        self._save_plot("summary_dashboard")

    def create_html_report(self):
        """Generate HTML report"""
        self.logger.info("Creating HTML report")

        # Generate a simple HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Protein Classification Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-high {{ color: green; font-weight: bold; }}
        .metric-medium {{ color: orange; }}
        .metric-low {{ color: red; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
        .image-item {{ text-align: center; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .summary-box {{ background-color: #e7f3fe; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Protein Variant Classification Evaluation Report</h1>

        <div class="summary-box">
            <h3>Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h3>
            <p><strong>Total Variants Evaluated:</strong> {len(self.true_labels)}</p>
            <p><strong>Classification Threshold:</strong> {self.threshold}</p>
        </div>

        <h2>📊 Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td class="{"metric-high" if self.metrics["Accuracy"] >= 0.8 else "metric-medium" if self.metrics["Accuracy"] >= 0.6 else "metric-low"}">{self.metrics["Accuracy"]:.4f}</td>
                <td>{"Excellent" if self.metrics["Accuracy"] >= 0.8 else "Good" if self.metrics["Accuracy"] >= 0.6 else "Needs Improvement"}</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{self.metrics["Precision"]:.4f}</td>
                <td>Proportion of positive predictions that are correct</td>
            </tr>
            <tr>
                <td>Recall (Sensitivity)</td>
                <td>{self.metrics["Recall"]:.4f}</td>
                <td>Proportion of actual positives correctly identified</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{self.metrics["F1-score"]:.4f}</td>
                <td>Harmonic mean of precision and recall</td>
            </tr>
            <tr>
                <td>ROC-AUC</td>
                <td class="{"metric-high" if self.metrics["ROC-AUC"] >= 0.9 else "metric-medium" if self.metrics["ROC-AUC"] >= 0.7 else "metric-low"}">{self.metrics["ROC-AUC"]:.4f}</td>
                <td>{"Excellent discrimination" if self.metrics["ROC-AUC"] >= 0.9 else "Good discrimination" if self.metrics["ROC-AUC"] >= 0.7 else "Fair discrimination"}</td>
            </tr>
            <tr>
                <td>PR-AUC</td>
                <td>{self.metrics["PR-AUC"]:.4f}</td>
                <td>Precision-Recall area under curve</td>
            </tr>
            <tr>
                <td>Specificity</td>
                <td>{self.metrics["Specificity"]:.4f}</td>
                <td>Proportion of actual negatives correctly identified</td>
            </tr>
        </table>

        <h2>📈 Visualizations</h2>
        <div class="image-grid">
            <div class="image-item">
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="image-item">
                <h3>Performance Metrics</h3>
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
            <div class="image-item">
                <h3>ROC Curve</h3>
                <img src="roc_curve.png" alt="ROC Curve">
            </div>
            <div class="image-item">
                <h3>Fitness Score Distribution</h3>
                <img src="fitness_score_distribution.png" alt="Score Distribution">
            </div>
        </div>

        <h2>📝 Dataset Information</h2>
        <ul>
            <li><strong>Benign Variants:</strong> {np.sum(self.true_labels == 0)} ({np.mean(self.true_labels == 0) * 100:.1f}%)</li>
            <li><strong>Pathogenic Variants:</strong> {np.sum(self.true_labels == 1)} ({np.mean(self.true_labels == 1) * 100:.1f}%)</li>
        </ul>

        <h2>🔍 Score Statistics</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            <tr>
                <td>Benign Variants</td>
                <td>{np.mean(self.fitness_scores[self.true_labels == 0]):.4f}</td>
                <td>{np.std(self.fitness_scores[self.true_labels == 0]):.4f}</td>
                <td>{np.min(self.fitness_scores[self.true_labels == 0]):.4f}</td>
                <td>{np.max(self.fitness_scores[self.true_labels == 0]):.4f}</td>
            </tr>
            <tr>
                <td>Pathogenic Variants</td>
                <td>{np.mean(self.fitness_scores[self.true_labels == 1]):.4f}</td>
                <td>{np.std(self.fitness_scores[self.true_labels == 1]):.4f}</td>
                <td>{np.min(self.fitness_scores[self.true_labels == 1]):.4f}</td>
                <td>{np.max(self.fitness_scores[self.true_labels == 1]):.4f}</td>
            </tr>
        </table>
    </div>
</body>
</html>
"""

        html_file = self.output_dir / "evaluation_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML report saved to {html_file}")

    def create_summary_report(self):
        """Generate summary report (text format)"""
        self.logger.info("Creating summary report")

        report = []
        report.append("=" * 80)
        report.append("PROTEIN VARIANT CLASSIFICATION EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset information
        report.append("Dataset Information:")
        report.append(f"  Total variants: {len(self.true_labels)}")
        report.append(f"  Benign variants: {np.sum(self.true_labels == 0)} ({np.mean(self.true_labels == 0) * 100:.1f}%)")
        report.append(f"  Pathogenic variants: {np.sum(self.true_labels == 1)} ({np.mean(self.true_labels == 1) * 100:.1f}%)")
        report.append(f"  Classification threshold: {self.threshold}")
        report.append("")

        # Performance indicators
        report.append("Performance Metrics:")
        report.append("-" * 40)
        for metric, value in self.metrics.items():
            report.append(f"  {metric:20s}: {value:.4f}")
        report.append("")

        # Mix rows
        cm = confusion_matrix(self.true_labels, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        report.append("Confusion Matrix:")
        report.append("-" * 40)
        report.append(f"  True Negative (TN):  {tn:4d}")
        report.append(f"  False Positive (FP): {fp:4d}")
        report.append(f"  False Negative (FN): {fn:4d}")
        report.append(f"  True Positive (TP):  {tp:4d}")
        report.append("")

        # Fitness scorestatistics
        report.append("Fitness Score Statistics:")
        report.append("-" * 40)
        benign_scores = self.fitness_scores[self.true_labels == 0]
        pathogenic_scores = self.fitness_scores[self.true_labels == 1]

        report.append("  Benign variants:")
        report.append(f"    Mean:   {np.mean(benign_scores):7.4f}")
        report.append(f"    Std:    {np.std(benign_scores):7.4f}")
        report.append(f"    Min:    {np.min(benign_scores):7.4f}")
        report.append(f"    Max:    {np.max(benign_scores):7.4f}")
        report.append("")

        report.append("  Pathogenic variants:")
        report.append(f"    Mean:   {np.mean(pathogenic_scores):7.4f}")
        report.append(f"    Std:    {np.std(pathogenic_scores):7.4f}")
        report.append(f"    Min:    {np.min(pathogenic_scores):7.4f}")
        report.append(f"    Max:    {np.max(pathogenic_scores):7.4f}")
        report.append("")

        # model interpretation
        report.append("Model Interpretation:")
        report.append("-" * 40)
        if self.metrics["Accuracy"] >= 0.8:
            report.append("  ✓ Excellent classification performance")
        elif self.metrics["Accuracy"] >= 0.7:
            report.append("  ✓ Good classification performance")
        elif self.metrics["Accuracy"] >= 0.6:
            report.append("  ⚠ Moderate classification performance")
        else:
            report.append("  ✗ Poor classification performance - further tuning needed")

        if self.metrics["ROC-AUC"] >= 0.9:
            report.append("  ✓ Excellent discriminative ability (ROC-AUC ≥ 0.9)")
        elif self.metrics["ROC-AUC"] >= 0.8:
            report.append("  ✓ Good discriminative ability (ROC-AUC ≥ 0.8)")
        elif self.metrics["ROC-AUC"] >= 0.7:
            report.append("  ⚠ Fair discriminative ability (ROC-AUC ≥ 0.7)")
        else:
            report.append("  ✗ Poor discriminative ability - model may need improvement")

        report.append("")
        report.append("=" * 80)

        # save report
        report_text = "\n".join(report)
        report_file = self.output_dir / "evaluation_summary.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        self.logger.info(f"Summary report saved to {report_file}")

        # Also output to console
        print("\n" + report_text)

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        self.logger.info("Generating all visualizations")

        try:
            self.plot_confusion_matrix()
            self.plot_performance_metrics()
            self.plot_auc_comparison()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_fitness_score_distribution()
            self.plot_performance_radar_chart()
            self.create_summary_dashboard()
            self.create_html_report()
            self.create_summary_report()

            self.logger.info("All visualizations completed successfully")
            self.logger.info(f"Output directory: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Visualize Protein Classification Evaluation Results")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./protein_classification_visualizations",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    # Check the result file
    if not os.path.exists(args.results_file):
        logger.error(f"Results file not found: {args.results_file}")
        sys.exit(1)

    # Visualize execution
    try:
        visualizer = ProteinClassificationVisualizer(results_file=args.results_file, output_dir=args.output_dir)

        visualizer.generate_all_visualizations()

        logger.info("=" * 80)
        logger.info("Visualization completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
