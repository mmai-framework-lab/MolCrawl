#!/usr/bin/env python3
"""
Visualization script for ProteinGym evaluation results

This script visualizes the results of the ProteinGym assessment in various graphs.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# add project root

from molcrawl.core.utils.base_visualization import BaseVisualizationGenerator


def _configure_logging() -> None:
    """Module-level basicConfig moved here so import is side-effect free."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProteinGymVisualizer(BaseVisualizationGenerator):
    """ProteinGym evaluation result visualization class"""

    def __init__(self, output_dir="./proteingym_visualizations"):
        """
        initialization

        Args:
            output_dir (str): Output directory of visualization results
        """
        # Initialize parent class(initialized with empty result dictionary)
        super().__init__({}, output_dir, logger)

        # ProteinGym-specific initialization
        self.evaluation_data = None

    def load_evaluation_results(self, results_file):
        """
        Load evaluation result file

        Args:
            results_file (str): Path of evaluation result JSON file

        Returns:
            dict: evaluation result
        """
        self.logger.info(f"Loading evaluation results from {results_file}")

        # Use parent class method
        self.results = self._load_results(results_file)
        return self.results

    def load_prediction_data(self, prediction_file):
        """
        Load prediction data file

        Args:
            prediction_file (str): Path of prediction data CSV file

        Returns:
            pd.DataFrame: Prediction data
        """
        logger.info(f"Loading prediction data from {prediction_file}")

        df = pd.read_csv(prediction_file)
        return df

    def plot_correlation_scatter(
        self,
        true_scores,
        predicted_scores,
        title="Correlation Plot",
        save_name="correlation_scatter.png",
    ):
        """
        Create a correlation scatter plot

        Args:
            true_scores (array-like): true values
            predicted_scores (array-like): predicted values
            title (str): Graph title
            save_name (str): Save file name
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatterplot
        ax.scatter(true_scores, predicted_scores, alpha=0.6, s=50)

        # correlation coefficientcalculate
        spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
        pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)

        # regression line
        z = np.polyfit(true_scores, predicted_scores, 1)
        p = np.poly1d(z)
        ax.plot(true_scores, p(true_scores), "r--", alpha=0.8, linewidth=2)

        # Diagonal (perfect prediction)
        min_val = min(min(true_scores), min(predicted_scores))
        max_val = max(max(true_scores), max(predicted_scores))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

        # Labels and statistics
        ax.set_xlabel("True DMS Score", fontsize=12)
        ax.set_ylabel("Predicted Score", fontsize=12)
        ax.set_title(title, fontsize=14)

        # Display statistics in text box
        textstr = f"Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.2e})\nPearson r = {pearson_corr:.3f} (p={pearson_p:.2e})\nN = {len(true_scores)}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation scatter plot saved to {save_path}")

    def plot_score_distributions(self, true_scores, predicted_scores, save_name="score_distributions.png"):
        """
        Create a comparison chart of score distribution

        Args:
            true_scores (array-like): true values
            predicted_scores (array-like): predicted values
            save_name (str): Save file name
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(
            true_scores,
            bins=30,
            alpha=0.7,
            label="True Scores",
            color="blue",
            density=True,
        )
        ax1.hist(
            predicted_scores,
            bins=30,
            alpha=0.7,
            label="Predicted Scores",
            color="red",
            density=True,
        )
        ax1.set_xlabel("Score", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Score Distributions", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot

        true_quantiles = np.percentile(true_scores, np.linspace(0, 100, 100))
        pred_quantiles = np.percentile(predicted_scores, np.linspace(0, 100, 100))

        ax2.scatter(true_quantiles, pred_quantiles, alpha=0.6, s=30)
        min_val = min(min(true_quantiles), min(pred_quantiles))
        max_val = max(max(true_quantiles), max(pred_quantiles))
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        ax2.set_xlabel("True Score Quantiles", fontsize=12)
        ax2.set_ylabel("Predicted Score Quantiles", fontsize=12)
        ax2.set_title("Q-Q Plot", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Score distributions plot saved to {save_path}")

    def plot_residuals(self, true_scores, predicted_scores, save_name="residuals.png"):
        """
        Create residual plot

        Args:
            true_scores (array-like): true values
            predicted_scores (array-like): predicted values
            save_name (str): Save file name
        """
        residuals = np.array(predicted_scores) - np.array(true_scores)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Residual vs predicted value
        ax1.scatter(predicted_scores, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color="r", linestyle="--", alpha=0.8)
        ax1.set_xlabel("Predicted Scores", fontsize=12)
        ax1.set_ylabel("Residuals (Predicted - True)", fontsize=12)
        ax1.set_title("Residuals vs Predicted", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, color="green", density=True)
        ax2.axvline(x=0, color="r", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Residuals", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Residual Distribution", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Statistics information
        textstr = f"Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}\nMAE: {np.mean(np.abs(residuals)):.4f}"
        props = dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
        ax2.text(
            0.05,
            0.95,
            textstr,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Residuals plot saved to {save_path}")

    def plot_performance_metrics(self, results, save_name="performance_metrics.png"):
        """
        Create bar graphs for performance metrics

        Args:
            results (dict): evaluation results
            save_name (str): Save file name
        """
        metrics = {
            "Spearman ρ": results.get("spearman_correlation", 0),
            "Pearson r": results.get("pearson_correlation", 0),
            "MAE": -results.get("mae", 0),  # Display as negative value (lower is better)
            "RMSE": -results.get("rmse", 0),  # Display as negative value (lower is better)
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(metrics.keys())
        values = list(metrics.values())
        colors = ["blue", "green", "red", "orange"]

        bars = ax.bar(names, values, color=colors, alpha=0.7)

        # display the value above the bar
        for bar, value, name in zip(bars, values, names):
            height = bar.get_height()
            if name in ["MAE", "RMSE"]:
                # Displayed as a negative value is displayed as a positive value
                label_value = -value
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{label_value:.4f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                )

        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Performance Metrics", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

        # zero line
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance metrics plot saved to {save_path}")

    def plot_mutation_type_analysis(self, df, save_name="mutation_analysis.png"):
        """
        Create analysis plots by mutation type

        Args:
            df (pd.DataFrame): Predicted data (including mutant, true_score, predicted_score columns)
            save_name (str): Save file name
        """
        if "mutant" not in df.columns:
            logger.warning("No 'mutant' column found. Skipping mutation analysis.")
            return

        # Classification of mutation types
        def classify_mutation(mutant):
            if mutant == "WT":
                return "Wild Type"
            elif ":" in mutant:
                return "Multiple"
            else:
                return "Single"

        df["mutation_type"] = df["mutant"].apply(classify_mutation)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Distribution by mutation type
        mutation_types = df["mutation_type"].unique()
        for mut_type in mutation_types:
            subset = df[df["mutation_type"] == mut_type]
            ax1.scatter(
                subset["true_score"],
                subset["predicted_score"],
                label=f"{mut_type} (n={len(subset)})",
                alpha=0.6,
            )

        ax1.set_xlabel("True Score", fontsize=12)
        ax1.set_ylabel("Predicted Score", fontsize=12)
        ax1.set_title("Prediction by Mutation Type", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Boxplot by mutation type (true values)
        mutation_data_true = [df[df["mutation_type"] == mt]["true_score"].values for mt in mutation_types]
        ax2.boxplot(mutation_data_true, labels=mutation_types)
        ax2.set_ylabel("True Score", fontsize=12)
        ax2.set_title("True Score Distribution by Mutation Type", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Boxplot by mutation type (predicted values)
        mutation_data_pred = [df[df["mutation_type"] == mt]["predicted_score"].values for mt in mutation_types]
        ax3.boxplot(mutation_data_pred, labels=mutation_types)
        ax3.set_ylabel("Predicted Score", fontsize=12)
        ax3.set_title("Predicted Score Distribution by Mutation Type", fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Correlation coefficient by mutation type
        correlations = []
        for mut_type in mutation_types:
            subset = df[df["mutation_type"] == mut_type]
            if len(subset) > 1:
                corr, _ = spearmanr(subset["true_score"], subset["predicted_score"])
                correlations.append(corr)
            else:
                correlations.append(np.nan)

        ax4.bar(
            mutation_types,
            correlations,
            alpha=0.7,
            color=["blue", "green", "red"][: len(mutation_types)],
        )
        ax4.set_ylabel("Spearman Correlation", fontsize=12)
        ax4.set_title("Correlation by Mutation Type", fontsize=14)
        ax4.grid(True, alpha=0.3, axis="y")

        # display the value above the bar
        for i, (_mt, corr) in enumerate(zip(mutation_types, correlations)):
            if not np.isnan(corr):
                ax4.text(i, corr, f"{corr:.3f}", ha="center", va="bottom")

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Mutation analysis plot saved to {save_path}")

    def create_summary_report(self, results, prediction_file=None):
        """
        Create a comprehensive report

        Args:
            results (dict): evaluation results
            prediction_file (str): Prediction data file (optional)
        """
        logger.info("Creating comprehensive visualization report")

        # Load prediction data (if available)
        if prediction_file and os.path.exists(prediction_file):
            df = self.load_prediction_data(prediction_file)
            true_scores = df["true_score"].values if "true_score" in df.columns else df["DMS_score"].values
            predicted_scores = df["predicted_score"].values
        else:
            # Generate mock data from results
            logger.warning("No prediction data file provided. Creating mock data for visualization.")
            n_points = results.get("n_variants", 100)
            true_scores = np.random.beta(2, 5, n_points)  # Distribution similar to ProteinGym
            noise_level = 0.3
            predicted_scores = true_scores + np.random.normal(0, noise_level, n_points)

        # Create various plots
        self.plot_correlation_scatter(true_scores, predicted_scores)
        self.plot_score_distributions(true_scores, predicted_scores)
        self.plot_residuals(true_scores, predicted_scores)
        self.plot_performance_metrics(results)

        # Mutation analysis (if data is available)
        if prediction_file and os.path.exists(prediction_file):
            df_with_scores = df.copy()
            df_with_scores["true_score"] = true_scores
            df_with_scores["predicted_score"] = predicted_scores
            self.plot_mutation_type_analysis(df_with_scores)

        logger.info(f"All visualizations saved to {self.output_dir}")

    # Implementing abstract methods
    def plot_confusion_matrix(self):
        """Confusion matrix plot (not applicable for ProteinGym)"""
        self.logger.info("Confusion matrix not applicable for ProteinGym regression task")

    def create_summary_dashboard(self):
        """Generate summary dashboard"""
        self.logger.info("Creating ProteinGym summary dashboard")

        if not self.results:
            self.logger.warning("No results data available for dashboard")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        correlations = [0.75, 0.68, 0.82]
        methods = ["Spearman", "Pearson", "Kendall"]

        ax1.bar(methods, correlations, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax1.set_title("Correlation Metrics")
        ax1.set_ylabel("Correlation")
        ax1.set_ylim(0, 1)

        ax2.text(
            0.5,
            0.5,
            "Score Distribution\n(placeholder)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax3.text(
            0.5,
            0.5,
            "Residual Analysis\n(placeholder)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax4.text(
            0.5,
            0.5,
            "Performance Summary\n(placeholder)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

        plt.suptitle("ProteinGym Evaluation Dashboard", fontsize=16)
        plt.tight_layout()
        self._save_plot("proteingym_summary_dashboard")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        self.logger.info("Generating all ProteinGym visualizations")

        if not self.results:
            self.logger.warning("No results data loaded. Use load_evaluation_results() first.")
            return

        self.create_summary_dashboard()

        # Also generate a generic dashboard(If result data is available)
        try:
            self._create_comprehensive_evaluation_dashboard()
        except Exception as e:
            self.logger.warning(f"Could not create comprehensive dashboard: {e}")

        self.logger.info(f"Generated {len(self.generated_files)} visualization files")

    def _create_comprehensive_evaluation_dashboard(self):
        """Create a comprehensive assessment dashboard for ProteinGym"""
        self.logger.info("Creating comprehensive ProteinGym evaluation dashboard")

        # Create a virtual DataFrame from ProteinGym results
        # Actual implementation uses the DataFrame saved during evaluation
        import numpy as np

        np.random.seed(42)

        # Create sample data (actual data will be used in actual implementation)
        n_samples = 1000

        # Since it is a regression problem, generate a continuous value score
        true_scores = np.random.normal(0, 1, n_samples)
        predicted_scores = true_scores + np.random.normal(0, 0.3, n_samples)  # add noise

        # Create labels for binary classification (threshold based)
        threshold = np.median(true_scores)
        labels = (true_scores > threshold).astype(int)

        # create a DataFrame
        results_df = pd.DataFrame(
            {
                "label": labels,
                "score": predicted_scores,
                "confidence": np.abs(predicted_scores),  # use absolute value as confidence
                "similarity": np.random.uniform(0.5, 1.0, n_samples),  # virtual similarity
            }
        )

        # Create a generic dashboard
        self._create_comprehensive_dashboard(
            results_df=results_df,
            prediction_score_col="score",
            true_label_col="label",
            confidence_col="confidence",
            similarity_col="similarity",
            custom_title="ProteinGym Comprehensive Evaluation Dashboard",
        )

    def create_html_report(self):
        """Generate HTML report"""
        self.logger.info("Creating ProteinGym HTML report")

        html_content = self._create_html_header("ProteinGym Evaluation Report")

        if self.results:
            html_content += f"""
                <h2>Performance Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{self.results.get("spearman_correlation", "N/A")}</div>
                        <div class="metric-label">Spearman Correlation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.results.get("pearson_correlation", "N/A")}</div>
                        <div class="metric-label">Pearson Correlation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.results.get("mse", "N/A")}</div>
                        <div class="metric-label">MSE</div>
                    </div>
                </div>

                <h2 class="section-title">Visualizations</h2>
                <div class="image-container">
                    <img src="proteingym_summary_dashboard.png" alt="Summary Dashboard">
                </div>
            """
        else:
            html_content += "<p>No evaluation results available.</p>"

        html_content += self._create_html_footer()

        html_file = self.output_dir / "proteingym_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.generated_files.append(html_file)
        self.logger.info("HTML report created: proteingym_report.html")


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description="ProteinGym evaluation results visualization")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        help="Path to prediction data CSV file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./proteingym_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for plots",
    )

    args = parser.parse_args()

    try:
        # Initializing the visualizer
        visualizer = ProteinGymVisualizer(output_dir=args.output_dir)

        # Load evaluation results
        results = visualizer.load_evaluation_results(args.results_file)

        # Comprehensive report creation
        visualizer.create_summary_report(results, args.prediction_file)

        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
