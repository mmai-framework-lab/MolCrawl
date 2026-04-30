#!/usr/bin/env python3
"""
Base Visualization Framework for Model Evaluation Results

This module provides a common interface and shared functionality for all
evaluation result visualization classes.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from molcrawl.core.utils.image_manager import get_image_output_dir


class BaseVisualizationGenerator(ABC):
    """
    Abstract base class for visualization of evaluation results

    Provides common interfaces and functionality that all visualization classes should inherit.

    Common features:
    - Manage output directory
    - Unification of plot settings
    - Common utility methods
    - Foundation for HTML report generation
    """

    def __init__(
        self,
        results_source: Union[str, Dict[str, Any]],
        output_dir: str = "./visualization_results",
        logger: Optional[logging.Logger] = None,
        model_type: Optional[str] = None,
    ):
        """
        Base class initialization

        Args:
            results_source: Evaluation results (file path or dictionary data)
            output_dir: Output directory of visualization results
            logger: logger for log output
            model_type: Model type(For determining image storage directory)
        """
        # Use unified image directory if model type is specified
        if model_type:
            try:
                self.image_dir = Path(get_image_output_dir(model_type))
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.model_type = model_type
            except Exception:
                # Fallback in case of environment variable setting error etc.
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.image_dir = self.output_dir
                self.model_type = None
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.image_dir = self.output_dir
            self.model_type = self._detect_model_type_from_output_dir(output_dir)

        self.logger = logger or self._setup_logger()

        # Load result data
        self.results = self._load_results(results_source)

        # Initializing visualization settings
        self._setup_plot_style()

        # generatelist of files
        self.generated_files: List[Path] = []

        self.logger.info(f"Visualization generator initialized. Output directory: {self.output_dir}")

    def _setup_logger(self) -> logging.Logger:
        """Set default logger"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(self.__class__.__name__)

    def _detect_model_type_from_output_dir(self, output_dir: str) -> Optional[str]:
        """Infer model type from output directory path"""
        output_path = str(output_dir).lower()

        model_keywords = {
            "protein_sequence": ["protein", "proteingym"],
            "genome_sequence": ["genome", "clinvar", "cosmic", "omim"],
            "compounds": ["compound", "smiles", "scaffold"],
            "rna": ["rna"],
            "molecule_nat_lang": ["molecule_nat_lang", "molecule-nl", "moleculenl"],
        }

        for model_type, keywords in model_keywords.items():
            if any(keyword in output_path for keyword in keywords):
                return model_type

        return None

    def _load_results(self, results_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Loading evaluation result data

        Args:
            results_source: file path or dictionary data

        Returns:
            Evaluation result dictionary
        """
        if isinstance(results_source, dict):
            self.logger.info("Using provided results dictionary")
            return results_source
        elif isinstance(results_source, str):
            results_file = Path(results_source)
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")

            self.logger.info(f"Loading results from {results_file}")
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError("results_source must be a file path or dictionary")

    def _setup_plot_style(self):
        """Unified plotting style settings"""
        if plt is None or sns is None:
            raise RuntimeError("matplotlib and seaborn are required for visualization")

        # Basic style settings
        plt.style.use("default")
        sns.set_palette("husl")

        # Supports Japanese fonts
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

        # Default settings for diagrams
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

        sns.set_style("whitegrid")

        self.logger.debug("Plot style configured")

    def _save_plot(self, filename: str, formats: List[str] = None, dpi: int = 300) -> List[Path]:
        """
        Save plot in specified format

        Args:
            filename: file name (without extension)
            formats: list of formats to save (default: ['png'])
            dpi: resolution (default: 300)

        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ["png"]  # Unified with PNG priority

        saved_files = []
        for fmt in formats:
            # Save image files (png, jpg, etc.) in the unified image directory
            if fmt.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "svg"]:
                filepath = self.image_dir / f"{filename}.{fmt}"
            else:
                # Save PDF etc. in output_dir as usual
                filepath = self.output_dir / f"{filename}.{fmt}"

            plt.savefig(filepath, format=fmt, bbox_inches="tight", dpi=dpi)
            saved_files.append(filepath)
            self.generated_files.append(filepath)

        plt.close()
        self.logger.debug(f"Plot saved: {filename} in {formats} (image_dir: {self.image_dir})")
        return saved_files

    def _create_comprehensive_dashboard(
        self,
        results_df: pd.DataFrame,
        prediction_score_col: str = "score",
        true_label_col: str = "label",
        confidence_col: str = None,
        similarity_col: str = None,
        custom_title: str = "Model Evaluation Dashboard",
    ):
        """
        Create a generic comprehensive dashboard (based on BERT format)

        Args:
            results_df: results data frame
            prediction_score_col: Prediction score column name
            true_label_col: True correct label column name
            confidence_col: confidence column name (optional)
            similarity_col: similarity column name (optional)
            custom_title: Dashboard title
        """
        self.logger.info("Creating comprehensive evaluation dashboard")

        # Validate data
        required_cols = [prediction_score_col, true_label_col]
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return

        plt.style.use("default")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Distribution of prediction scores (by label)
        self._plot_score_distribution(axes[0, 0], results_df, prediction_score_col, true_label_col)

        # Similarity distribution (if available)
        if similarity_col and similarity_col in results_df.columns:
            self._plot_similarity_distribution(axes[0, 1], results_df, similarity_col, true_label_col)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Similarity data\nnot available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Similarity Distribution (N/A)")

        # Scatterplot: Score vs Similarity
        if similarity_col and similarity_col in results_df.columns:
            self._plot_score_vs_similarity_scatter(
                axes[0, 2],
                results_df,
                prediction_score_col,
                similarity_col,
                true_label_col,
            )
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "Scatter plot\nnot available",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
            )
            axes[0, 2].set_title("Score vs Similarity (N/A)")

        # ROCcurve
        self._plot_roc_curve(axes[1, 0], results_df, prediction_score_col, true_label_col)

        # Mix rows
        self._plot_confusion_matrix_subplot(axes[1, 1], results_df, prediction_score_col, true_label_col)

        # Confidence distribution
        if confidence_col and confidence_col in results_df.columns:
            self._plot_confidence_distribution(axes[1, 2], results_df, confidence_col)
        else:
            # Use as an alternative to prediction score
            self._plot_confidence_distribution(axes[1, 2], results_df, prediction_score_col)

        plt.suptitle(custom_title, fontsize=16, y=0.98)
        plt.tight_layout()
        self._save_plot("comprehensive_dashboard")

    def _plot_score_distribution(self, ax, results_df: pd.DataFrame, score_col: str, label_col: str):
        """Plot the distribution of predicted scores"""
        try:
            positive_scores = results_df[results_df[label_col] == 1][score_col]
            negative_scores = results_df[results_df[label_col] == 0][score_col]

            ax.hist(positive_scores, alpha=0.7, label="Positive", bins=30, color="red")
            ax.hist(negative_scores, alpha=0.7, label="Negative", bins=30, color="blue")
            ax.set_xlabel("Prediction Score")
            ax.set_ylabel("Count")
            ax.set_title("Score Distribution")
            ax.legend()
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Score distribution\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Score Distribution (Error)")

    def _plot_similarity_distribution(self, ax, results_df: pd.DataFrame, sim_col: str, label_col: str):
        """Plot similarity distribution"""
        try:
            positive_sim = results_df[results_df[label_col] == 1][sim_col]
            negative_sim = results_df[results_df[label_col] == 0][sim_col]

            ax.hist(positive_sim, alpha=0.7, label="Positive", bins=30, color="red")
            ax.hist(negative_sim, alpha=0.7, label="Negative", bins=30, color="blue")
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Count")
            ax.set_title("Similarity Distribution")
            ax.legend()
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Similarity distribution\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Similarity Distribution (Error)")

    def _plot_score_vs_similarity_scatter(self, ax, results_df: pd.DataFrame, score_col: str, sim_col: str, label_col: str):
        """Score vs Similarity Scatter Plot"""
        try:
            positive_data = results_df[results_df[label_col] == 1]
            negative_data = results_df[results_df[label_col] == 0]

            ax.scatter(
                positive_data[score_col],
                positive_data[sim_col],
                alpha=0.6,
                c="red",
                label="Positive",
                s=20,
            )
            ax.scatter(
                negative_data[score_col],
                negative_data[sim_col],
                alpha=0.6,
                c="blue",
                label="Negative",
                s=20,
            )
            ax.set_xlabel("Prediction Score")
            ax.set_ylabel("Similarity Score")
            ax.set_title("Score vs Similarity")
            ax.legend()
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Scatter plot\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Score vs Similarity (Error)")

    def _plot_roc_curve(self, ax, results_df: pd.DataFrame, score_col: str, label_col: str):
        """Plot ROC curve"""
        try:
            from sklearn.metrics import roc_auc_score, roc_curve

            fpr, tpr, _ = roc_curve(results_df[label_col], results_df[score_col])
            auc = roc_auc_score(results_df[label_col], results_df[score_col])

            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"ROC curve\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("ROC Curve (Error)")

    def _plot_confusion_matrix_subplot(self, ax, results_df: pd.DataFrame, score_col: str, label_col: str):
        """Plot the confusion matrix as a subplot"""
        try:
            from sklearn.metrics import confusion_matrix

            # Generate predicted value (threshold 0.5 or 0)
            threshold = 0.5 if results_df[score_col].max() > 1 else 0
            predictions = (results_df[score_col] > threshold).astype(int)
            cm = confusion_matrix(results_df[label_col], predictions)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                ax=ax,
                cmap="Blues",
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"],
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Confusion matrix\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Confusion Matrix (Error)")

    def _plot_confidence_distribution(self, ax, results_df: pd.DataFrame, conf_col: str):
        """Plot confidence distribution"""
        try:
            ax.hist(results_df[conf_col], bins=30, alpha=0.7, color="green")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count")
            ax.set_title("Confidence Distribution")
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Confidence distribution\nerror: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Confidence Distribution (Error)")

    def _create_figure_grid(self, nrows: int, ncols: int, figsize: tuple = None) -> tuple:
        """
        Create a diagram in grid format

        Args:
            nrows: number of rows
            ncols: number of columns
            figsize: figure size

        Returns:
            tuple of (figure, axes)
        """
        if figsize is None:
            figsize = (ncols * 5, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()

        return fig, axes

    def _validate_results(self, required_keys: List[str]):
        """
        Verify that the required key exists in the result data

        Args:
            required_keys: list of required keys

        Raises:
            KeyError: If the required key does not exist
        """
        missing_keys = [key for key in required_keys if key not in self.results]
        if missing_keys:
            raise KeyError(f"Missing required keys in results: {missing_keys}")

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _create_html_header(self, title: str) -> str:
        """Generate header part of HTML report"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #fafafa;
                }}
                .header {{
                    background-color: #f0f0f0;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid #2196F3;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #2196F3;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 30px 0;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                }}
                .section-title {{
                    color: #333;
                    border-bottom: 2px solid #2196F3;
                    padding-bottom: 10px;
                    margin-top: 40px;
                    margin-bottom: 20px;
                }}
                .footer {{
                    margin-top: 50px;
                    padding: 20px;
                    background-color: #f0f0f0;
                    border-radius: 8px;
                    text-align: center;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on: {self._get_timestamp()}</p>
            </div>
        """

    def _create_html_footer(self) -> str:
        """Generate footer part of HTML report"""
        return f"""
            <div class="footer">
                <p>Report generated by {self.__class__.__name__}</p>
                <p>Files created: {len(self.generated_files)} visualizations</p>
            </div>
        </body>
        </html>
        """

    def get_generated_files(self) -> List[Path]:
        """Get list of generated files"""
        return self.generated_files.copy()

    # Abstract methods - Must be implemented in subclasses

    @abstractmethod
    def plot_confusion_matrix(self):
        """Generation of confusion matrix plot (required to implement)"""
        pass

    @abstractmethod
    def plot_performance_metrics(self):
        """Generation of performance index plot (required to implement)"""
        pass

    @abstractmethod
    def create_summary_dashboard(self):
        """Generate summary dashboard (required to implement)"""
        pass

    @abstractmethod
    def generate_all_visualizations(self):
        """Generation of all visualizations (required to implement)"""
        pass

    @abstractmethod
    def create_html_report(self):
        """Generate HTML report (required to implement)"""
        pass

    # Optional abstract method (implemented by subclass as needed)

    def plot_auc_comparison(self):
        """Comparison plot of AUC metrics (optional)"""
        self.logger.info("AUC comparison plot not implemented in this visualizer")

    def create_performance_radar_chart(self):
        """Performance indicator radar chart (optional)"""
        self.logger.info("Performance radar chart not implemented in this visualizer")

    def plot_score_distribution(self):
        """Score distribution plot (optional)"""
        self.logger.info("Score distribution plot not implemented in this visualizer")
