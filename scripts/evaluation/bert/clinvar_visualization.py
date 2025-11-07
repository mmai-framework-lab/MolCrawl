#!/usr/bin/env python3
"""
BERT ClinVar評価結果可視化スクリプト

BERT ClinVar評価の結果を可視化し、詳細な分析を行います。
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from utils.base_visualization import BaseVisualizationGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BERTClinVarVisualizationGenerator(BaseVisualizationGenerator):
    """BERT ClinVar評価結果の可視化クラス"""

    def __init__(self, results_file, output_dir="./bert_clinvar_visualization_results"):
        """
        初期化

        Args:
            results_file (str): 評価結果JSONファイルまたはCSVファイルのパス
            output_dir (str): 可視化結果の出力ディレクトリ
        """
        # 親クラスの初期化
        super().__init__(results_file, output_dir, logger)

        # BERT ClinVar固有の検証
        if self.results_file.endswith(".csv"):
            # CSVファイルの場合はDataFrameとして読み込み
            self.results_df = pd.read_csv(results_file)
            self._validate_csv_results()
        else:
            # JSONファイルの場合は従来通り
            required_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
            self._validate_results(required_keys)
            self.results_df = None

    def _validate_csv_results(self):
        """CSV結果データの検証"""
        required_columns = [
            "pathogenic",
            "mlm_score",
            "cosine_similarity",
            "confidence",
        ]
        missing_columns = [
            col for col in required_columns if col not in self.results_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        self.logger.info(
            f"Loaded BERT ClinVar results: {len(self.results_df)} variants"
        )

    def plot_confusion_matrix(self):
        """混同行列をプロット"""
        self.logger.info("Creating BERT confusion matrix plot")

        if self.results_df is not None:
            # CSVデータから混同行列を計算
            predictions = (self.results_df["mlm_score"] > 0).astype(int)
            cm = confusion_matrix(self.results_df["pathogenic"], predictions)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted Benign", "Predicted Pathogenic"],
                yticklabels=["Actual Benign", "Actual Pathogenic"],
            )

            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            plt.title(
                f"BERT Confusion Matrix - ClinVar Pathogenicity Prediction\nAccuracy: {accuracy:.3f}"
            )
        else:
            # JSONデータから混同行列をプロット
            cm_data = self.results["confusion_matrix"]
            cm = np.array(cm_data)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted Benign", "Predicted Pathogenic"],
                yticklabels=["Actual Benign", "Actual Pathogenic"],
            )

            plt.title(
                f"BERT Confusion Matrix - ClinVar Pathogenicity Prediction\nAccuracy: {self.results['accuracy']:.3f}"
            )

        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        self._save_plot("bert_confusion_matrix")

    def plot_performance_metrics(self):
        """性能指標の棒グラフをプロット"""
        self.logger.info("Creating BERT performance metrics plot")

        if self.results_df is not None:
            # CSVデータから指標を計算
            predictions = (self.results_df["mlm_score"] > 0).astype(int)
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            accuracy = accuracy_score(self.results_df["pathogenic"], predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.results_df["pathogenic"], predictions, average="binary"
            )
            try:
                auc = roc_auc_score(
                    self.results_df["pathogenic"], self.results_df["mlm_score"]
                )
            except (ValueError, RuntimeError):
                auc = 0.5

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "AUC-ROC": auc,
            }
        else:
            # JSONデータから指標を取得
            metrics = {
                "Accuracy": self.results["accuracy"],
                "Precision": self.results["precision"],
                "Recall": self.results["recall"],
                "F1-Score": self.results["f1_score"],
                "AUC-ROC": self.results["auc"],
            }

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            metrics.keys(),
            metrics.values(),
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        )

        plt.title("BERT Model Performance Metrics on ClinVar Data")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # 値をバーの上に表示
        for bar, value in zip(bars, metrics.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot("bert_performance_metrics")

    def plot_mlm_score_distribution(self):
        """MLMスコアの分布をプロット"""
        self.logger.info("Creating MLM score distribution plot")

        if self.results_df is None:
            self.logger.warning("CSV data required for MLM score distribution plot")
            return

        plt.figure(figsize=(10, 6))

        pathogenic_scores = self.results_df[self.results_df["pathogenic"] == 1][
            "mlm_score"
        ]
        benign_scores = self.results_df[self.results_df["pathogenic"] == 0]["mlm_score"]

        plt.hist(pathogenic_scores, alpha=0.7, label="Pathogenic", bins=30, color="red")
        plt.hist(benign_scores, alpha=0.7, label="Benign", bins=30, color="blue")

        plt.xlabel("MLM Score")
        plt.ylabel("Count")
        plt.title("BERT MLM Score Distribution by Pathogenicity")
        plt.legend()

        # 統計情報を追加
        path_mean = pathogenic_scores.mean()
        benign_mean = benign_scores.mean()
        plt.axvline(path_mean, color="red", linestyle="--", alpha=0.7)
        plt.axvline(benign_mean, color="blue", linestyle="--", alpha=0.7)

        # 統計情報をテキストで表示
        plt.text(
            0.02,
            0.98,
            f"Pathogenic Mean: {path_mean:.3f}\nBenign Mean: {benign_mean:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        self._save_plot("bert_mlm_score_distribution")

    def plot_similarity_distribution(self):
        """コサイン類似度の分布をプロット"""
        self.logger.info("Creating cosine similarity distribution plot")

        if self.results_df is None:
            self.logger.warning("CSV data required for similarity distribution plot")
            return

        plt.figure(figsize=(10, 6))

        pathogenic_sim = self.results_df[self.results_df["pathogenic"] == 1][
            "cosine_similarity"
        ]
        benign_sim = self.results_df[self.results_df["pathogenic"] == 0][
            "cosine_similarity"
        ]

        plt.hist(pathogenic_sim, alpha=0.7, label="Pathogenic", bins=30, color="red")
        plt.hist(benign_sim, alpha=0.7, label="Benign", bins=30, color="blue")

        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.title("Reference-Variant Sequence Similarity Distribution")
        plt.legend()

        # 統計情報を追加
        path_mean = pathogenic_sim.mean()
        benign_mean = benign_sim.mean()
        plt.axvline(path_mean, color="red", linestyle="--", alpha=0.7)
        plt.axvline(benign_mean, color="blue", linestyle="--", alpha=0.7)

        # 統計情報をテキストで表示
        plt.text(
            0.02,
            0.98,
            f"Pathogenic Mean: {path_mean:.3f}\nBenign Mean: {benign_mean:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        self._save_plot("bert_similarity_distribution")

    def create_summary_dashboard(self):
        """全体的なサマリーダッシュボードを作成"""
        self.logger.info("Creating BERT summary dashboard")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        if self.results_df is not None:
            # CSVデータを使用
            predictions = (self.results_df["mlm_score"] > 0).astype(int)
            cm = confusion_matrix(self.results_df["pathogenic"], predictions)

            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            accuracy = accuracy_score(self.results_df["pathogenic"], predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.results_df["pathogenic"], predictions, average="binary"
            )
            try:
                auc = roc_auc_score(
                    self.results_df["pathogenic"], self.results_df["mlm_score"]
                )
            except (ValueError, RuntimeError):
                auc = 0.5

            # 1. 混同行列
            ax1 = fig.add_subplot(gs[0, 0])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred Benign", "Pred Pathogenic"],
                yticklabels=["Act Benign", "Act Pathogenic"],
                ax=ax1,
            )
            ax1.set_title("Confusion Matrix")

            # 2. 性能指標
            ax2 = fig.add_subplot(gs[0, 1])
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            values = [accuracy, precision, recall, f1]
            bars = ax2.bar(
                metrics, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            )
            ax2.set_title("Performance Metrics")
            ax2.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # 3. MLMスコア分布
            ax3 = fig.add_subplot(gs[0, 2])
            pathogenic_scores = self.results_df[self.results_df["pathogenic"] == 1][
                "mlm_score"
            ]
            benign_scores = self.results_df[self.results_df["pathogenic"] == 0][
                "mlm_score"
            ]
            ax3.hist(
                pathogenic_scores, alpha=0.7, label="Pathogenic", bins=20, color="red"
            )
            ax3.hist(benign_scores, alpha=0.7, label="Benign", bins=20, color="blue")
            ax3.set_title("MLM Score Distribution")
            ax3.legend()

            # 4. 類似度分布
            ax4 = fig.add_subplot(gs[1, 0])
            pathogenic_sim = self.results_df[self.results_df["pathogenic"] == 1][
                "cosine_similarity"
            ]
            benign_sim = self.results_df[self.results_df["pathogenic"] == 0][
                "cosine_similarity"
            ]
            ax4.hist(
                pathogenic_sim, alpha=0.7, label="Pathogenic", bins=20, color="red"
            )
            ax4.hist(benign_sim, alpha=0.7, label="Benign", bins=20, color="blue")
            ax4.set_title("Similarity Distribution")
            ax4.legend()

            # 5. 散布図
            ax5 = fig.add_subplot(gs[1, 1])
            pathogenic_data = self.results_df[self.results_df["pathogenic"] == 1]
            benign_data = self.results_df[self.results_df["pathogenic"] == 0]
            ax5.scatter(
                pathogenic_data["mlm_score"],
                pathogenic_data["cosine_similarity"],
                alpha=0.6,
                c="red",
                label="Pathogenic",
                s=10,
            )
            ax5.scatter(
                benign_data["mlm_score"],
                benign_data["cosine_similarity"],
                alpha=0.6,
                c="blue",
                label="Benign",
                s=10,
            )
            ax5.set_xlabel("MLM Score")
            ax5.set_ylabel("Cosine Similarity")
            ax5.set_title("MLM vs Similarity")
            ax5.legend()

            # 6. 統計情報
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis("off")

            stats_text = f"""
            BERT Model Performance Summary
            
            Overall Accuracy: {accuracy:.3f}
            Precision: {precision:.3f}
            Recall: {recall:.3f}
            F1-Score: {f1:.3f}
            AUC-ROC: {auc:.3f}
            
            Dataset Statistics:
            Total Variants: {len(self.results_df)}
            Pathogenic: {self.results_df["pathogenic"].sum()}
            Benign: {len(self.results_df) - self.results_df["pathogenic"].sum()}
            
            MLM Score Means:
            Pathogenic: {pathogenic_scores.mean():.3f}
            Benign: {benign_scores.mean():.3f}
            """

            ax6.text(
                0.1,
                0.9,
                stats_text,
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )

        plt.suptitle("BERT ClinVar Evaluation Dashboard", fontsize=16, y=0.98)
        self._save_plot("bert_summary_dashboard")

    def generate_all_visualizations(self):
        """全ての可視化を生成"""
        self.logger.info("Generating all BERT visualizations")

        self.plot_confusion_matrix()
        self.plot_performance_metrics()

        # CSV固有の可視化
        if self.results_df is not None:
            self.plot_mlm_score_distribution()
            self.plot_similarity_distribution()

        self.create_summary_dashboard()

        self.logger.info(f"All BERT visualizations saved to {self.output_dir}")
        self.logger.info(f"Generated {len(self.generated_files)} files")

    def create_html_report(self):
        """HTML形式の総合レポートを作成"""
        self.logger.info("Creating BERT HTML report")

        html_content = self._create_html_header(
            "BERT ClinVar Pathogenicity Prediction Evaluation"
        )

        if self.results_df is not None:
            predictions = (self.results_df["mlm_score"] > 0).astype(int)
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            accuracy = accuracy_score(self.results_df["pathogenic"], predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.results_df["pathogenic"], predictions, average="binary"
            )
            try:
                auc = roc_auc_score(
                    self.results_df["pathogenic"], self.results_df["mlm_score"]
                )
            except (ValueError, RuntimeError):
                auc = 0.5
        else:
            accuracy = self.results["accuracy"]
            precision = self.results["precision"]
            recall = self.results["recall"]
            f1 = self.results["f1_score"]
            auc = self.results["auc"]

        html_content += f"""
            <h2>BERT Model Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{accuracy:.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{precision:.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{recall:.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{f1:.3f}</div>
                    <div class="metric-label">F1-Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{auc:.3f}</div>
                    <div class="metric-label">AUC-ROC</div>
                </div>
            </div>
            
            <h2 class="section-title">BERT Summary Dashboard</h2>
            <div class="image-container">
                <img src="bert_summary_dashboard.png" alt="BERT Summary Dashboard">
            </div>
            
            <h2 class="section-title">Detailed BERT Analysis</h2>
            
            <h3>Confusion Matrix</h3>
            <div class="image-container">
                <img src="bert_confusion_matrix.png" alt="BERT Confusion Matrix">
            </div>
            
            <h3>Performance Metrics</h3>
            <div class="image-container">
                <img src="bert_performance_metrics.png" alt="BERT Performance Metrics">
            </div>
        """

        # CSV固有の可視化を追加
        if self.results_df is not None:
            html_content += """
            <h3>MLM Score Distribution</h3>
            <div class="image-container">
                <img src="bert_mlm_score_distribution.png" alt="MLM Score Distribution">
            </div>
            
            <h3>Sequence Similarity Distribution</h3>
            <div class="image-container">
                <img src="bert_similarity_distribution.png" alt="Similarity Distribution">
            </div>
            """

        html_content += self._create_html_footer()

        html_file = self.output_dir / "bert_evaluation_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.generated_files.append(html_file)
        self.logger.info("BERT HTML report created: bert_evaluation_report.html")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize BERT ClinVar evaluation results"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to evaluation results JSON or CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bert_clinvar_visualization_results",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--html_report", action="store_true", help="Generate HTML report"
    )

    args = parser.parse_args()

    try:
        visualizer = BERTClinVarVisualizationGenerator(
            args.results_file, args.output_dir
        )

        # 全ての可視化を生成
        visualizer.generate_all_visualizations()

        # HTMLレポートを生成
        if args.html_report:
            visualizer.create_html_report()

        logger.info("BERT visualization completed successfully")

        # 生成されたファイル一覧をログ出力
        generated_files = visualizer.get_generated_files()
        logger.info(f"Generated {len(generated_files)} visualization files:")
        for file_path in generated_files:
            logger.info(f"  - {file_path}")

    except Exception as e:
        logger.error(f"BERT visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
