#!/usr/bin/env python3
"""
Molecule Natural Language Validator - Visualization Module

BaseVisualizationGeneratorを継承したMolecule NL検証結果の可視化クラス
Performance MetricsとConfusion Matrixを含む包括的なダッシュボードを生成
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# プロジェクトルートを追加

from molcrawl.utils.base_visualization import BaseVisualizationGenerator


class MoleculeNLValidatorVisualization(BaseVisualizationGenerator):
    """
    Molecule NL検証結果の可視化クラス

    BaseVisualizationGeneratorを継承し、分子自然言語モデルの
    検証結果を包括的に可視化します。

    Features:
        - Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
        - Confusion Matrix with TP, FP, TN, FN
        - Score distribution by label
        - ROC curve and PR curve
        - Comprehensive HTML report
    """

    def __init__(self, results_file, output_dir, metrics=None):
        """
        初期化

        Args:
            results_file (str): 検証結果CSVファイルのパス
            output_dir (str): 出力ディレクトリ
            metrics (dict): メトリクス情報（オプション）
        """
        # 親クラスの初期化（results_sourceとしてresults_fileを渡す）
        super().__init__(results_source=results_file)

        self.results_file = results_file
        self.output_dir = output_dir
        self.metrics = metrics

        # データの読み込み
        self.df = pd.read_csv(results_file)

        # メトリクスファイルが存在する場合は読み込み
        if metrics is None:
            metrics_path = os.path.join(os.path.dirname(results_file), "validation_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)

        print("✅ MoleculeNLValidatorVisualization initialized")
        print(f"   Results: {len(self.df)} samples")

    def create_summary_dashboard(self, output_path):
        """
        要約ダッシュボードの作成（BaseVisualizationGenerator要求メソッド）

        Args:
            output_path (str): 出力ファイルパス
        """
        # 包括的ダッシュボードをサマリーとして使用
        self._create_comprehensive_dashboard(
            df=self.df,
            output_path=output_path,
            title="Molecule NL Validation - Summary Dashboard",
        )

    def plot_performance_metrics(self, output_path):
        """
        パフォーマンスメトリクスプロット（BaseVisualizationGenerator要求メソッド）

        Args:
            output_path (str): 出力ファイルパス
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_performance_metrics(ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, output_path):
        """
        混同行列プロット（BaseVisualizationGenerator要求メソッド）

        Args:
            output_path (str): 出力ファイルパス
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_confusion_matrix(
            self.df["true_label"].values,
            self.df["predicted_label"].values,
            ax,
            "Confusion Matrix",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_all_visualizations(self):
        """すべての可視化を生成"""
        print("🎨 Generating comprehensive visualizations...")

        # BaseVisualizationGeneratorの包括的ダッシュボードを使用
        self._create_comprehensive_dashboard(
            df=self.df,
            output_path=os.path.join(self.output_dir, "comprehensive_dashboard.png"),
            title="Molecule NL Validation - Comprehensive Dashboard",
        )

        # ドメイン固有の追加プロットを生成
        self._create_domain_specific_plots()

        print("✅ All visualizations generated successfully")

    def _prepare_data_for_dashboard(self, df):
        """
        ダッシュボード用のデータを準備（BaseVisualizationGenerator要求形式）

        Args:
            df (pd.DataFrame): 検証結果データフレーム

        Returns:
            dict: ダッシュボード用データ
        """
        # BaseVisualizationGeneratorが期待する形式にデータを変換
        data = {
            "y_true": df["true_label"].values,
            "y_pred": df["predicted_label"].values,
            "y_scores": df["prediction_score"].values,
            "confidence": df["prediction_score"].values,  # スコアを信頼度として使用
        }

        return data

    def _create_comprehensive_dashboard(self, df, output_path, title):
        """
        包括的ダッシュボードの作成

        BaseVisualizationGeneratorのダッシュボード機能を活用し、
        6パネルの詳細な可視化を生成
        """
        # データの準備
        dashboard_data = self._prepare_data_for_dashboard(df)

        # 親クラスのダッシュボード生成メソッドを呼び出し
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        # 1. Score Distribution (左上)
        self._plot_score_distribution(
            dashboard_data["y_scores"],
            dashboard_data["y_true"],
            axes[0, 0],
            "Prediction Score Distribution by True Label",
        )

        # 2. Performance Metrics (右上)
        self._plot_performance_metrics(axes[0, 1])

        # 3. Confusion Matrix (中央左)
        self._plot_confusion_matrix(
            dashboard_data["y_true"],
            dashboard_data["y_pred"],
            axes[1, 0],
            "Confusion Matrix",
        )

        # 4. ROC Curve (中央右)
        self._plot_roc_curve(
            dashboard_data["y_true"],
            dashboard_data["y_scores"],
            axes[1, 1],
            "ROC Curve",
        )

        # 5. Precision-Recall Curve (左下)
        self._plot_pr_curve(
            dashboard_data["y_true"],
            dashboard_data["y_scores"],
            axes[2, 0],
            "Precision-Recall Curve",
        )

        # 6. Label Distribution (右下)
        self._plot_label_distribution(dashboard_data["y_true"], axes[2, 1], "Dataset Label Distribution")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   ✅ Comprehensive dashboard saved: {output_path}")

    def _plot_performance_metrics(self, ax):
        """Performance Metricsの棒グラフ"""
        if self.metrics is None:
            ax.text(0.5, 0.5, "Metrics not available", ha="center", va="center")
            ax.set_title("Performance Metrics")
            return

        pm = self.metrics["performance_metrics"]

        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC",
            "Sensitivity",
            "Specificity",
        ]
        metric_values = [
            pm["accuracy"],
            pm["precision"],
            pm["recall"],
            pm["f1_score"],
            pm["roc_auc"],
            pm["pr_auc"],
            pm["sensitivity"],
            pm["specificity"],
        ]

        colors = ["#3498db" if v >= 0.7 else "#e74c3c" if v < 0.5 else "#f39c12" for v in metric_values]

        bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7)

        # 値をバーの横に表示
        for _i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Score", fontsize=10)
        ax.set_xlim(0, 1.1)
        ax.set_title("Performance Metrics", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    def _plot_label_distribution(self, y_true, ax, title):
        """ラベル分布の可視化"""
        unique, counts = np.unique(y_true, return_counts=True)

        colors = ["#3498db", "#e74c3c"]
        bars = ax.bar(
            ["Negative (0)", "Positive (1)"],
            counts,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # パーセンテージを追加
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    def _create_domain_specific_plots(self):
        """ドメイン固有の追加プロット"""
        # 1. Detailed Confusion Matrix with annotations
        self._create_detailed_confusion_matrix()

        # 2. Prediction confidence histogram
        self._create_confidence_histogram()

        # 3. Error analysis plot
        self._create_error_analysis_plot()

    def _create_detailed_confusion_matrix(self):
        """詳細なConfusion Matrixヒートマップ"""
        if self.metrics is None:
            return

        cm = self.metrics["confusion_matrix"]
        cm_array = np.array(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ]
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            cbar_kws={"label": "Count"},
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
            linewidths=2,
            linecolor="black",
        )

        # パーセンテージを追加
        total = cm_array.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm_array[i, j] / total) * 100
                ax.text(
                    j + 0.5,
                    i + 0.7,
                    f"({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="darkgreen",
                )

        ax.set_title("Detailed Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "detailed_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   ✅ Detailed confusion matrix saved: {output_path}")

    def _create_confidence_histogram(self):
        """予測信頼度のヒストグラム"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 正解・不正解別の信頼度分布
        correct = self.df[self.df["true_label"] == self.df["predicted_label"]]
        incorrect = self.df[self.df["true_label"] != self.df["predicted_label"]]

        axes[0].hist(
            correct["prediction_score"],
            bins=30,
            alpha=0.7,
            label="Correct Predictions",
            color="green",
            edgecolor="black",
        )
        axes[0].hist(
            incorrect["prediction_score"],
            bins=30,
            alpha=0.7,
            label="Incorrect Predictions",
            color="red",
            edgecolor="black",
        )
        axes[0].set_xlabel("Prediction Score", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Confidence Distribution by Correctness", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ラベル別の信頼度分布
        pos_samples = self.df[self.df["true_label"] == 1]
        neg_samples = self.df[self.df["true_label"] == 0]

        axes[1].hist(
            pos_samples["prediction_score"],
            bins=30,
            alpha=0.7,
            label="Positive Samples",
            color="#e74c3c",
            edgecolor="black",
        )
        axes[1].hist(
            neg_samples["prediction_score"],
            bins=30,
            alpha=0.7,
            label="Negative Samples",
            color="#3498db",
            edgecolor="black",
        )
        axes[1].set_xlabel("Prediction Score", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title("Confidence Distribution by True Label", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confidence_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   ✅ Confidence histogram saved: {output_path}")

    def _create_error_analysis_plot(self):
        """エラー分析プロット"""
        # False PositivesとFalse Negativesの分析
        fp = self.df[(self.df["true_label"] == 0) & (self.df["predicted_label"] == 1)]
        fn = self.df[(self.df["true_label"] == 1) & (self.df["predicted_label"] == 0)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # False Positivesのスコア分布
        if len(fp) > 0:
            axes[0].hist(
                fp["prediction_score"],
                bins=20,
                color="orange",
                alpha=0.7,
                edgecolor="black",
            )
            axes[0].axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold")
            axes[0].set_xlabel("Prediction Score", fontsize=11)
            axes[0].set_ylabel("Count", fontsize=11)
            axes[0].set_title(f"False Positives (n={len(fp)})", fontsize=12, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No False Positives", ha="center", va="center", fontsize=14)
            axes[0].set_title("False Positives", fontsize=12, fontweight="bold")

        # False Negativesのスコア分布
        if len(fn) > 0:
            axes[1].hist(
                fn["prediction_score"],
                bins=20,
                color="purple",
                alpha=0.7,
                edgecolor="black",
            )
            axes[1].axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold")
            axes[1].set_xlabel("Prediction Score", fontsize=11)
            axes[1].set_ylabel("Count", fontsize=11)
            axes[1].set_title(f"False Negatives (n={len(fn)})", fontsize=12, fontweight="bold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No False Negatives", ha="center", va="center", fontsize=14)
            axes[1].set_title("False Negatives", fontsize=12, fontweight="bold")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "error_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   ✅ Error analysis plot saved: {output_path}")

    def create_html_report(self):
        """HTMLレポートの生成"""
        html_content = self._generate_html_report()

        output_path = os.path.join(self.output_dir, "validation_report.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"   ✅ HTML report saved: {output_path}")

    def _generate_html_report(self):
        """HTML形式のレポート生成"""
        # メトリクスの取得
        if self.metrics:
            pm = self.metrics["performance_metrics"]
            cm = self.metrics["confusion_matrix"]
            di = self.metrics["dataset_info"]
        else:
            pm = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "roc_auc": 0,
                "pr_auc": 0,
                "sensitivity": 0,
                "specificity": 0,
            }
            cm = {
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
            }
            di = {
                "total_samples": len(self.df),
                "positive_samples": 0,
                "negative_samples": 0,
            }

        # F1スコアに基づく評価
        f1 = pm["f1_score"]
        if f1 >= 0.8:
            assessment = "🟢 Excellent"
            assessment_color = "#27ae60"
        elif f1 >= 0.6:
            assessment = "🟡 Good"
            assessment_color = "#f39c12"
        elif f1 >= 0.4:
            assessment = "🟠 Moderate"
            assessment_color = "#e67e22"
        else:
            assessment = "🔴 Poor"
            assessment_color = "#e74c3c"

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Molecule NL Validation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-name {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
            max-width: 600px;
        }}
        .cm-cell {{
            padding: 25px;
            text-align: center;
            border-radius: 10px;
            font-weight: bold;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .cm-tp {{ background-color: #27ae60; color: white; }}
        .cm-fp {{ background-color: #e74c3c; color: white; }}
        .cm-tn {{ background-color: #3498db; color: white; }}
        .cm-fn {{ background-color: #f39c12; color: white; }}
        .assessment {{
            background-color: {assessment_color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 30px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .info-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Molecule Natural Language Model<br>Validation Report</h1>

        <div class="info-box">
            <p><strong>🕐 Validation Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>📊 Total Samples:</strong> {di["total_samples"]}</p>
            <p><strong>✅ Positive Samples:</strong> {di["positive_samples"]} ({di["positive_samples"] / di["total_samples"] * 100:.1f}%)</p>
            <p><strong>❌ Negative Samples:</strong> {di["negative_samples"]} ({di["negative_samples"] / di["total_samples"] * 100:.1f}%)</p>
        </div>

        <div class="assessment">
            Overall Assessment: {assessment}<br>
            F1-Score: {f1:.4f}
        </div>

        <h2>📊 Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-name">Accuracy</div>
                <div class="metric-value">{pm["accuracy"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Precision</div>
                <div class="metric-value">{pm["precision"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Recall</div>
                <div class="metric-value">{pm["recall"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">F1-Score</div>
                <div class="metric-value">{pm["f1_score"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">ROC-AUC</div>
                <div class="metric-value">{pm["roc_auc"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">PR-AUC</div>
                <div class="metric-value">{pm["pr_auc"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Sensitivity</div>
                <div class="metric-value">{pm["sensitivity"]:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Specificity</div>
                <div class="metric-value">{pm["specificity"]:.3f}</div>
            </div>
        </div>

        <h2>📋 Confusion Matrix</h2>
        <div class="confusion-matrix">
            <div class="cm-cell cm-tn">
                <div>True Negative</div>
                <div style="font-size: 28px; margin-top: 10px;">{cm["true_negative"]}</div>
            </div>
            <div class="cm-cell cm-fp">
                <div>False Positive</div>
                <div style="font-size: 28px; margin-top: 10px;">{cm["false_positive"]}</div>
            </div>
            <div class="cm-cell cm-fn">
                <div>False Negative</div>
                <div style="font-size: 28px; margin-top: 10px;">{cm["false_negative"]}</div>
            </div>
            <div class="cm-cell cm-tp">
                <div>True Positive</div>
                <div style="font-size: 28px; margin-top: 10px;">{cm["true_positive"]}</div>
            </div>
        </div>

        <h2>🎨 Visualizations</h2>

        <div class="visualization">
            <h3>Comprehensive Dashboard</h3>
            <img src="comprehensive_dashboard.png" alt="Comprehensive Dashboard">
        </div>

        <div class="visualization">
            <h3>Detailed Confusion Matrix</h3>
            <img src="detailed_confusion_matrix.png" alt="Detailed Confusion Matrix">
        </div>

        <div class="visualization">
            <h3>Confidence Distribution</h3>
            <img src="confidence_histogram.png" alt="Confidence Distribution">
        </div>

        <div class="visualization">
            <h3>Error Analysis</h3>
            <img src="error_analysis.png" alt="Error Analysis">
        </div>

        <h2>💡 Key Insights</h2>
        <div class="info-box">
            <ul>
                <li><strong>Accuracy:</strong> Overall correctness of predictions</li>
                <li><strong>Precision:</strong> Accuracy of positive predictions</li>
                <li><strong>Recall (Sensitivity):</strong> Ability to find all positive cases</li>
                <li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall</li>
                <li><strong>ROC-AUC:</strong> Overall discrimination ability</li>
                <li><strong>PR-AUC:</strong> Precision-Recall trade-off performance</li>
                <li><strong>Specificity:</strong> Ability to correctly identify negative cases</li>
            </ul>
        </div>

        <div class="footer">
            Generated by Molecule NL Validator | Powered by BaseVisualizationGenerator
        </div>
    </div>
</body>
</html>
        """

        return html


def main():
    """スタンドアローン実行用メインルーチン"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations for Molecule NL validation results")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to validation results CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_file)

    print("🎨 Starting Molecule NL Validator Visualization...")
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")

    # 可視化の生成
    visualizer = MoleculeNLValidatorVisualization(results_file=args.results_file, output_dir=args.output_dir)

    visualizer.generate_all_visualizations()
    visualizer.create_html_report()

    print("✅ Visualization generation completed!")


if __name__ == "__main__":
    main()
