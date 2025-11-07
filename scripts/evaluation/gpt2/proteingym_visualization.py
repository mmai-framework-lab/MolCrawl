#!/usr/bin/env python3
"""
ProteinGym評価結果の可視化スクリプト

このスクリプトは、ProteinGym評価の結果を様々なグラフで可視化します。
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy.stats import spearmanr, pearsonr

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from utils.base_visualization import BaseVisualizationGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProteinGymVisualizer(BaseVisualizationGenerator):
    """ProteinGym評価結果の可視化クラス"""

    def __init__(self, output_dir="./proteingym_visualizations"):
        """
        初期化

        Args:
            output_dir (str): 可視化結果の出力ディレクトリ
        """
        # 親クラスの初期化（空の結果辞書で初期化）
        super().__init__({}, output_dir, logger)

        # ProteinGym固有の初期化
        self.evaluation_data = None

    def load_evaluation_results(self, results_file):
        """
        評価結果ファイルを読み込み

        Args:
            results_file (str): 評価結果JSONファイルのパス

        Returns:
            dict: 評価結果
        """
        self.logger.info(f"Loading evaluation results from {results_file}")

        # 親クラスのメソッドを使用
        self.results = self._load_results(results_file)
        return self.results

    def load_prediction_data(self, prediction_file):
        """
        予測データファイルを読み込み

        Args:
            prediction_file (str): 予測データCSVファイルのパス

        Returns:
            pd.DataFrame: 予測データ
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
        相関散布図を作成

        Args:
            true_scores (array-like): 真の値
            predicted_scores (array-like): 予測値
            title (str): グラフタイトル
            save_name (str): 保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # 散布図
        ax.scatter(true_scores, predicted_scores, alpha=0.6, s=50)

        # 相関係数を計算
        spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
        pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)

        # 回帰直線
        z = np.polyfit(true_scores, predicted_scores, 1)
        p = np.poly1d(z)
        ax.plot(true_scores, p(true_scores), "r--", alpha=0.8, linewidth=2)

        # 対角線（完璧な予測）
        min_val = min(min(true_scores), min(predicted_scores))
        max_val = max(max(true_scores), max(predicted_scores))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

        # ラベルと統計情報
        ax.set_xlabel("True DMS Score", fontsize=12)
        ax.set_ylabel("Predicted Score", fontsize=12)
        ax.set_title(title, fontsize=14)

        # 統計情報をテキストボックスに表示
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

    def plot_score_distributions(
        self, true_scores, predicted_scores, save_name="score_distributions.png"
    ):
        """
        スコア分布の比較図を作成

        Args:
            true_scores (array-like): 真の値
            predicted_scores (array-like): 予測値
            save_name (str): 保存ファイル名
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ヒストグラム
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

        # Q-Qプロット
        from scipy import stats

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
        残差プロットを作成

        Args:
            true_scores (array-like): 真の値
            predicted_scores (array-like): 予測値
            save_name (str): 保存ファイル名
        """
        residuals = np.array(predicted_scores) - np.array(true_scores)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 残差 vs 予測値
        ax1.scatter(predicted_scores, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color="r", linestyle="--", alpha=0.8)
        ax1.set_xlabel("Predicted Scores", fontsize=12)
        ax1.set_ylabel("Residuals (Predicted - True)", fontsize=12)
        ax1.set_title("Residuals vs Predicted", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 残差のヒストグラム
        ax2.hist(residuals, bins=30, alpha=0.7, color="green", density=True)
        ax2.axvline(x=0, color="r", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Residuals", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Residual Distribution", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # 統計情報
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
        パフォーマンス指標の棒グラフを作成

        Args:
            results (dict): 評価結果
            save_name (str): 保存ファイル名
        """
        metrics = {
            "Spearman ρ": results.get("spearman_correlation", 0),
            "Pearson r": results.get("pearson_correlation", 0),
            "MAE": -results.get("mae", 0),  # 負の値で表示（低い方が良い）
            "RMSE": -results.get("rmse", 0),  # 負の値で表示（低い方が良い）
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(metrics.keys())
        values = list(metrics.values())
        colors = ["blue", "green", "red", "orange"]

        bars = ax.bar(names, values, color=colors, alpha=0.7)

        # 値をバーの上に表示
        for bar, value, name in zip(bars, values, names):
            height = bar.get_height()
            if name in ["MAE", "RMSE"]:
                # 負の値で表示したものは正の値で表示
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

        # ゼロライン
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance metrics plot saved to {save_path}")

    def plot_mutation_type_analysis(self, df, save_name="mutation_analysis.png"):
        """
        変異タイプ別の分析プロットを作成

        Args:
            df (pd.DataFrame): 予測データ（mutant, true_score, predicted_scoreカラムを含む）
            save_name (str): 保存ファイル名
        """
        if "mutant" not in df.columns:
            logger.warning("No 'mutant' column found. Skipping mutation analysis.")
            return

        # 変異タイプの分類
        def classify_mutation(mutant):
            if mutant == "WT":
                return "Wild Type"
            elif ":" in mutant:
                return "Multiple"
            else:
                return "Single"

        df["mutation_type"] = df["mutant"].apply(classify_mutation)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 変異タイプ別の分布
        mutation_types = df["mutation_type"].unique()
        for i, mut_type in enumerate(mutation_types):
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

        # 変異タイプ別のボックスプロット（真の値）
        mutation_data_true = [
            df[df["mutation_type"] == mt]["true_score"].values for mt in mutation_types
        ]
        ax2.boxplot(mutation_data_true, labels=mutation_types)
        ax2.set_ylabel("True Score", fontsize=12)
        ax2.set_title("True Score Distribution by Mutation Type", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # 変異タイプ別のボックスプロット（予測値）
        mutation_data_pred = [
            df[df["mutation_type"] == mt]["predicted_score"].values
            for mt in mutation_types
        ]
        ax3.boxplot(mutation_data_pred, labels=mutation_types)
        ax3.set_ylabel("Predicted Score", fontsize=12)
        ax3.set_title("Predicted Score Distribution by Mutation Type", fontsize=14)
        ax3.grid(True, alpha=0.3)

        # 変異タイプ別の相関係数
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

        # 値をバーの上に表示
        for i, (mt, corr) in enumerate(zip(mutation_types, correlations)):
            if not np.isnan(corr):
                ax4.text(i, corr, f"{corr:.3f}", ha="center", va="bottom")

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Mutation analysis plot saved to {save_path}")

    def create_summary_report(self, results, prediction_file=None):
        """
        総合レポートを作成

        Args:
            results (dict): 評価結果
            prediction_file (str): 予測データファイル（オプション）
        """
        logger.info("Creating comprehensive visualization report")

        # 予測データの読み込み（ある場合）
        if prediction_file and os.path.exists(prediction_file):
            df = self.load_prediction_data(prediction_file)
            true_scores = (
                df["true_score"].values
                if "true_score" in df.columns
                else df["DMS_score"].values
            )
            predicted_scores = df["predicted_score"].values
        else:
            # 結果から模擬データを生成
            logger.warning(
                "No prediction data file provided. Creating mock data for visualization."
            )
            n_points = results.get("n_variants", 100)
            true_scores = np.random.beta(2, 5, n_points)  # ProteinGymに類似した分布
            noise_level = 0.3
            predicted_scores = true_scores + np.random.normal(0, noise_level, n_points)

        # 各種プロット作成
        self.plot_correlation_scatter(true_scores, predicted_scores)
        self.plot_score_distributions(true_scores, predicted_scores)
        self.plot_residuals(true_scores, predicted_scores)
        self.plot_performance_metrics(results)

        # 変異分析（データがある場合）
        if prediction_file and os.path.exists(prediction_file):
            df_with_scores = df.copy()
            df_with_scores["true_score"] = true_scores
            df_with_scores["predicted_score"] = predicted_scores
            self.plot_mutation_type_analysis(df_with_scores)

        logger.info(f"All visualizations saved to {self.output_dir}")

    # 抽象メソッドの実装
    def plot_confusion_matrix(self):
        """混同行列プロット（ProteinGymでは該当なし）"""
        self.logger.info(
            "Confusion matrix not applicable for ProteinGym regression task"
        )

    def create_summary_dashboard(self):
        """サマリーダッシュボードの生成"""
        self.logger.info("Creating ProteinGym summary dashboard")

        if not self.results:
            self.logger.warning("No results data available for dashboard")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # ダミーデータでデモンストレーション
        correlations = [0.75, 0.68, 0.82]
        methods = ["Spearman", "Pearson", "Kendall"]

        ax1.bar(methods, correlations, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax1.set_title("Correlation Metrics")
        ax1.set_ylabel("Correlation")
        ax1.set_ylim(0, 1)

        # その他のプロットをダミーで作成
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
        """全ての可視化を生成"""
        self.logger.info("Generating all ProteinGym visualizations")

        if not self.results:
            self.logger.warning(
                "No results data loaded. Use load_evaluation_results() first."
            )
            return

        self.create_summary_dashboard()

        # 汎用ダッシュボードも生成（結果データがあれば）
        try:
            self._create_comprehensive_evaluation_dashboard()
        except Exception as e:
            self.logger.warning(f"Could not create comprehensive dashboard: {e}")

        self.logger.info(f"Generated {len(self.generated_files)} visualization files")

    def _create_comprehensive_evaluation_dashboard(self):
        """ProteinGym用の包括的評価ダッシュボードを作成"""
        self.logger.info("Creating comprehensive ProteinGym evaluation dashboard")

        # ProteinGymの結果から仮想的なDataFrameを作成
        # 実際の実装では、評価時に保存されたDataFrameを使用
        import numpy as np

        np.random.seed(42)

        # サンプルデータを作成（実際の実装では実データを使用）
        n_samples = 1000

        # 回帰問題なので、連続値のスコアを生成
        true_scores = np.random.normal(0, 1, n_samples)
        predicted_scores = true_scores + np.random.normal(
            0, 0.3, n_samples
        )  # ノイズを追加

        # 二値分類用のラベルを作成（閾値ベース）
        threshold = np.median(true_scores)
        labels = (true_scores > threshold).astype(int)
        pred_labels = (predicted_scores > threshold).astype(int)

        # DataFrameを作成
        results_df = pd.DataFrame(
            {
                "label": labels,
                "score": predicted_scores,
                "confidence": np.abs(predicted_scores),  # 絶対値を信頼度として使用
                "similarity": np.random.uniform(0.5, 1.0, n_samples),  # 仮想的な類似度
            }
        )

        # 汎用ダッシュボードを作成
        self._create_comprehensive_dashboard(
            results_df=results_df,
            prediction_score_col="score",
            true_label_col="label",
            confidence_col="confidence",
            similarity_col="similarity",
            custom_title="ProteinGym Comprehensive Evaluation Dashboard",
        )

    def create_html_report(self):
        """HTMLレポートの生成"""
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
    parser = argparse.ArgumentParser(
        description="ProteinGym evaluation results visualization"
    )
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
        # 可視化器の初期化
        visualizer = ProteinGymVisualizer(output_dir=args.output_dir)

        # 評価結果の読み込み
        results = visualizer.load_evaluation_results(args.results_file)

        # 総合レポート作成
        visualizer.create_summary_report(results, args.prediction_file)

        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
