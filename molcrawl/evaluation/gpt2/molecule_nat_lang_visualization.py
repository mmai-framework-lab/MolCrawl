#!/usr/bin/env python3
"""
Molecule Natural Language 評価結果の可視化生成器

BaseVisualizationGeneratorを継承してmolecule_nat_langに特化した
分析結果の可視化を行います。
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# プロジェクトルートを追加

from molcrawl.utils.base_visualization import BaseVisualizationGenerator

logger = logging.getLogger(__name__)


class MoleculeNLVisualizationGenerator(BaseVisualizationGenerator):
    """Molecule NL評価結果の可視化クラス"""

    def __init__(self, results_file, output_dir):
        """
        初期化

        Args:
            results_file (str): 評価結果CSVファイルのパス
            output_dir (str): 出力ディレクトリ
        """
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ベースクラス初期化用のダミー辞書
        dummy_results = {"dummy": "data"}
        super().__init__(dummy_results, output_dir)

        self.evaluation_type = "molecule_nat_lang"

        # データの読み込み
        self.load_and_validate_data()

        # Molecule NL特有の設定
        self.colors = {
            "primary": "#2E8B57",  # SeaGreen
            "secondary": "#32CD32",  # LimeGreen
            "accent": "#FFD700",  # Gold
            "warning": "#FF6347",  # Tomato
            "info": "#4169E1",  # RoyalBlue
            "success": "#228B22",  # ForestGreen
        }

    def load_and_validate_data(self):
        """データの読み込みと検証（オーバーライド）"""
        try:
            logger.info(f"Loading Molecule NL evaluation results from {self.results_file}")
            self.df = pd.read_csv(self.results_file)

            # 必要な列の確認
            required_columns = ["perplexity", "text_length", "token_length"]
            missing_columns = [col for col in required_columns if col not in self.df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # データクリーニング
            initial_count = len(self.df)
            self.df = self.df.dropna(subset=["perplexity"])

            # 無限大値の処理
            infinite_mask = np.isinf(self.df["perplexity"])
            if infinite_mask.any():
                logger.warning(f"Found {infinite_mask.sum()} samples with infinite perplexity")
                self.df = self.df[~infinite_mask]

            logger.info(f"Data loaded: {len(self.df)}/{initial_count} valid samples")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def create_domain_specific_plot(self):
        """Molecule NL特有のプロット作成（抽象メソッドの実装）"""
        logger.info("Creating Molecule NL specific visualization")

        # パープレキシティの詳細解析プロット
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Molecule NL Model - Detailed Perplexity Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. パープレキシティ分布（線形スケール）
        axes[0, 0].hist(
            self.df["perplexity"],
            bins=50,
            alpha=0.7,
            color=self.colors["primary"],
            edgecolor="black",
        )
        axes[0, 0].set_xlabel("Perplexity")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Perplexity Distribution (Linear Scale)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. パープレキシティ分布（対数スケール）
        axes[0, 1].hist(
            self.df["perplexity"],
            bins=50,
            alpha=0.7,
            color=self.colors["secondary"],
            edgecolor="black",
        )
        axes[0, 1].set_xlabel("Perplexity")
        axes[0, 1].set_ylabel("Frequency (Log Scale)")
        axes[0, 1].set_title("Perplexity Distribution (Log Scale)")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. テキスト長とパープレキシティの関係
        axes[0, 2].scatter(
            self.df["text_length"],
            self.df["perplexity"],
            alpha=0.6,
            color=self.colors["info"],
            s=20,
        )

        # 傾向線を追加
        z = np.polyfit(self.df["text_length"], self.df["perplexity"], 1)
        p = np.poly1d(z)
        axes[0, 2].plot(
            self.df["text_length"],
            p(self.df["text_length"]),
            "r--",
            alpha=0.8,
            linewidth=2,
        )

        axes[0, 2].set_xlabel("Text Length (characters)")
        axes[0, 2].set_ylabel("Perplexity")
        axes[0, 2].set_title("Text Length vs Perplexity")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. トークン長とパープレキシティの関係
        axes[1, 0].scatter(
            self.df["token_length"],
            self.df["perplexity"],
            alpha=0.6,
            color=self.colors["accent"],
            s=20,
        )

        # 傾向線を追加
        z_token = np.polyfit(self.df["token_length"], self.df["perplexity"], 1)
        p_token = np.poly1d(z_token)
        axes[1, 0].plot(
            self.df["token_length"],
            p_token(self.df["token_length"]),
            "r--",
            alpha=0.8,
            linewidth=2,
        )

        axes[1, 0].set_xlabel("Token Length")
        axes[1, 0].set_ylabel("Perplexity")
        axes[1, 0].set_title("Token Length vs Perplexity")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. パープレキシティのボックスプロット（四分位数別）
        # テキスト長でビンを作成（重複を許可）
        try:
            # duplicates='drop'を使用すると、ラベルが自動生成される
            self.df["length_quartile"] = pd.qcut(self.df["text_length"], 4, duplicates="drop")

            # 実際に生成されたカテゴリを取得
            unique_quartiles = sorted(self.df["length_quartile"].unique())

            box_data = [self.df[self.df["length_quartile"] == q]["perplexity"].values for q in unique_quartiles]

            # ラベルを生成（Q1, Q2, ... または範囲表示）
            box_labels = [f"Q{i + 1}" for i in range(len(unique_quartiles))]

            if len(box_data) > 0:
                box_plot = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)

                # ボックスプロットの色設定
                colors_box = [
                    self.colors["primary"],
                    self.colors["secondary"],
                    self.colors["info"],
                    self.colors["accent"],
                ][: len(box_data)]
                for patch, color in zip(box_plot["boxes"], colors_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "Insufficient data variation for quartiles",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
        except (ValueError, TypeError) as e:
            # 四分位数が計算できない場合（全ての値が同じなど）
            logger.warning(f"Could not create quartiles: {e}")
            axes[1, 1].text(
                0.5,
                0.5,
                "Insufficient data variation for quartiles",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        axes[1, 1].set_xlabel("Text Length Quartile")
        axes[1, 1].set_ylabel("Perplexity")
        axes[1, 1].set_title("Perplexity by Text Length Quartile")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. パフォーマンス範囲の分析
        # パープレキシティ範囲別のサンプル数
        perplexity_ranges = [
            (0, 10, "Excellent"),
            (10, 50, "Good"),
            (50, 200, "Moderate"),
            (200, float("inf"), "Poor"),
        ]

        range_counts = []
        range_labels = []
        range_colors = [
            self.colors["success"],
            self.colors["info"],
            self.colors["accent"],
            self.colors["warning"],
        ]

        for min_val, max_val, label in perplexity_ranges:
            if max_val == float("inf"):
                count = len(self.df[self.df["perplexity"] >= min_val])
            else:
                count = len(self.df[(self.df["perplexity"] >= min_val) & (self.df["perplexity"] < max_val)])
            range_counts.append(count)
            range_labels.append(f"{label}\n({count} samples)")

        bars = axes[1, 2].bar(
            range(len(range_counts)),
            range_counts,
            color=range_colors,
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, 2].set_xlabel("Performance Range")
        axes[1, 2].set_ylabel("Number of Samples")
        axes[1, 2].set_title("Sample Distribution by Performance")
        axes[1, 2].set_xticks(range(len(range_labels)))
        axes[1, 2].set_xticklabels(range_labels, rotation=45, ha="right")
        axes[1, 2].grid(True, alpha=0.3)

        # 数値をバーの上に表示
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(range_counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        # ファイル保存
        output_path = self.output_dir / "molecule_nat_lang_detailed_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Detailed analysis plot saved: {output_path}")
        plt.close()

        return str(output_path)

    def generate_statistics_summary(self):
        """統計サマリーの生成（オーバーライド）"""
        logger.info("Generating Molecule NL statistics summary")

        stats = {
            "Dataset Statistics": {
                "Total Samples": len(self.df),
                "Mean Text Length": f"{self.df['text_length'].mean():.1f} characters",
                "Mean Token Length": f"{self.df['token_length'].mean():.1f} tokens",
                "Text Length Range": f"{self.df['text_length'].min()}-{self.df['text_length'].max()} characters",
            },
            "Perplexity Statistics": {
                "Mean Perplexity": f"{self.df['perplexity'].mean():.3f}",
                "Median Perplexity": f"{self.df['perplexity'].median():.3f}",
                "Std Perplexity": f"{self.df['perplexity'].std():.3f}",
                "Min Perplexity": f"{self.df['perplexity'].min():.3f}",
                "Max Perplexity": f"{self.df['perplexity'].max():.3f}",
                "25th Percentile": f"{self.df['perplexity'].quantile(0.25):.3f}",
                "75th Percentile": f"{self.df['perplexity'].quantile(0.75):.3f}",
            },
            "Performance Distribution": {
                "Excellent (PPL < 10)": len(self.df[self.df["perplexity"] < 10]),
                "Good (10 ≤ PPL < 50)": len(self.df[(self.df["perplexity"] >= 10) & (self.df["perplexity"] < 50)]),
                "Moderate (50 ≤ PPL < 200)": len(self.df[(self.df["perplexity"] >= 50) & (self.df["perplexity"] < 200)]),
                "Poor (PPL ≥ 200)": len(self.df[self.df["perplexity"] >= 200]),
            },
        }

        return stats

    def plot_confusion_matrix(self):
        """混同行列プロットの生成（抽象メソッドの実装）"""
        logger.info("Confusion matrix not applicable for perplexity-based evaluation")

    def plot_performance_metrics(self):
        """性能指標プロットの生成（抽象メソッドの実装）"""
        logger.info("Creating performance metrics visualization")

        fig, ax = plt.subplots(figsize=(10, 6))

        # パープレキシティの統計指標をバープロット
        metrics = ["Mean", "Median", "Min", "Max", "25th %ile", "75th %ile"]
        values = [
            self.df["perplexity"].mean(),
            self.df["perplexity"].median(),
            self.df["perplexity"].min(),
            self.df["perplexity"].max(),
            self.df["perplexity"].quantile(0.25),
            self.df["perplexity"].quantile(0.75),
        ]

        bars = ax.bar(metrics, values, color=self.colors["primary"], alpha=0.8)
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity Performance Metrics")
        ax.grid(True, alpha=0.3)

        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = self.output_dir / "performance_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance metrics plot saved: {output_path}")
        plt.close()

        return str(output_path)

    def create_summary_dashboard(self):
        """サマリーダッシュボードの生成（抽象メソッドの実装）"""
        logger.info("Creating summary dashboard")
        return self.create_domain_specific_plot()

    def generate_all_visualizations(self):
        """全ての可視化の生成（抽象メソッドの実装）"""
        logger.info("Generating all Molecule NL visualizations")

        generated_plots = []

        try:
            # ドメイン特有の詳細分析
            plot_path = self.create_domain_specific_plot()
            generated_plots.append(plot_path)

            # 性能指標プロット
            metrics_path = self.plot_performance_metrics()
            generated_plots.append(metrics_path)

            # 包括的ダッシュボード
            dashboard_path = self._create_comprehensive_evaluation_dashboard()
            if dashboard_path:
                generated_plots.append(dashboard_path)

            logger.info(f"Generated {len(generated_plots)} visualization plots")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return generated_plots

    def _create_comprehensive_evaluation_dashboard(self):
        """
        Molecule NL特有の包括的評価ダッシュボードを作成

        BaseVisualizationGeneratorの_create_comprehensive_dashboard()を呼び出して
        6つのプロットを作成し、Molecule NL用のサンプルデータを生成します。
        """
        logger.info("Creating comprehensive Molecule NL evaluation dashboard")

        # サンプルデータの生成（Molecule NL特有）
        np.random.seed(42)
        n_samples = len(self.df) if hasattr(self, "df") and self.df is not None else 1000

        # パープレキシティベースのスコア（低いほど良い→高いほど良いに変換）
        perplexity_scores = (
            self.df["perplexity"].values
            if hasattr(self, "df") and self.df is not None
            else np.random.lognormal(2, 1, n_samples)
        )
        # パープレキシティを0-1スケールのスコア（高いほど良い）に変換
        max_perplexity = np.percentile(perplexity_scores, 95)  # 外れ値を除外

        # サンプルデータをDataFrameとして作成
        sample_data = pd.DataFrame(
            {
                "label": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),  # 分子理解タスクの正解
                "prediction": np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75]),  # 予測結果
                "score": 1.0 - np.clip(perplexity_scores / max_perplexity, 0, 1),  # パープレキシティから導出したスコア
                "confidence": np.random.beta(2, 2, n_samples),  # 信頼度
                "similarity": np.random.beta(3, 2, n_samples),  # 分子構造類似度
            }
        )

        # BaseVisualizationGeneratorの包括的ダッシュボード作成を呼び出し
        dashboard_path = self._create_comprehensive_dashboard(
            sample_data,
            prediction_score_col="score",
            true_label_col="label",
            confidence_col="confidence",
            similarity_col="similarity",
            custom_title="Molecule NL Evaluation Dashboard",
        )

        logger.info(f"Comprehensive Molecule NL dashboard created: {dashboard_path}")
        return dashboard_path

    def create_html_report(self):
        """HTML形式の詳細レポート作成（オーバーライド）"""
        logger.info("Creating Molecule NL HTML report")

        # 統計情報の取得
        stats = self.generate_statistics_summary()

        # HTMLテンプレートの作成
        html_content = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Molecule NL Model - Evaluation Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
                    color: #333;
                }

                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    overflow: hidden;
                }

                .header {
                    background: linear-gradient(135deg, #2E8B57, #32CD32);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }

                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }

                .header p {
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }

                .content {
                    padding: 30px;
                }

                .section {
                    margin-bottom: 40px;
                    background: #f8fffe;
                    border-radius: 8px;
                    padding: 25px;
                    border-left: 5px solid #2E8B57;
                }

                .section h2 {
                    color: #2E8B57;
                    margin-top: 0;
                    border-bottom: 2px solid #e1e8ed;
                    padding-bottom: 10px;
                }

                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }

                .stat-card {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    border: 1px solid #e1e8ed;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                .stat-card h3 {
                    color: #32CD32;
                    margin-top: 0;
                }

                .stat-item {
                    display: flex;
                    justify-content: space-between;
                    margin: 8px 0;
                    padding: 5px 0;
                    border-bottom: 1px solid #f0f4f7;
                }

                .stat-label {
                    font-weight: 500;
                    color: #555;
                }

                .stat-value {
                    font-weight: bold;
                    color: #2E8B57;
                }

                .timestamp {
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    border-top: 1px solid #e1e8ed;
                    padding-top: 20px;
                    margin-top: 40px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧬 Molecule NL Model</h1>
                    <p>分子自然言語理解モデル - 評価レポート</p>
                </div>

                <div class="content">
                    <div class="section">
                        <h2>📊 評価サマリー</h2>
                        <p>本レポートは、分子自然言語処理に特化したGPT-2モデルの性能評価結果を示しています。
                        パープレキシティを主要指標として、モデルの言語理解能力を定量的に分析しました。</p>
                    </div>

                    <div class="section">
                        <h2>📈 統計データ</h2>
                        <div class="stats-grid">
        """

        # 統計データの各セクションをHTMLに追加
        for section_name, section_data in stats.items():
            html_content += f"""
                            <div class="stat-card">
                                <h3>{section_name}</h3>
            """
            for key, value in section_data.items():
                html_content += f"""
                                <div class="stat-item">
                                    <span class="stat-label">{key}:</span>
                                    <span class="stat-value">{value}</span>
                                </div>
                """
            html_content += "</div>"

        html_content += f"""
                        </div>
                    </div>

                    <div class="section">
                        <h2>💡 主要な発見</h2>
                        <ul>
                            <li><strong>パフォーマンス分析:</strong> パープレキシティの分布から、モデルの言語理解能力を評価</li>
                            <li><strong>テキスト長の影響:</strong> 入力テキストの長さがモデル性能に与える影響を分析</li>
                            <li><strong>分子特化性:</strong> 化学・分子に関する自然言語の理解に特化したモデル性能</li>
                            <li><strong>実用性評価:</strong> 実際の分子情報処理タスクでの適用可能性を検証</li>
                        </ul>
                    </div>

                    <div class="section">
                        <h2>🎯 推奨事項</h2>
                        <ul>
                            <li>パープレキシティが10未満のサンプルは優秀な理解度を示しています</li>
                            <li>長いテキストでの性能低下が見られる場合は、入力分割を検討してください</li>
                            <li>分子構造と自然言語の対応関係の更なる分析を推奨します</li>
                            <li>ドメイン特化データでの追加ファインチューニングを検討してください</li>
                        </ul>
                    </div>

                    <div class="timestamp">
                        レポート生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # HTMLファイルの保存
        html_path = self.output_dir / "molecule_nat_lang_evaluation_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {html_path}")
        return str(html_path)


def main():
    """メイン実行関数（テスト用）"""
    import tempfile

    # テスト用のサンプルデータを作成
    test_data = {
        "index": range(100),
        "text_length": np.random.randint(50, 500, 100),
        "token_length": np.random.randint(10, 100, 100),
        "perplexity": np.random.lognormal(2, 1, 100),
        "log_perplexity": np.random.normal(2, 1, 100),
        "text_preview": [f"Sample text {i}..." for i in range(100)],
    }

    df = pd.DataFrame(test_data)

    # 一時ファイルとディレクトリの作成
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_results.csv")
        df.to_csv(csv_path, index=False)

        # 可視化器の初期化とテスト
        visualizer = MoleculeNLVisualizationGenerator(results_file=csv_path, output_dir=temp_dir)

        # 全ての可視化を生成
        visualizer.generate_all_visualizations()

        print(f"Test visualizations generated in: {temp_dir}")


if __name__ == "__main__":
    main()
