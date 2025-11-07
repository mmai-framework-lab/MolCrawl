#!/usr/bin/env python3
"""
Base Visualization Framework for Model Evaluation Results

This module provides a common interface and shared functionality for all
evaluation result visualization classes.
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BaseVisualizationGenerator(ABC):
    """
    評価結果可視化の抽象基底クラス

    全ての可視化クラスが継承すべき共通インターフェースと機能を提供する。

    共通機能:
    - 出力ディレクトリの管理
    - プロット設定の統一
    - 共通ユーティリティメソッド
    - HTMLレポート生成の基盤
    """

    def __init__(
        self,
        results_source: Union[str, Dict[str, Any]],
        output_dir: str = "./visualization_results",
        logger: Optional[logging.Logger] = None,
    ):
        """
        基底クラス初期化

        Args:
            results_source: 評価結果（ファイルパスまたは辞書データ）
            output_dir: 可視化結果の出力ディレクトリ
            logger: ログ出力用のロガー
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or self._setup_logger()

        # 結果データの読み込み
        self.results = self._load_results(results_source)

        # 可視化設定の初期化
        self._setup_plot_style()

        # 生成されたファイルリスト
        self.generated_files = []

        self.logger.info(
            f"Visualization generator initialized. Output directory: {self.output_dir}"
        )

    def _setup_logger(self) -> logging.Logger:
        """デフォルトのロガーを設定"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(self.__class__.__name__)

    def _load_results(
        self, results_source: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        評価結果データの読み込み

        Args:
            results_source: ファイルパスまたは辞書データ

        Returns:
            評価結果辞書
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
        """プロットスタイルの統一設定"""
        # 基本スタイル設定
        plt.style.use("default")
        sns.set_palette("husl")

        # 日本語フォント対応
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

        # 図のデフォルト設定
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

        # グリッドとスタイル
        sns.set_style("whitegrid")

        self.logger.debug("Plot style configured")

    def _save_plot(self, filename: str, formats: List[str] = None) -> List[Path]:
        """
        プロットを指定された形式で保存

        Args:
            filename: ファイル名（拡張子なし）
            formats: 保存する形式のリスト（デフォルト: ['png', 'pdf']）

        Returns:
            保存されたファイルパスのリスト
        """
        if formats is None:
            formats = ["png", "pdf"]

        saved_files = []
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            plt.savefig(filepath, format=fmt, bbox_inches="tight")
            saved_files.append(filepath)
            self.generated_files.append(filepath)

        plt.close()
        self.logger.debug(f"Plot saved: {filename} in {formats}")
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
        汎用的な包括ダッシュボードを作成（BERT形式ベース）

        Args:
            results_df: 結果データフレーム
            prediction_score_col: 予測スコア列名
            true_label_col: 真の正解ラベル列名
            confidence_col: 信頼度列名（オプション）
            similarity_col: 類似度列名（オプション）
            custom_title: ダッシュボードのタイトル
        """
        self.logger.info("Creating comprehensive evaluation dashboard")

        # データの検証
        required_cols = [prediction_score_col, true_label_col]
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return

        plt.style.use("default")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 予測スコアの分布（ラベル別）
        self._plot_score_distribution(
            axes[0, 0], results_df, prediction_score_col, true_label_col
        )

        # 類似度分布（もし利用可能なら）
        if similarity_col and similarity_col in results_df.columns:
            self._plot_similarity_distribution(
                axes[0, 1], results_df, similarity_col, true_label_col
            )
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

        # 散布図: スコア vs 類似度
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

        # ROC曲線
        self._plot_roc_curve(
            axes[1, 0], results_df, prediction_score_col, true_label_col
        )

        # 混同行列
        self._plot_confusion_matrix_subplot(
            axes[1, 1], results_df, prediction_score_col, true_label_col
        )

        # 信頼度分布
        if confidence_col and confidence_col in results_df.columns:
            self._plot_confidence_distribution(axes[1, 2], results_df, confidence_col)
        else:
            # 予測スコアの代替として使用
            self._plot_confidence_distribution(
                axes[1, 2], results_df, prediction_score_col
            )

        plt.suptitle(custom_title, fontsize=16, y=0.98)
        plt.tight_layout()
        self._save_plot("comprehensive_dashboard")

    def _plot_score_distribution(
        self, ax, results_df: pd.DataFrame, score_col: str, label_col: str
    ):
        """予測スコアの分布をプロット"""
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

    def _plot_similarity_distribution(
        self, ax, results_df: pd.DataFrame, sim_col: str, label_col: str
    ):
        """類似度分布をプロット"""
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

    def _plot_score_vs_similarity_scatter(
        self, ax, results_df: pd.DataFrame, score_col: str, sim_col: str, label_col: str
    ):
        """スコア vs 類似度の散布図"""
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

    def _plot_roc_curve(
        self, ax, results_df: pd.DataFrame, score_col: str, label_col: str
    ):
        """ROC曲線をプロット"""
        try:
            from sklearn.metrics import roc_curve, roc_auc_score

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

    def _plot_confusion_matrix_subplot(
        self, ax, results_df: pd.DataFrame, score_col: str, label_col: str
    ):
        """混同行列をサブプロットとしてプロット"""
        try:
            from sklearn.metrics import confusion_matrix

            # 予測値を生成（閾値0.5または0）
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

    def _plot_confidence_distribution(
        self, ax, results_df: pd.DataFrame, conf_col: str
    ):
        """信頼度分布をプロット"""
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

    def _create_figure_grid(
        self, nrows: int, ncols: int, figsize: tuple = None
    ) -> tuple:
        """
        グリッド形式の図を作成

        Args:
            nrows: 行数
            ncols: 列数
            figsize: 図のサイズ

        Returns:
            (figure, axes)のタプル
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
        結果データに必要なキーが存在するか検証

        Args:
            required_keys: 必要なキーのリスト

        Raises:
            KeyError: 必要なキーが存在しない場合
        """
        missing_keys = [key for key in required_keys if key not in self.results]
        if missing_keys:
            raise KeyError(f"Missing required keys in results: {missing_keys}")

    def _get_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _create_html_header(self, title: str) -> str:
        """HTML レポートのヘッダー部分を生成"""
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
        """HTML レポートのフッター部分を生成"""
        return f"""
            <div class="footer">
                <p>Report generated by {self.__class__.__name__}</p>
                <p>Files created: {len(self.generated_files)} visualizations</p>
            </div>
        </body>
        </html>
        """

    def get_generated_files(self) -> List[Path]:
        """生成されたファイルのリストを取得"""
        return self.generated_files.copy()

    # 抽象メソッド群 - サブクラスで実装必須

    @abstractmethod
    def plot_confusion_matrix(self):
        """混同行列プロットの生成（実装必須）"""
        pass

    @abstractmethod
    def plot_performance_metrics(self):
        """性能指標プロットの生成（実装必須）"""
        pass

    @abstractmethod
    def create_summary_dashboard(self):
        """サマリーダッシュボードの生成（実装必須）"""
        pass

    @abstractmethod
    def generate_all_visualizations(self):
        """全ての可視化の生成（実装必須）"""
        pass

    @abstractmethod
    def create_html_report(self):
        """HTMLレポートの生成（実装必須）"""
        pass

    # オプショナル抽象メソッド（サブクラスで必要に応じて実装）

    def plot_auc_comparison(self):
        """AUC指標の比較プロット（オプション）"""
        self.logger.info("AUC comparison plot not implemented in this visualizer")

    def create_performance_radar_chart(self):
        """性能指標レーダーチャート（オプション）"""
        self.logger.info("Performance radar chart not implemented in this visualizer")

    def plot_score_distribution(self):
        """スコア分布プロット（オプション）"""
        self.logger.info("Score distribution plot not implemented in this visualizer")
