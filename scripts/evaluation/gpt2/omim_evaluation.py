#!/usr/bin/env python3
"""
OMIM Evaluation Script
======================

OMIM (Online Mendelian Inheritance in Man) データベースを使用して
ゲノム配列モデルの遺伝性疾患予測性能を評価するスクリプト

主な機能:
- OMIM遺伝性疾患データでのモデル評価
- 病原性変異予測精度の測定
- 遺伝形式別の詳細分析
- 包括的な評価レポート生成

評価指標:
- 精度 (Accuracy)
- 適合率 (Precision)
- 再現率 (Recall)
- F1スコア
- ROC-AUC, PR-AUC
- 感度・特異度
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
import warnings

warnings.filterwarnings("ignore")

# プロジェクトルートを追加
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

try:
    from src.config.paths import get_genome_tokenizer_path
    from gpt2.model import GPTConfig, GPT
    from src.utils.evaluation_output import (
        get_evaluation_output_dir,
        get_model_type_from_path,
        get_model_name_from_path,
        setup_evaluation_logging,
    )
    from src.utils.model_evaluator import ModelEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# setup_logging関数は utils.evaluation_output.setup_evaluation_logging を使用


class OMIMEvaluator(ModelEvaluator):
    """OMIM遺伝性疾患予測評価クラス"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # 親クラスの初期化（トークナイザーとモデルを自動初期化）
        super().__init__(
            model_path,
            tokenizer_path,
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # 評価結果保存用
        self.results = {}
        self.predictions = []

    def _init_tokenizer(self):
        """トークナイザーの初期化（抽象メソッドの実装）"""
        self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        self.vocab_size = tokenizer.vocab_size()
        self.logger.info(f"Tokenizer loaded with vocab_size: {self.vocab_size}")
        return tokenizer

    def _init_model(self):
        """モデルの初期化（抽象メソッドの実装）"""
        self.logger.info(f"Loading model from {self.model_path}")
        return self.load_model_and_tokenizer()

    def encode_sequence(self, sequence: str, **kwargs) -> List[int]:
        """
        配列をトークンIDにエンコード（抽象メソッドの実装）

        Args:
            sequence: ゲノム配列
            **kwargs: 追加の引数

        Returns:
            トークンIDのリスト
        """
        return self.tokenizer.encode(sequence)
        self.true_labels = []
        self.prediction_scores = []

    def load_model_and_tokenizer(self):
        """モデルとトークナイザーをロード"""
        try:
            # トークナイザーをロード
            self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.tokenizer_path)
            self.logger.info("Tokenizer loaded successfully")

            # モデルをロード（ファイルが存在しない場合はダミーモデル作成）
            self.logger.info(f"Loading model from {self.model_path}")

            if os.path.exists(self.model_path):
                # 実際のモデルファイルが存在する場合
                checkpoint = torch.load(self.model_path, map_location=self.device)

                model_args = checkpoint["model_args"]
                gptconf = GPTConfig(**model_args)
                self.model = GPT(gptconf)

                state_dict = checkpoint["model"]
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("_orig_mod.")
                }
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model.to(self.device)

                self.logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                # ダミーモデルを作成（テスト用）
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.logger.info("Creating dummy model for testing purposes")

                # ダミー設定でGPTConfigを作成
                vocab_size = self.tokenizer.vocab_size()
                gptconf = GPTConfig(
                    block_size=512,
                    vocab_size=vocab_size,
                    n_layer=6,
                    n_head=6,
                    n_embd=384,
                    dropout=0.0,
                    bias=False,
                )

                self.model = GPT(gptconf)
                self.model.eval()
                self.model.to(self.device)

                self.logger.info("Dummy model created successfully")

        except Exception as e:
            self.logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def tokenize_sequence(self, sequence: str, max_length: int = 512) -> torch.Tensor:
        """配列をトークン化"""
        # スペース区切りでトークン化
        spaced_sequence = " ".join(list(sequence.upper()))
        tokens = self.tokenizer.encode(spaced_sequence, out_type=int)

        # 長さ調整
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def calculate_sequence_likelihood(self, sequence: str) -> float:
        """配列の尤度を計算"""
        try:
            tokens = self.tokenize_sequence(sequence)

            with torch.no_grad():
                # モデル予測
                logits, _ = self.model(tokens)

                # 対数尤度計算
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # 各位置での対数尤度を合計
                total_log_likelihood = 0.0
                for i in range(tokens.size(1) - 1):
                    target_token = tokens[0, i + 1]
                    token_log_prob = log_probs[0, i, target_token]
                    total_log_likelihood += token_log_prob.item()

                # 平均対数尤度
                avg_log_likelihood = total_log_likelihood / (tokens.size(1) - 1)

                return avg_log_likelihood

        except Exception as e:
            self.logger.warning(f"Error calculating likelihood for sequence: {e}")
            return -np.inf

    def load_omim_data(self, data_path: str) -> pd.DataFrame:
        """OMIMデータをロード"""
        self.logger.info(f"Loading OMIM data from {data_path}")

        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} OMIM variants")

        # データの前処理
        if "is_disease_causing" in df.columns:
            df = df.dropna(subset=["sequence", "is_disease_causing"])
            df["is_disease_causing"] = df["is_disease_causing"].astype(int)
        else:
            self.logger.error("Required column 'is_disease_causing' not found in data")
            raise ValueError("Missing required column")

        # 遺伝形式分布を表示
        if "inheritance_pattern" in df.columns:
            self.logger.info("Inheritance pattern distribution:")
            self.logger.info(df["inheritance_pattern"].value_counts())

        # 病原性分布を表示
        self.logger.info("Disease causality distribution:")
        self.logger.info(df["is_disease_causing"].value_counts())

        return df

    def evaluate_variants(
        self, df: pd.DataFrame, batch_size: int = 16
    ) -> Dict[str, Any]:
        """バリアントを評価"""
        self.logger.info("Starting model evaluation on OMIM data")

        # リスト初期化
        predictions = []
        true_labels = []
        prediction_scores = []
        inheritance_patterns = []

        # バッチ処理
        total_variants = len(df)
        for i in range(0, total_variants, batch_size):
            batch_end = min(i + batch_size, total_variants)
            batch_df = df.iloc[i:batch_end]

            batch_scores = []
            for _, row in batch_df.iterrows():
                sequence = row["sequence"]
                likelihood = self.calculate_sequence_likelihood(sequence)
                batch_scores.append(likelihood)

            # バッチ結果を追加
            prediction_scores.extend(batch_scores)
            true_labels.extend(batch_df["is_disease_causing"].tolist())

            if "inheritance_pattern" in batch_df.columns:
                inheritance_patterns.extend(batch_df["inheritance_pattern"].tolist())

            # 進捗表示
            if (batch_end) % 50 == 0 or batch_end == total_variants:
                self.logger.info(f"Processed {batch_end}/{total_variants} variants")

        # 予測閾値の最適化
        optimal_threshold = self.find_optimal_threshold(true_labels, prediction_scores)

        # 予測ラベル生成
        predictions = [
            1 if score > optimal_threshold else 0 for score in prediction_scores
        ]

        # 結果保存
        self.true_labels = true_labels
        self.predictions = predictions
        self.prediction_scores = prediction_scores

        # 評価指標計算
        results = self.calculate_metrics(
            true_labels, predictions, prediction_scores, optimal_threshold
        )

        # 遺伝形式別分析
        if inheritance_patterns:
            results["inheritance_analysis"] = self.analyze_by_inheritance_pattern(
                true_labels, predictions, inheritance_patterns
            )

        self.results = results
        self.logger.info("Model evaluation completed")

        return results

    def find_optimal_threshold(
        self, true_labels: List[int], scores: List[float]
    ) -> float:
        """最適な閾値を見つける"""
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]
        except (ValueError, IndexError):
            return np.median(scores)

    def calculate_metrics(
        self,
        true_labels: List[int],
        predictions: List[int],
        scores: List[float],
        threshold: float,
    ) -> Dict[str, Any]:
        """評価指標を計算"""

        # 基本指標
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        # ROC-AUC, PR-AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except (ValueError, RuntimeError):
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(true_labels, scores)
        except (ValueError, RuntimeError):
            pr_auc = 0.0

        # 感度・特異度
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
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
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            "total_samples": len(true_labels),
            "positive_samples": sum(true_labels),
            "negative_samples": len(true_labels) - sum(true_labels),
        }

        # 結果をログ出力
        self.logger.info("=== Evaluation Results ===")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-score: {f1:.4f}")
        self.logger.info(f"ROC-AUC: {roc_auc:.4f}")
        self.logger.info(f"PR-AUC: {pr_auc:.4f}")
        self.logger.info(f"Sensitivity: {sensitivity:.4f}")
        self.logger.info(f"Specificity: {specificity:.4f}")

        return results

    def analyze_by_inheritance_pattern(
        self,
        true_labels: List[int],
        predictions: List[int],
        inheritance_patterns: List[str],
    ) -> Dict[str, Dict]:
        """遺伝形式別分析"""
        analysis = {}
        unique_patterns = set(inheritance_patterns)

        for pattern in unique_patterns:
            # パターン別データ抽出
            pattern_indices = [
                i for i, p in enumerate(inheritance_patterns) if p == pattern
            ]
            pattern_true = [true_labels[i] for i in pattern_indices]
            pattern_pred = [predictions[i] for i in pattern_indices]

            if len(pattern_true) > 0:
                # パターン別指標計算
                accuracy = accuracy_score(pattern_true, pattern_pred)
                precision = precision_score(pattern_true, pattern_pred, zero_division=0)
                recall = recall_score(pattern_true, pattern_pred, zero_division=0)
                f1 = f1_score(pattern_true, pattern_pred, zero_division=0)

                analysis[pattern] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "sample_count": len(pattern_true),
                    "positive_count": sum(pattern_true),
                }

        return analysis

    def save_results(self, output_dir: str):
        """結果を保存"""
        os.makedirs(output_dir, exist_ok=True)

        # JSON形式で保存
        results_file = os.path.join(output_dir, "omim_evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # 詳細レポート保存
        report_file = os.path.join(output_dir, "omim_evaluation_report.txt")
        self.generate_detailed_report(report_file)

        # 予測結果保存
        predictions_file = os.path.join(output_dir, "omim_predictions.csv")
        pred_df = pd.DataFrame(
            {
                "true_label": self.true_labels,
                "prediction": self.predictions,
                "prediction_score": self.prediction_scores,
            }
        )
        pred_df.to_csv(predictions_file, index=False)

        self.logger.info(f"Results saved to {output_dir}")

    def generate_detailed_report(self, report_file: str):
        """詳細レポートを生成"""
        with open(report_file, "w") as f:
            f.write("OMIM Hereditary Disease Prediction Evaluation Report\n")
            f.write("=" * 55 + "\n\n")

            # 基本統計
            f.write(f"Total samples: {self.results['total_samples']}\n")
            f.write(f"Disease-causing samples: {self.results['positive_samples']}\n")
            f.write(f"Benign samples: {self.results['negative_samples']}\n\n")

            # 性能指標
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {self.results['accuracy']:.4f}\n")
            f.write(f"  Precision: {self.results['precision']:.4f}\n")
            f.write(f"  Recall: {self.results['recall']:.4f}\n")
            f.write(f"  F1-score: {self.results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {self.results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {self.results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {self.results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {self.results['specificity']:.4f}\n")
            f.write(f"  Optimal threshold: {self.results['optimal_threshold']:.4f}\n\n")

            # 混同行列
            cm = self.results["confusion_matrix"]
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives: {cm['tp']}\n")
            f.write(f"  False Positives: {cm['fp']}\n")
            f.write(f"  True Negatives: {cm['tn']}\n")
            f.write(f"  False Negatives: {cm['fn']}\n\n")

            # 遺伝形式別分析
            if "inheritance_analysis" in self.results:
                f.write("Analysis by Inheritance Pattern:\n")
                for pattern, metrics in self.results["inheritance_analysis"].items():
                    f.write(f"\n  {pattern}:\n")
                    f.write(f"    Sample count: {metrics['sample_count']}\n")
                    f.write(f"    Positive count: {metrics['positive_count']}\n")
                    f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1-score: {metrics['f1_score']:.4f}\n")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="OMIM Evaluation for Genome Sequence Model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dummy_model",
        help="Path to the trained GPT-2 model (default: dummy_model)",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to OMIM evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
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

    try:
        # 出力ディレクトリを自動生成または指定されたものを使用
        if args.output_dir is None:
            model_type = get_model_type_from_path(args.model_path)
            model_name = get_model_name_from_path(args.model_path)
            args.output_dir = get_evaluation_output_dir(model_type, "omim", model_name)
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        # ログ設定
        logger = setup_evaluation_logging(Path(args.output_dir), "omim_evaluation")

        # トークナイザーパス取得
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
            logger.info(f"Using specified tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = get_genome_tokenizer_path()
            logger.info(f"Using auto-detected tokenizer: {tokenizer_path}")

        # 評価実行
        evaluator = OMIMEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
            logger=logger,
        )

        # データロードと評価
        df = evaluator.load_omim_data(args.data_path)
        evaluator.evaluate_variants(df, batch_size=args.batch_size)

        # 結果保存
        evaluator.save_results(args.output_dir)

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
