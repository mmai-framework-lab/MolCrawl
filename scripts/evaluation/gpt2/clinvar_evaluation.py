#!/usr/bin/env python3
"""
ClinVarデータベースを使用したgenome sequenceモデルの精度検証スクリプト

このスクリプトは、訓練済みのGPT-2 genome sequenceモデルを使って
ClinVarデータベースの病原性変異を識別する精度を検証します。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "gpt2"))

from model import GPT, GPTConfig

from config.paths import get_genome_tokenizer_path
from utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from utils.model_evaluator import ModelEvaluator

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class GPT2ClinVarEvaluator(ModelEvaluator):
    """ClinVarデータを使用したモデル評価クラス"""

    def __init__(self, model_path, tokenizer_path, device="cuda"):
        """
        初期化

        Args:
            model_path (str): 訓練済みモデルのパス
            tokenizer_path (str): SentencePieceトークナイザーのパス
            device (str): 使用デバイス
        """
        # 親クラスの初期化（トークナイザーとモデルを自動初期化）
        super().__init__(model_path, tokenizer_path, device)

    def _init_tokenizer(self):
        """トークナイザーの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        self.vocab_size = tokenizer.vocab_size()
        logger.info(f"Tokenizer loaded with vocab_size: {self.vocab_size}")
        return tokenizer

    def _init_model(self):
        """モデルの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading model from {self.model_path}")
        return self._load_model()

    def encode_sequence(self, sequence: str, **kwargs):
        """
        配列をトークンIDにエンコード（抽象メソッドの実装）

        Args:
            sequence: ゲノム配列
            **kwargs: 追加の引数

        Returns:
            トークンIDのリスト
        """
        return self.tokenizer.encode(sequence)

    def _load_model(self):
        """訓練済みモデルの読み込み"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # モデル設定の復元
        model_args = checkpoint.get("model_args", {})
        config = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=model_args.get("block_size", 1024),
            n_layer=model_args.get("n_layer", 12),
            n_head=model_args.get("n_head", 12),
            n_embd=model_args.get("n_embd", 768),
            dropout=0.0,  # 評価時はdropoutを無効
            bias=model_args.get("bias", True),
        )

        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully. Config: {config}")
        return model

    def calculate_perplexity(self, sequence, variant_pos=None):
        """シーケンスのパープレキシティを計算"""
        """DNA配列をトークンIDにエンコード"""
        # 配列を適切にフォーマット（大文字変換、無効文字の除去など）
        sequence = sequence.upper().replace("N", "").replace("-", "")

        # SentencePieceでエンコード
        tokens = self.tokenizer.encode(sequence)

        # デバッグ用ログ
        logger.debug(f"Original sequence length: {len(sequence)}")
        logger.debug(f"Encoded tokens length: {len(tokens)}")
        logger.debug(f"First 10 chars: {sequence[:10]}")
        logger.debug(f"First 5 tokens: {tokens[:5] if tokens else 'Empty'}")

        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            # 空の場合は未知トークンを返す
            tokens = [self.tokenizer.unk_id()] if hasattr(self.tokenizer, "unk_id") else [0]

        return torch.tensor(tokens, dtype=torch.long)

    def get_variant_probability(self, reference_seq, variant_seq, context_length=512):
        """
        変異の病原性確率を計算

        Args:
            reference_seq (str): 参照配列
            variant_seq (str): 変異配列
            context_length (int): 評価に使用するコンテキスト長

        Returns:
            float: 変異の病原性確率（相対的な尤度の差）
        """
        with torch.no_grad():
            # 参照配列と変異配列をエンコード
            ref_tokens = self.encode_sequence(reference_seq)
            var_tokens = self.encode_sequence(variant_seq)

            # 最小長を確保（1以上）
            if len(ref_tokens) == 0:
                logger.warning("Reference sequence tokenization resulted in empty tokens")
                return 0.0
            if len(var_tokens) == 0:
                logger.warning("Variant sequence tokenization resulted in empty tokens")
                return 0.0

            # コンテキスト長に調整
            if len(ref_tokens) > context_length:
                ref_tokens = ref_tokens[:context_length]
            if len(var_tokens) > context_length:
                var_tokens = var_tokens[:context_length]

            # リストをテンソルに変換してからバッチ次元を追加
            ref_tokens = torch.tensor(ref_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            var_tokens = torch.tensor(var_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

            logger.debug(f"Model input shapes - ref: {ref_tokens.shape}, var: {var_tokens.shape}")

            # モデルの予測確率を取得（全系列の予測）
            ref_logits, _ = self.model(ref_tokens)
            var_logits, _ = self.model(var_tokens)

            logger.debug(f"Model output shapes - ref: {ref_logits.shape}, var: {var_logits.shape}")

            # GPT-2は各位置で次のトークンを予測するため、系列長と一致するはず
            if ref_logits.size(1) != ref_tokens.size(1):
                logger.warning(f"Expected ref_logits length {ref_tokens.size(1)}, got {ref_logits.size(1)}")
                # 最後の位置のlogitsのみ使用（生成モードの場合）
                if ref_logits.size(1) == 1 and ref_tokens.size(1) > 1:
                    # 各トークンを個別に処理
                    ref_likelihood = self._calculate_likelihood_token_by_token(ref_tokens)
                    var_likelihood = self._calculate_likelihood_token_by_token(var_tokens)
                else:
                    ref_likelihood = torch.tensor(0.0, device=self.device)
                    var_likelihood = torch.tensor(0.0, device=self.device)
            else:
                # 対数尤度を計算
                ref_log_prob = F.log_softmax(ref_logits, dim=-1)
                var_log_prob = F.log_softmax(var_logits, dim=-1)

                # 各トークンの尤度を計算
                ref_likelihood = self._calculate_sequence_likelihood(ref_tokens, ref_log_prob)
                var_likelihood = self._calculate_sequence_likelihood(var_tokens, var_log_prob)

            # 相対的な尤度の差を病原性スコアとして使用
            pathogenicity_score = ref_likelihood - var_likelihood

            return pathogenicity_score.item()

    def _calculate_sequence_likelihood(self, tokens, log_probs):
        """配列の対数尤度を計算"""
        # 入力チェック
        if tokens.size(1) <= 1:
            logger.warning(f"Sequence too short for likelihood calculation: {tokens.shape}")
            return torch.tensor(0.0, device=tokens.device)

        if log_probs.size(1) == 0:
            logger.warning(f"Empty log_probs: {log_probs.shape}")
            return torch.tensor(0.0, device=tokens.device)

        # 最後のトークンを除く（予測対象がないため）
        target_tokens = tokens[:, 1:]
        pred_log_probs = log_probs[:, :-1, :]

        # サイズの再確認
        if pred_log_probs.size(1) != target_tokens.size(1):
            logger.warning(f"Size mismatch: pred_log_probs={pred_log_probs.shape}, target_tokens={target_tokens.shape}")
            return torch.tensor(0.0, device=tokens.device)

        # 各位置での正解トークンの対数確率を取得
        token_log_probs = pred_log_probs.gather(2, target_tokens.unsqueeze(2)).squeeze(2)

        # 平均対数尤度を返す
        return token_log_probs.mean()

    def _calculate_likelihood_token_by_token(self, tokens):
        """トークンごとに尤度を計算（生成モード対応）"""
        if tokens.size(1) <= 1:
            return torch.tensor(0.0, device=tokens.device)

        total_log_prob = 0.0
        count = 0

        # 各位置での条件付き確率を計算
        for i in range(1, tokens.size(1)):
            context = tokens[:, :i]  # i番目までのコンテキスト
            target = tokens[:, i : i + 1]  # i+1番目のトークン（予測対象）

            with torch.no_grad():
                logits, _ = self.model(context)
                log_probs = F.log_softmax(logits[:, -1:, :], dim=-1)  # 最後の位置の予測のみ

                # 正解トークンの対数確率
                token_log_prob = log_probs.gather(2, target.unsqueeze(2)).squeeze()
                total_log_prob += token_log_prob.item()
                count += 1

        return torch.tensor(total_log_prob / count if count > 0 else 0.0, device=tokens.device)

    def load_clinvar_data(self, clinvar_file):
        """
        ClinVarデータの読み込み

        Args:
            clinvar_file (str): ClinVarデータファイルのパス

        Returns:
            pd.DataFrame: ClinVarデータ
        """
        logger.info(f"Loading ClinVar data from {clinvar_file}")

        # ファイル形式に応じて読み込み
        if clinvar_file.endswith(".csv"):
            df = pd.read_csv(clinvar_file)
        elif clinvar_file.endswith(".tsv"):
            df = pd.read_csv(clinvar_file, sep="\t")
        elif clinvar_file.endswith(".json"):
            df = pd.read_json(clinvar_file)
        else:
            raise ValueError(f"Unsupported file format: {clinvar_file}")

        # 必要なカラムの確認
        required_columns = [
            "reference_sequence",
            "variant_sequence",
            "ClinicalSignificance",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")

        # 病原性ラベルの標準化
        df = self._standardize_clinical_significance(df)

        logger.info(f"Loaded {len(df)} ClinVar variants")
        logger.info(f"Clinical significance distribution:\n{df['pathogenic'].value_counts()}")

        return df

    def _standardize_clinical_significance(self, df):
        """臨床的意義を病原性/非病原性の二値分類に標準化"""
        pathogenic_terms = [
            "pathogenic",
            "likely pathogenic",
            "pathogenic/likely pathogenic",
        ]
        benign_terms = ["benign", "likely benign", "benign/likely benign"]

        def classify_pathogenicity(significance):
            if pd.isna(significance):
                return None
            significance = significance.lower()
            if any(term in significance for term in pathogenic_terms):
                return 1  # 病原性
            elif any(term in significance for term in benign_terms):
                return 0  # 非病原性
            else:
                return None  # 不明（評価から除外）

        df["pathogenic"] = df["ClinicalSignificance"].apply(classify_pathogenicity)

        # 不明なものを除外
        df = df.dropna(subset=["pathogenic"])
        df["pathogenic"] = df["pathogenic"].astype(int)

        return df

    def evaluate_model(self, clinvar_data, batch_size=32):
        """
        モデルの評価実行

        Args:
            clinvar_data (pd.DataFrame): ClinVarデータ
            batch_size (int): バッチサイズ

        Returns:
            dict: 評価結果
        """
        logger.info("Starting model evaluation on ClinVar data")

        predictions = []
        true_labels = []
        pathogenicity_scores = []

        # バッチ処理で評価
        for i in range(0, len(clinvar_data), batch_size):
            batch = clinvar_data.iloc[i : i + batch_size]

            for _, row in batch.iterrows():
                try:
                    # 病原性スコアを計算
                    score = self.get_variant_probability(row["reference_sequence"], row["variant_sequence"])

                    pathogenicity_scores.append(score)
                    true_labels.append(row["pathogenic"])

                    logger.debug(f"Processed variant {len(predictions) + 1}/{len(clinvar_data)}")

                except Exception as e:
                    logger.warning(f"Error processing variant: {e}")
                    continue

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(clinvar_data)} variants")

        # 閾値を最適化して予測ラベルを決定
        pathogenicity_scores = np.array(pathogenicity_scores)
        true_labels = np.array(true_labels)

        # ROC曲線から最適閾値を計算
        optimal_threshold = self._find_optimal_threshold(pathogenicity_scores, true_labels)
        predictions = (pathogenicity_scores > optimal_threshold).astype(int)

        # 評価指標を計算
        results = self._calculate_metrics(true_labels, predictions, pathogenicity_scores)
        results["optimal_threshold"] = optimal_threshold

        logger.info("Model evaluation completed")
        return results

    def _find_optimal_threshold(self, scores, labels):
        """ROC曲線からF1スコアを最大化する閾値を見つける"""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(labels, scores)

        # F1スコアを最大化する閾値を見つける
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            pred = (scores > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, pred, average="binary")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def _calculate_metrics(self, true_labels, predictions, scores):
        """評価指標を計算"""
        from sklearn.metrics import average_precision_score, roc_auc_score

        # 基本指標
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

        # AUC指標
        roc_auc = roc_auc_score(true_labels, scores)
        pr_auc = average_precision_score(true_labels, scores)

        # 混同行列
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        }

        return results

    def save_results(self, results, output_file):
        """評価結果を保存"""
        logger.info(f"Saving results to {output_file}")

        # NumPy配列をリストに変換
        serializable_results = self._make_serializable(results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def _make_serializable(self, obj):
        """オブジェクトをJSON serializable形式に変換"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def create_sample_clinvar_data(output_file):
    """
    ⚠️ DEPRECATED: データ準備機能は clinvar_data_preparation.py に移行してください
    Use: python scripts/evaluation/gpt2/clinvar_data_preparation.py --mode sample
    """
    logger.warning("⚠️  create_sample_clinvar_data() is deprecated.")
    logger.warning("Please use: python scripts/evaluation/gpt2/clinvar_data_preparation.py --mode sample")
    raise DeprecationWarning("Use clinvar_data_preparation.py for data preparation")


def main():
    parser = argparse.ArgumentParser(description="ClinVar evaluation for genome sequence model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--clinvar_data",
        type=str,
        required=True,
        help="Path to ClinVar data file (CSV/TSV/JSON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample ClinVar data for testing",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to SentencePiece tokenizer model (auto-detect if not provided)",
    )

    args = parser.parse_args()

    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "clinvar", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), "clinvar_evaluation")

    # サンプルデータ作成モード
    if args.create_sample_data:
        create_sample_clinvar_data(args.clinvar_data)
        logger.info("Sample data created. Run again without --create_sample_data to evaluate.")
        return

    try:
        # トークナイザーパスの取得
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
            logger.info(f"Using specified tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = get_genome_tokenizer_path()
            logger.info(f"Using auto-detected tokenizer: {tokenizer_path}")

        # 評価器の初期化
        evaluator = GPT2ClinVarEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
        )

        # ClinVarデータの読み込み
        clinvar_data = evaluator.load_clinvar_data(args.clinvar_data)

        # モデル評価の実行
        results = evaluator.evaluate_model(clinvar_data, batch_size=args.batch_size)

        # 結果の表示
        logger.info("=== Evaluation Results ===")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-score: {results['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {results['pr_auc']:.4f}")
        logger.info(f"Sensitivity: {results['sensitivity']:.4f}")
        logger.info(f"Specificity: {results['specificity']:.4f}")

        # 結果の保存
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        evaluator.save_results(results, results_file)

        # 詳細レポートの作成
        report_file = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_file, "w") as f:
            f.write("ClinVar Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Data: {args.clinvar_data}\n")
            f.write(f"Total variants evaluated: {len(clinvar_data)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1-score: {results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {results['specificity']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            cm = results["confusion_matrix"]
            f.write(f"  True Positive: {cm['true_positive']}\n")
            f.write(f"  False Positive: {cm['false_positive']}\n")
            f.write(f"  True Negative: {cm['true_negative']}\n")
            f.write(f"  False Negative: {cm['false_negative']}\n")

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
