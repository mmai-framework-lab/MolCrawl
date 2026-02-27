#!/usr/bin/env python3
"""
BERT Genome Sequence Model - ClinVar Pathogenicity Evaluation

独立したBERT評価システム：訓練済みBERTモデルによるClinVar病原性変異の精度検証
GPT2とは完全に独立した実装で、BERT特有の評価手法を活用します。

主要な評価手法：
1. マスク言語モデル(MLM)による変異位置の予測確率
2. 変異前後の配列表現ベクトルの類似度分析
3. 注意重み可視化による重要領域の特定
4. 配列コンテキスト理解度の定量評価
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from importlib import import_module
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
    roc_auc_score,
)
from transformers import BertConfig, BertForMaskedLM

from molcrawl.utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from molcrawl.utils.model_evaluator import ModelEvaluator

# プロジェクトルートを追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 共通環境チェックモジュールを追加
check_learning_source_dir = import_module("utils.environment_check").check_learning_source_dir

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class BERTClinVarEvaluator(ModelEvaluator):
    """ClinVarデータを使用したBERTモデル評価クラス"""

    def __init__(self, model_path, tokenizer_path, device="cuda", max_length=512):
        """
        初期化

        Args:
            model_path (str): 訓練済みBERTモデルのパス
            tokenizer_path (str): SentencePieceトークナイザーのパス
            device (str): 使用デバイス
            max_length (int): 最大入力長
        """
        # 親クラスの初期化
        super().__init__(model_path, tokenizer_path, device)

        self.max_length = max_length

        # サブクラス固有の初期化
        self.tokenizer = self._init_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size()
        self.model = self._init_model()

        # 特殊トークンのIDを取得
        self.mask_token_id = self._get_mask_token_id()
        self.cls_token_id = self._get_cls_token_id()
        self.sep_token_id = self._get_sep_token_id()

    def _init_tokenizer(self):
        """トークナイザーの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        return spm.SentencePieceProcessor(model_file=self.tokenizer_path)

    def _init_model(self):
        """モデルの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading BERT model from {self.model_path}")
        return self._load_model()

    def _get_mask_token_id(self):
        """MASKトークンのIDを取得"""
        mask_candidates = ["<mask>", "[MASK]", "<unk>"]
        for candidate in mask_candidates:
            try:
                token_id = self.tokenizer.piece_to_id(candidate)
                if token_id != self.tokenizer.unk_id():
                    logger.info(f"Using {candidate} as MASK token (ID: {token_id})")
                    return token_id
            except (KeyError, IndexError, AttributeError):
                continue

        # フォールバック: unk_idを使用
        logger.warning("MASK token not found, using UNK token")
        return self.tokenizer.unk_id()

    def _get_cls_token_id(self):
        """CLSトークンのIDを取得"""
        cls_candidates = ["<cls>", "[CLS]", "<s>"]
        for candidate in cls_candidates:
            try:
                token_id = self.tokenizer.piece_to_id(candidate)
                if token_id != self.tokenizer.unk_id():
                    logger.info(f"Using {candidate} as CLS token (ID: {token_id})")
                    return token_id
            except (KeyError, IndexError, AttributeError):
                continue

        # フォールバック: 最初のトークンを使用
        logger.warning("CLS token not found, using first token")
        return 0

    def _get_sep_token_id(self):
        """SEPトークンのIDを取得"""
        sep_candidates = ["<sep>", "[SEP]", "</s>"]
        for candidate in sep_candidates:
            try:
                token_id = self.tokenizer.piece_to_id(candidate)
                if token_id != self.tokenizer.unk_id():
                    logger.info(f"Using {candidate} as SEP token (ID: {token_id})")
                    return token_id
            except (KeyError, IndexError, AttributeError):
                continue

        # フォールバック: EOSトークンを使用
        logger.warning("SEP token not found, using EOS token")
        return self.tokenizer.eos_id() if hasattr(self.tokenizer, "eos_id") else 1

    def _load_model(self):
        """訓練済みBERTモデルの読み込み（safetensors対応）"""
        try:
            logger.info(f"Loading trained BERT model from: {self.model_path}")

            # Hugging Face transformers形式での読み込み
            config = BertConfig.from_pretrained(self.model_path)
            logger.info(f"Model config loaded: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

            # トークナイザーのサイズと一致するかチェック
            if config.vocab_size != self.vocab_size:
                logger.warning(f"Vocab size mismatch: model={config.vocab_size}, tokenizer={self.vocab_size}")
                logger.info("Using model's original vocab size for compatibility")
                # モデルの元のvocab_sizeを保持
                original_vocab_size = config.vocab_size
                self.vocab_size = original_vocab_size  # トークナイザーのサイズを調整

            # safetensorsファイルから訓練済みモデルを読み込み
            model = BertForMaskedLM.from_pretrained(
                self.model_path,
                config=config,
                local_files_only=True,  # ローカルファイルのみ使用
                use_safetensors=True,  # safetensors形式を使用
                ignore_mismatched_sizes=False,  # サイズ不一致を厳密にチェック
            )

            logger.info("✅ Successfully loaded trained BERT model with safetensors")

        except Exception as e:
            logger.error(f"❌ Failed to load trained model: {e}")
            logger.info("🔄 Creating new untrained model as fallback")

            # フォールバック: 新しい未訓練モデル
            config = BertConfig(
                vocab_size=self.vocab_size,
                max_position_embeddings=self.max_length,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            model = BertForMaskedLM(config)
            logger.warning("⚠️  Using untrained model - results may not be meaningful")

        model.to(self.device)
        model.eval()

        # モデル統計の表示
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("📊 Model Statistics:")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Hidden size: {model.config.hidden_size}")
        logger.info(f"   - Number of layers: {model.config.num_hidden_layers}")
        logger.info(f"   - Attention heads: {model.config.num_attention_heads}")
        logger.info(f"   - Max sequence length: {model.config.max_position_embeddings}")

        return model

    def encode_sequence(self, sequence, add_special_tokens=True):
        """DNA配列をトークンIDにエンコード（改良版）"""
        # 配列を適切にフォーマット（大文字変換、無効文字の除去など）
        original_length = len(sequence)
        sequence = sequence.upper().replace("N", "").replace("-", "")

        # SentencePieceでエンコード
        tokens = self.tokenizer.encode(sequence)

        # デバッグ情報
        logger.debug(f"Sequence processing: {original_length} -> {len(sequence)} -> {len(tokens)} tokens")

        if add_special_tokens:
            # [CLS] + sequence + [SEP] の形式
            tokens = [self.cls_token_id] + tokens + [self.sep_token_id]

        # 最大長に調整
        if len(tokens) > self.max_length:
            if add_special_tokens:
                # [CLS] + sequence[:max_length-2] + [SEP]
                tokens = tokens[: self.max_length - 1] + [self.sep_token_id]
            else:
                tokens = tokens[: self.max_length]
            logger.debug(f"Sequence truncated to {len(tokens)} tokens")

        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            # 空の場合は最低限のトークンを返す
            if add_special_tokens:
                tokens = [self.cls_token_id, self.sep_token_id]
            else:
                tokens = [self.tokenizer.unk_id()]

        return torch.tensor(tokens, dtype=torch.long)

    def find_sequence_differences(self, ref_sequence, var_sequence):
        """参照配列と変異配列の詳細な差分分析"""
        ref_clean = ref_sequence.upper().replace("N", "").replace("-", "")
        var_clean = var_sequence.upper().replace("N", "").replace("-", "")

        differences = []
        min_len = min(len(ref_clean), len(var_clean))

        for i in range(min_len):
            if ref_clean[i] != var_clean[i]:
                differences.append({"position": i, "ref_base": ref_clean[i], "var_base": var_clean[i]})

        # 長さの差も記録
        if len(ref_clean) != len(var_clean):
            differences.append(
                {
                    "type": "length_difference",
                    "ref_length": len(ref_clean),
                    "var_length": len(var_clean),
                }
            )

        return differences

    def get_masked_variant_probability(self, reference_seq, variant_seq, variant_position=None):
        """
        マスク言語モデルを使用した変異の病原性確率を計算

        Args:
            reference_seq (str): 参照配列
            variant_seq (str): 変異配列
            variant_position (int): 変異位置（わからない場合はNone）

        Returns:
            dict: 各種スコア
        """
        with torch.no_grad():
            # 参照配列をエンコード
            ref_tokens = self.encode_sequence(reference_seq, add_special_tokens=True)
            var_tokens = self.encode_sequence(variant_seq, add_special_tokens=True)

            if len(ref_tokens) == 0 or len(var_tokens) == 0:
                logger.warning("Empty tokenization")
                return {"mlm_score": 0.0, "confidence": 0.0}

            # 変異位置を特定（簡単な方法）
            if variant_position is None:
                variant_position = self._find_variant_position(ref_tokens, var_tokens)

            if variant_position == -1:
                logger.warning("Could not find variant position")
                return {"mlm_score": 0.0, "confidence": 0.0}

            # 参照配列の変異位置をマスク
            masked_ref_tokens = ref_tokens.clone()
            original_token = masked_ref_tokens[variant_position].item()
            masked_ref_tokens[variant_position] = self.mask_token_id

            # バッチ次元を追加してデバイスに転送
            masked_ref_tokens = masked_ref_tokens.unsqueeze(0).to(self.device)

            # アテンションマスクを作成
            attention_mask = torch.ones_like(masked_ref_tokens)

            # モデルの予測
            outputs = self.model(
                input_ids=masked_ref_tokens,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            # マスク位置の予測確率を取得
            logits = outputs.logits[0, variant_position]  # [vocab_size]
            probs = F.softmax(logits, dim=-1)

            # 変異トークンと参照トークンの確率
            variant_token = var_tokens[variant_position].item()

            ref_prob = probs[original_token].item()
            var_prob = probs[variant_token].item()

            # スコア計算
            mlm_score = np.log(ref_prob + 1e-10) - np.log(var_prob + 1e-10)
            confidence = max(ref_prob, var_prob)

            # 注意重みの取得（最後の層の平均）
            attentions = outputs.attentions[-1].mean(dim=1).squeeze()  # [seq_len, seq_len]
            mask_attention = attentions[variant_position].cpu().numpy()

            return {
                "mlm_score": mlm_score,
                "ref_prob": ref_prob,
                "var_prob": var_prob,
                "confidence": confidence,
                "attention_weights": mask_attention,
                "variant_position": variant_position,
            }

    def _find_variant_position(self, ref_tokens, var_tokens):
        """参照配列と変異配列の差分位置を特定（改良版）"""
        min_len = min(len(ref_tokens), len(var_tokens))

        # 長さが大きく異なる場合は中央付近を変異位置とする
        if abs(len(ref_tokens) - len(var_tokens)) > 5:
            return min_len // 2

        differences = []
        for i in range(1, min_len - 1):  # [CLS]と[SEP]を除く
            if ref_tokens[i] != var_tokens[i]:
                differences.append(i)

        if differences:
            # 複数の差分がある場合は最初の位置を使用
            return differences[0]

        # 差分が見つからない場合は配列の中央を使用
        logger.warning("No differences found between reference and variant sequences")
        return min_len // 2 if min_len > 2 else 1

    def get_sequence_representation(self, sequence):
        """配列の表現ベクトルを取得"""
        with torch.no_grad():
            tokens = self.encode_sequence(sequence, add_special_tokens=True)
            tokens = tokens.unsqueeze(0).to(self.device)

            attention_mask = torch.ones_like(tokens)

            outputs = self.model.bert(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # [CLS]トークンの最終層表現を使用
            cls_representation = outputs.last_hidden_state[0, 0]  # [hidden_size]

            return cls_representation.cpu().numpy()

    def evaluate_dataset(
        self,
        dataset_path,
        output_dir="./reports/bert_clinvar_evaluation_results",
        sample_size=None,
    ):
        """
        ClinVarデータセット全体の独立BERT評価

        Args:
            dataset_path (str): データセットのパス
            output_dir (str): 出力ディレクトリ
            sample_size (int): サンプルサイズ（None=全データ）
        """

        # 出力ディレクトリの作成（タイムスタンプは付けない - GPT-2評価と統一）
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🔬 Starting Independent BERT ClinVar Evaluation")
        logger.info("=" * 60)

        # データセット読み込み
        logger.info("📚 Loading ClinVar dataset...")
        df = pd.read_csv(dataset_path)

        # カラム名の標準化（GPT-2形式との互換性）
        column_mapping = {
            "Chromosome": "chrom",
            "Start": "pos",
            "ReferenceAllele": "ref",
            "AlternateAllele": "alt",
        }
        # 存在するカラムのみリネーム
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        if existing_mappings:
            df = df.rename(columns=existing_mappings)
            logger.info(f"Standardized column names: {list(existing_mappings.keys())} → {list(existing_mappings.values())}")

        # データの前処理とラベル変換
        logger.info("🔄 Preprocessing ClinVar data...")
        df["pathogenic"] = (df["ClinicalSignificance"] == "Pathogenic").astype(int)

        # VariationIDがない場合は生成
        if "VariationID" not in df.columns or df["VariationID"].isnull().any():
            df["VariationID"] = range(len(df))

        # GeneSymbolの生成（chromとposが利用可能な場合）
        if "chrom" in df.columns and "pos" in df.columns:
            df["GeneSymbol"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)
        elif "GeneSymbol" not in df.columns:
            df["GeneSymbol"] = "UNKNOWN"  # フォールバック

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"📊 Using sample of {len(df)} variants")
        else:
            logger.info(f"📊 Evaluating all {len(df)} variants")

        # データセット統計
        pathogenic_count = df["pathogenic"].sum()
        benign_count = len(df) - pathogenic_count
        logger.info(f"   - Pathogenic variants: {pathogenic_count}")
        logger.info(f"   - Benign variants: {benign_count}")
        logger.info("   - Data format: chrom:pos (ref>alt)")

        # データ品質チェック
        missing_ref = df["reference_sequence"].isnull().sum()
        missing_var = df["variant_sequence"].isnull().sum()
        if missing_ref > 0 or missing_var > 0:
            logger.warning(f"⚠️  Missing sequences: ref={missing_ref}, var={missing_var}")
            df = df.dropna(subset=["reference_sequence", "variant_sequence"])

        results = []
        predictions = []
        true_labels = []
        processing_errors = 0

        logger.info("🧬 Processing variants...")

        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                accuracy_so_far = accuracy_score(true_labels, predictions) if predictions else 0
                logger.info(f"   Progress: {idx}/{len(df)} variants processed, accuracy: {accuracy_so_far:.3f}")

            try:
                # MLMベースの評価
                scores = self.get_masked_variant_probability(row["reference_sequence"], row["variant_sequence"])

                # 表現ベクトルベースの評価
                ref_repr = self.get_sequence_representation(row["reference_sequence"])
                var_repr = self.get_sequence_representation(row["variant_sequence"])

                # 類似度計算
                cosine_sim = np.dot(ref_repr, var_repr) / (np.linalg.norm(ref_repr) * np.linalg.norm(var_repr) + 1e-10)

                # BERT独自のスコア計算
                bert_pathogenicity_score = self._calculate_bert_pathogenicity_score(scores, cosine_sim)

                result = {
                    "VariationID": row["VariationID"],
                    "GeneSymbol": row.get("GeneSymbol", "UNKNOWN"),
                    "chrom": row.get("chrom", "N/A"),
                    "pos": row.get("pos", "N/A"),
                    "ref": row.get("ref", "N/A"),
                    "alt": row.get("alt", "N/A"),
                    "ClinicalSignificance": row["ClinicalSignificance"],
                    "pathogenic": row["pathogenic"],
                    "mlm_score": scores["mlm_score"],
                    "ref_prob": scores["ref_prob"],
                    "var_prob": scores["var_prob"],
                    "confidence": scores["confidence"],
                    "cosine_similarity": cosine_sim,
                    "representation_distance": np.linalg.norm(ref_repr - var_repr),
                    "bert_pathogenicity_score": bert_pathogenicity_score,
                    "variant_position": scores.get("variant_position", -1),
                    "sequence_length": len(row["reference_sequence"]),
                }

                results.append(result)

                # 病原性予測（複合スコア使用）
                prediction = 1 if bert_pathogenicity_score > 0.5 else 0
                predictions.append(prediction)
                true_labels.append(row["pathogenic"])

            except Exception as e:
                processing_errors += 1
                logger.debug(f"Error processing variant {idx}: {e}")
                continue

        logger.info("✅ Processing completed!")
        logger.info(f"   - Successfully processed: {len(results)} variants")
        logger.info(f"   - Processing errors: {processing_errors}")

        # 結果の保存と分析
        os.makedirs(output_dir, exist_ok=True)

        # 詳細結果をCSVで保存
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "bert_clinvar_detailed_results.csv"), index=False)

        # 性能指標の計算
        metrics = self._calculate_metrics(predictions, true_labels, results_df)

        # 結果の保存
        with open(os.path.join(output_dir, "bert_clinvar_evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 可視化
        self._create_visualizations(results_df, output_dir)

        # レポート生成
        self._generate_report(metrics, results_df, output_dir)

        logger.info(f"📁 Results saved to: {output_dir}")
        return metrics, results_df

    def _calculate_bert_pathogenicity_score(self, mlm_scores, cosine_similarity):
        """BERT独自の病原性スコア計算"""
        # MLMスコアと類似度スコアを組み合わせた複合スコア
        mlm_component = 1 / (1 + np.exp(-mlm_scores["mlm_score"]))  # シグモイド変換
        similarity_component = 1 - cosine_similarity  # 類似度が低いほど病原性
        confidence_weight = mlm_scores["confidence"]

        # 重み付き平均
        composite_score = 0.6 * mlm_component + 0.3 * similarity_component + 0.1 * confidence_weight

        return float(composite_score)

    def _calculate_metrics(self, predictions, true_labels, results_df):
        """性能指標の計算"""
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # 基本的な分類指標
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

        # ROC-AUC（MLMスコアを使用）
        mlm_scores = results_df["mlm_score"].values
        try:
            auc = roc_auc_score(true_labels, mlm_scores)
        except (ValueError, RuntimeError):
            auc = 0.5

        # 混同行列
        cm = confusion_matrix(true_labels, predictions)

        metrics = {
            "total_variants": len(predictions),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "pathogenic_count": int(np.sum(true_labels)),
            "benign_count": int(len(true_labels) - np.sum(true_labels)),
            "mean_mlm_score_pathogenic": float(results_df[results_df["pathogenic"] == 1]["mlm_score"].mean()),
            "mean_mlm_score_benign": float(results_df[results_df["pathogenic"] == 0]["mlm_score"].mean()),
            "mean_cosine_similarity_pathogenic": float(results_df[results_df["pathogenic"] == 1]["cosine_similarity"].mean()),
            "mean_cosine_similarity_benign": float(results_df[results_df["pathogenic"] == 0]["cosine_similarity"].mean()),
        }

        return metrics

    def _create_visualizations(self, results_df, output_dir):
        """結果の可視化 - clinvar_visualization.pyに移行推奨"""
        logger.warning("⚠️  Visualization code in evaluation script. Please use clinvar_visualization.py instead.")
        logger.info(
            "Skipping inline visualization. Use: python scripts/evaluation/bert/clinvar_visualization.py --result-dir <output_dir>"
        )

    def _generate_report(self, metrics, results_df, output_dir):
        """独立BERT評価レポートの生成"""
        report_path = os.path.join(output_dir, "bert_clinvar_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("Independent BERT Genome Sequence Model - ClinVar Evaluation Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"🕐 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("🧬 Model Type: BERT for Masked Language Modeling\n")
            f.write("📊 Evaluation Method: Independent pathogenicity assessment\n\n")

            f.write("📈 Dataset Summary:\n")
            f.write(f"   • Total variants evaluated: {metrics['total_variants']}\n")
            f.write(f"   • Pathogenic variants: {metrics['pathogenic_count']}\n")
            f.write(f"   • Benign variants: {metrics['benign_count']}\n")
            f.write(f"   • Class balance: {metrics['pathogenic_count'] / (metrics['total_variants']):.1%} pathogenic\n\n")

            f.write("🎯 Performance Metrics:\n")
            f.write(f"   • Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"   • Precision: {metrics['precision']:.3f}\n")
            f.write(f"   • Recall: {metrics['recall']:.3f}\n")
            f.write(f"   • F1-score: {metrics['f1_score']:.3f}\n")
            f.write(f"   • AUC-ROC: {metrics['auc']:.3f}\n\n")

            f.write("🧠 BERT-Specific Analysis:\n")
            f.write(f"   • MLM Score (Pathogenic): {metrics['mean_mlm_score_pathogenic']:.3f}\n")
            f.write(f"   • MLM Score (Benign): {metrics['mean_mlm_score_benign']:.3f}\n")
            f.write(f"   • Sequence Similarity (Pathogenic): {metrics['mean_cosine_similarity_pathogenic']:.3f}\n")
            f.write(f"   • Sequence Similarity (Benign): {metrics['mean_cosine_similarity_benign']:.3f}\n\n")

            # BERT独自の解釈
            f.write("🔍 BERT Model Interpretation:\n")
            f.write("   • MLM Scores: Higher values indicate reference sequence is more probable\n")
            f.write("   • Cosine Similarity: Lower values suggest greater sequence disruption\n")
            f.write("   • BERT Pathogenicity Score: Composite score combining MLM and similarity\n")
            f.write("   • Bidirectional Context: BERT considers both upstream and downstream context\n\n")

            # パフォーマンス解釈
            performance_interpretation = ""
            if metrics["accuracy"] > 0.8:
                performance_interpretation = "🟢 Excellent performance"
            elif metrics["accuracy"] > 0.7:
                performance_interpretation = "🟡 Good performance"
            elif metrics["accuracy"] > 0.6:
                performance_interpretation = "🟠 Moderate performance"
            else:
                performance_interpretation = "🔴 Poor performance"

            f.write(f"📊 Overall Assessment: {performance_interpretation}\n")
            f.write(f"   Accuracy: {metrics['accuracy']:.1%}, F1-Score: {metrics['f1_score']:.3f}\n\n")

            f.write("💡 Key Insights:\n")
            f.write("   • This evaluation is independent of GPT2 or other generative models\n")
            f.write("   • BERT's bidirectional attention enables context-aware variant assessment\n")
            f.write("   • MLM predictions provide direct evidence of sequence disruption\n")
            f.write("   • Results reflect BERT's understanding of genomic sequence patterns\n")


def main():
    # 環境変数からデフォルトのトークナイザーパスを構築
    try:
        learning_source_dir = check_learning_source_dir()
        default_tokenizer_path = f"{learning_source_dir}/genome_sequence/spm_tokenizer.model"
    except SystemExit:
        default_tokenizer_path = None

    parser = argparse.ArgumentParser(description="BERT ClinVar evaluation for genome sequence model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained BERT model")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=default_tokenizer_path,
        help="Path to SentencePiece tokenizer model (uses LEARNING_SOURCE_DIR if not specified)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/clinvar/random_2000_clinvar.csv",
        help="Path to ClinVar evaluation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for testing (default: use all data)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    # トークナイザーパスのチェック
    if not args.tokenizer_path:
        print("❌ ERROR: No tokenizer path specified!")
        print("")
        print("Please either:")
        print("  1. Set LEARNING_SOURCE_DIR environment variable:")
        print("     export LEARNING_SOURCE_DIR='...'")
        print("     python scripts/evaluation/bert/clinvar_evaluation.py --model_path <path>")
        print("")
        print("  2. Or specify --tokenizer_path explicitly:")
        print("     python scripts/evaluation/bert/clinvar_evaluation.py --model_path <path> --tokenizer_path <path>")
        sys.exit(1)

    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "bert_clinvar", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), "bert_clinvar_evaluation")

    logger.info("Starting BERT ClinVar evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # 評価器の初期化
        evaluator = BERTClinVarEvaluator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            max_length=args.max_length,
        )

        # 評価の実行
        metrics, results_df = evaluator.evaluate_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )

        # 結果の表示
        logger.info("Evaluation completed successfully!")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"F1-score: {metrics['f1_score']:.3f}")
        logger.info(f"AUC: {metrics['auc']:.3f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
