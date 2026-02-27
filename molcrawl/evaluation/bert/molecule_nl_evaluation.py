#!/usr/bin/env python3
"""
BERT Molecule Natural Language Model - Evaluation Script

訓練済みBERTモデルによる分子関連自然言語タスクの精度検証
GPT-2版と同じ評価基準と出力フォーマットを使用します。

主要な評価手法：
1. Masked Language Modeling (MLM)による予測確率
2. パープレキシティ計算（MLMベース）
3. 配列長・トークン長の分析
4. GPT-2版と同じ可視化フォーマット
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM
from molcrawl.utils.environment_check import check_learning_source_dir

# プロジェクトルートを追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from molcrawl.utils.evaluation_output import (  # noqa: E402
    setup_evaluation_logging,
)
from molcrawl.utils.model_evaluator import ModelEvaluator  # noqa: E402

# Molecule NL tokenizer
from molcrawl.molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer  # noqa: E402

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class BERTMoleculeNLEvaluator(ModelEvaluator):
    """Molecule NLデータを使用したBERTモデル評価クラス"""

    def __init__(self, model_path, tokenizer_path="", device="cuda", max_length=512):
        """
        初期化

        Args:
            model_path (str): 訓練済みBERTモデルのパス
            tokenizer_path (str): 未使用（MoleculeNatLangTokenizerを内部で初期化）
            device (str): 使用デバイス
            max_length (int): 最大入力長
        """
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer_path = "molecule_nl_internal"  # ダミーパス
        self.device = device

        # ModelEvaluatorのパス検証をスキップして直接初期化
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # トークナイザーを初期化
        self.tokenizer = self._init_tokenizer()

        # 語彙サイズを設定（モデル読み込み前に必要）
        self.vocab_size = getattr(self.tokenizer, "vocab_size", 32024)

        # モデルを初期化
        self.model = self._init_model()

        # 特殊トークンのIDを取得
        self.mask_token_id = self._get_mask_token_id()
        self.cls_token_id = self._get_cls_token_id()
        self.sep_token_id = self._get_sep_token_id()
        self.pad_token_id = self._get_pad_token_id()

    def _init_tokenizer(self):
        """トークナイザーの初期化（抽象メソッドの実装）"""
        logger.info("Loading MoleculeNatLangTokenizer")
        try:
            tokenizer = MoleculeNatLangTokenizer()
            logger.info(f"✅ MoleculeNatLangTokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Failed to load MoleculeNatLangTokenizer: {e}")
            raise

    def _init_model(self):
        """モデルの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading BERT model from {self.model_path}")
        return self._load_model()

    def _get_mask_token_id(self):
        """MASKトークンのIDを取得"""
        # MoleculeNatLangTokenizerから取得を試みる
        if hasattr(self.tokenizer, "mask_token_id"):
            logger.info(f"Using MASK token ID: {self.tokenizer.mask_token_id}")
            return self.tokenizer.mask_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "mask_token_id"):
            logger.info(f"Using MASK token ID: {self.tokenizer.tokenizer.mask_token_id}")
            return self.tokenizer.tokenizer.mask_token_id
        else:
            # フォールバック
            logger.warning("MASK token not found, using ID 103 (BERT default)")
            return 103

    def _get_cls_token_id(self):
        """CLSトークンのIDを取得"""
        if hasattr(self.tokenizer, "cls_token_id"):
            return self.tokenizer.cls_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "cls_token_id"):
            return self.tokenizer.tokenizer.cls_token_id
        else:
            logger.warning("CLS token not found, using ID 101 (BERT default)")
            return 101

    def _get_sep_token_id(self):
        """SEPトークンのIDを取得"""
        if hasattr(self.tokenizer, "sep_token_id"):
            return self.tokenizer.sep_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "sep_token_id"):
            return self.tokenizer.tokenizer.sep_token_id
        else:
            logger.warning("SEP token not found, using ID 102 (BERT default)")
            return 102

    def _get_pad_token_id(self):
        """PADトークンのIDを取得"""
        if hasattr(self.tokenizer, "pad_token_id"):
            return self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "pad_token_id"):
            return self.tokenizer.tokenizer.pad_token_id
        else:
            logger.warning("PAD token not found, using ID 0")
            return 0

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

    def encode_text(self, text):
        """テキストをトークンIDにエンコード"""
        try:
            # MoleculeNatLangTokenizerを使用
            if hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(text, max_length=self.max_length)
                return torch.tensor(tokens, dtype=torch.long)
            else:
                # HuggingFace tokenizer形式の場合
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                return encoding["input_ids"][0]
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            return torch.tensor([self.cls_token_id, self.sep_token_id], dtype=torch.long)

    def encode_sequence(self, sequence):
        """
        配列をトークンIDにエンコード（抽象メソッドの実装）

        molecule_nlではencode_textと同じ処理
        """
        return self.encode_text(sequence)

    def calculate_perplexity(self, text_or_tokens):
        """
        テキストまたはトークンIDsのパープレキシティを計算（BERT MLMベース）

        BERTはMasked Language Modelingを使用するため、各トークンを順番にマスクして
        予測確率を計算し、その平均からパープレキシティを算出します。

        Args:
            text_or_tokens: テキスト文字列 または トークンIDのリスト/配列
        """
        with torch.no_grad():
            try:
                # 入力がトークンIDsかテキストかを判定
                if isinstance(text_or_tokens, (list, tuple)) or (
                    hasattr(text_or_tokens, "__iter__") and not isinstance(text_or_tokens, str)
                ):
                    # 既にトークン化されている場合
                    if hasattr(text_or_tokens, "tolist"):
                        tokens = torch.tensor(text_or_tokens.tolist(), dtype=torch.long)
                    else:
                        tokens = torch.tensor(list(text_or_tokens), dtype=torch.long)
                else:
                    # テキストの場合はエンコード
                    tokens = self.encode_text(text_or_tokens)

                if len(tokens) < 2:
                    logger.debug(f"Sequence too short for perplexity calculation: {len(tokens)} tokens")
                    return float("inf")

                # バッチ次元を追加してデバイスに転送
                tokens = tokens.unsqueeze(0).to(self.device)

                # 各トークンを順番にマスクして予測確率を計算
                total_log_prob = 0.0
                num_predictions = 0

                for i in range(tokens.shape[1]):
                    # 特殊トークンはスキップ
                    if tokens[0, i].item() in [
                        self.cls_token_id,
                        self.sep_token_id,
                        self.pad_token_id,
                    ]:
                        continue

                    # トークンをマスク
                    masked_tokens = tokens.clone()
                    original_token = masked_tokens[0, i].item()
                    masked_tokens[0, i] = self.mask_token_id

                    # 予測
                    outputs = self.model(masked_tokens)
                    logits = outputs.logits

                    # マスク位置の予測確率を取得
                    masked_token_logits = logits[0, i, :]
                    probs = F.softmax(masked_token_logits, dim=-1)

                    # 元のトークンの予測確率
                    token_prob = probs[original_token].item()

                    # 確率が0の場合は小さな値に置き換え
                    if token_prob <= 0:
                        token_prob = 1e-10

                    # 対数確率を累積
                    total_log_prob += math.log(token_prob)
                    num_predictions += 1

                if num_predictions == 0:
                    logger.debug("No valid tokens for perplexity calculation")
                    return float("inf")

                # 平均対数確率からパープレキシティを計算
                avg_log_prob = total_log_prob / num_predictions
                perplexity = math.exp(-avg_log_prob)

                return perplexity

            except Exception as e:
                logger.debug(f"Error in perplexity calculation: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                return float("inf")

    def evaluate_dataset(
        self,
        dataset_path,
        output_dir="./reports/bert_molecule_nl_evaluation",
        sample_size=None,
    ):
        """
        Molecule NLデータセット全体の評価（GPT-2版と同じフォーマット）

        Args:
            dataset_path (str): データセットのパス
            output_dir (str): 出力ディレクトリ
            sample_size (int): サンプルサイズ（None=全データ）
        """
        # タイムスタンプを追加してoutput_dirを更新
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🔬 Starting BERT Molecule NL Model Evaluation")
        logger.info("=" * 60)

        # データセット読み込み
        logger.info("📚 Loading Molecule NL dataset...")
        try:
            dataset = load_from_disk(dataset_path)

            # DatasetDictの場合、適切なsplitを選択
            if hasattr(dataset, "keys"):
                # DatasetDictの場合、testまたはvalidationを優先的に使用
                if "test" in dataset:
                    dataset_split = dataset["test"]
                    logger.info("Using 'test' split from dataset")
                elif "validation" in dataset:
                    dataset_split = dataset["validation"]
                    logger.info("Using 'validation' split from dataset")
                elif "train" in dataset:
                    dataset_split = dataset["train"]
                    logger.info("Using 'train' split from dataset")
                else:
                    # 最初のsplitを使用
                    split_name = list(dataset.keys())[0]
                    dataset_split = dataset[split_name]
                    logger.info(f"Using '{split_name}' split from dataset")
            else:
                # 単一のDatasetの場合
                dataset_split = dataset

            # HuggingFace DatasetをpandasのDataFrameに変換
            df = dataset_split.to_pandas()
            logger.info(f"✅ Dataset loaded successfully: {len(df)} samples")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"📊 Using sample of {len(df)} texts")
        else:
            logger.info(f"📊 Evaluating all {len(df)} texts")

        # データセット統計
        logger.info(f"   - Total samples: {len(df)}")
        logger.info(f"   - Available columns: {list(df.columns)}")
        if "input_ids" in df.columns:
            # input_idsがリスト型の場合の長さ計算
            try:
                avg_length = df["input_ids"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0).mean()
                logger.info(f"   - Average sequence length: {avg_length:.1f}")
            except Exception as e:
                logger.warning(f"   - Could not calculate average sequence length: {e}")

        results = []
        perplexities = []
        processing_errors = 0

        logger.info("🧬 Processing texts...")
        logger.info("   Note: Using 'input_text' field and re-tokenizing with BERT tokenizer")

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                avg_perplexity = np.mean(perplexities) if perplexities else float("inf")
                logger.info(f"   Progress: {idx}/{len(df)} texts processed, avg perplexity: {avg_perplexity:.2f}")

            try:
                # input_textを使用してBERT tokenizerで再トークン化
                # (既存のinput_idsはGPT-2用なので使わない)
                if "input_text" in row and row["input_text"]:
                    text = row["input_text"]

                    # デバッグ: 最初のサンプルで確認
                    if idx == 0:
                        logger.info("First sample analysis:")
                        logger.info(f"  input_text: {text[:100]}...")
                        logger.info(f"  Re-tokenizing with BERT tokenizer (vocab_size={self.vocab_size})")

                    # BERT tokenizerでテキストをエンコード
                    # encode_textメソッドを使用（内部でMoleculeNatLangTokenizerが使われる）
                    try:
                        tokens = self.encode_text(text)

                        if idx == 0:
                            logger.info(f"  BERT tokenized length: {len(tokens)}")
                            logger.info(
                                f"  BERT token sample: {tokens[:20].tolist() if hasattr(tokens, 'tolist') else tokens[:20]}"
                            )
                            if hasattr(tokens, "__iter__"):
                                token_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
                                if token_list:
                                    logger.info(f"  BERT token range: {min(token_list)} - {max(token_list)}")

                        if len(tokens) == 0:
                            logger.warning(f"Sample {idx}: Empty tokens after encoding")
                            continue

                    except Exception as e:
                        logger.warning(f"Sample {idx}: Failed to encode text: {e}")
                        continue

                    # パープレキシティ計算
                    if idx < 5 or idx % 500 == 0:  # 最初の5個と500個おきにログ
                        logger.info(f"Sample {idx}: Calculating perplexity for {len(tokens)} tokens...")

                    perplexity = self.calculate_perplexity(tokens)

                    if idx < 5 or idx % 500 == 0:
                        logger.info(f"Sample {idx}: Perplexity = {perplexity:.4f}")

                    # テキストプレビュー用
                    text_preview = text[:100]

                else:
                    logger.warning(f"Sample {idx}: No input_text found in row")
                    continue

                # 結果を記録（GPT-2版と同じフォーマット）
                result = {
                    "index": idx,
                    "text_length": len(text_preview),
                    "token_length": len(tokens),
                    "perplexity": perplexity,
                    "log_perplexity": (math.log(perplexity) if perplexity > 0 and perplexity != float("inf") else float("inf")),
                    "text_preview": text_preview,
                }

                results.append(result)
                if perplexity != float("inf"):
                    perplexities.append(perplexity)

            except Exception as e:
                processing_errors += 1
                logger.warning(f"Error processing sample {idx}: {e}")
                import traceback

                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        logger.info("✅ Processing completed!")
        logger.info(f"   - Successfully processed: {len(results)} texts")
        logger.info(f"   - Processing errors: {processing_errors}")

        # 結果が空の場合の処理
        if not results:
            logger.error("❌ No samples were successfully processed!")
            logger.error("   Please check:")
            logger.error("   1. Dataset format (should have 'input_ids' or 'text' field)")
            logger.error("   2. Tokenizer compatibility")
            logger.error("   3. Model checkpoint path")

            # 空のDataFrameを作成（カラムは定義する）
            results_df = pd.DataFrame(
                columns=[
                    "index",
                    "text_length",
                    "token_length",
                    "perplexity",
                    "log_perplexity",
                    "text_preview",
                ]
            )

            # 空のメトリクスを返す
            metrics = {
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": 0.0,
                "total_samples": 0,
                "valid_samples": 0,
                "processing_errors": processing_errors,
            }

            return metrics, results_df

        # 結果の保存と分析（GPT-2版と同じフォーマット）
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "molecule_nl_detailed_results.csv"), index=False)

        logger.info(f"📊 Results DataFrame shape: {results_df.shape}")
        logger.info(f"   Columns: {list(results_df.columns)}")
        if len(results_df) > 0:
            logger.info(f"   Sample data:\n{results_df.head()}")

        # 性能指標の計算
        metrics = self._calculate_metrics(perplexities, results_df)

        # 結果の保存
        with open(os.path.join(output_dir, "molecule_nl_evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 可視化（GPT-2版と同じクラスを使用）
        self._create_visualizations_with_separate_class(results_df, output_dir)

        # レポート生成
        self._generate_report(metrics, results_df, output_dir)

        logger.info(f"📁 Results saved to: {output_dir}")
        return metrics, results_df

    def _create_visualizations_with_separate_class(self, results_df, output_dir):
        """分離された可視化クラスを使用して可視化を生成（GPT-2版と同じ）"""
        try:
            # CSV結果ファイルのパスを生成
            csv_file = os.path.join(output_dir, "molecule_nl_detailed_results.csv")

            # 可視化クラスをインポート（GPT-2版と同じクラスを使用）
            from molecule_nl_visualization import MoleculeNLVisualizationGenerator

            # 可視化器を初期化
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # 可視化を生成
            logger.info("📊 Generating visualizations...")
            visualizer.generate_all_visualizations()
            logger.info("✅ Visualization completed successfully!")

        except ImportError as e:
            logger.warning(f"Could not import visualization module: {e}. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _calculate_metrics(self, perplexities, results_df):
        """性能指標の計算（GPT-2版と同じフォーマット）"""
        if not perplexities:
            return {
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": 0.0,
                "min_perplexity": float("inf"),
                "max_perplexity": float("inf"),
                "total_samples": len(results_df),
                "valid_samples": 0,
                "invalid_samples": len(results_df),
            }

        valid_perplexities = [p for p in perplexities if p != float("inf")]

        metrics = {
            "mean_perplexity": float(np.mean(valid_perplexities)),
            "median_perplexity": float(np.median(valid_perplexities)),
            "std_perplexity": float(np.std(valid_perplexities)),
            "min_perplexity": float(np.min(valid_perplexities)),
            "max_perplexity": float(np.max(valid_perplexities)),
            "total_samples": len(results_df),
            "valid_samples": len(valid_perplexities),
            "invalid_samples": len(results_df) - len(valid_perplexities),
        }

        logger.info("📊 Evaluation Metrics:")
        logger.info(f"   - Mean Perplexity: {metrics['mean_perplexity']:.3f}")
        logger.info(f"   - Median Perplexity: {metrics['median_perplexity']:.3f}")
        logger.info(f"   - Std Perplexity: {metrics['std_perplexity']:.3f}")
        logger.info(f"   - Min/Max Perplexity: {metrics['min_perplexity']:.3f} / {metrics['max_perplexity']:.3f}")
        logger.info(f"   - Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")

        return metrics

    def _generate_report(self, metrics, results_df, output_dir):
        """テキストレポート生成（GPT-2版と同じフォーマット）"""
        report_path = os.path.join(output_dir, "molecule_nl_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BERT Molecule Natural Language Model - Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("Model Information:\n")
            f.write(f"  Model Path: {self.model_path}\n")
            f.write(f"  Vocabulary Size: {self.vocab_size}\n")
            f.write(f"  Max Sequence Length: {self.max_length}\n")
            f.write(f"  Device: {self.device}\n\n")

            f.write("Overall Performance Metrics:\n")
            f.write(f"  Mean Perplexity: {metrics['mean_perplexity']:.3f}\n")
            f.write(f"  Median Perplexity: {metrics['median_perplexity']:.3f}\n")
            f.write(f"  Std Perplexity: {metrics['std_perplexity']:.3f}\n")
            f.write(f"  Min Perplexity: {metrics['min_perplexity']:.3f}\n")
            f.write(f"  Max Perplexity: {metrics['max_perplexity']:.3f}\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"  Total Samples: {metrics['total_samples']}\n")
            f.write(f"  Valid Samples: {metrics['valid_samples']}\n")
            f.write(f"  Invalid Samples: {metrics['invalid_samples']}\n\n")

            if len(results_df) > 0:
                f.write("Token Length Statistics:\n")
                f.write(f"  Mean Token Length: {results_df['token_length'].mean():.1f}\n")
                f.write(f"  Median Token Length: {results_df['token_length'].median():.1f}\n")
                f.write(f"  Min/Max Token Length: {results_df['token_length'].min()} / {results_df['token_length'].max()}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"📝 Report saved to: {report_path}")


def main():
    """メイン評価フロー"""
    parser = argparse.ArgumentParser(description="Evaluate BERT Molecule NL model on dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained BERT model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to Molecule NL dataset (parquet or HuggingFace format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing results (skip evaluation)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory containing existing results for visualization-only mode",
    )

    args = parser.parse_args()

    # 出力ディレクトリの設定
    if args.output_dir is None:
        # LEARNING_SOURCE_DIR環境変数を使用
        learning_source_dir = check_learning_source_dir()
        args.output_dir = os.path.join(
            learning_source_dir,
            "molecule_nl",
            "report",
            f"bert_molecule_nl_ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # 出力ディレクトリをPath型に変換してディレクトリ作成
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ログ設定（Path型を渡す）
    setup_evaluation_logging(output_dir_path, "bert_molecule_nl_evaluation")
    logger.info("Starting BERT Molecule NL model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Visualization-onlyモード
    if args.visualize_only:
        logger.info("📊 Visualization-only mode enabled")

        if not args.result_dir:
            logger.error("--result-dir is required in visualization-only mode")
            return

        if not os.path.exists(args.result_dir):
            logger.error(f"Result directory not found: {args.result_dir}")
            return

        # 既存の結果ファイルを確認
        csv_file = os.path.join(args.result_dir, "molecule_nl_detailed_results.csv")
        json_file = os.path.join(args.result_dir, "molecule_nl_evaluation_results.json")

        if not os.path.exists(csv_file):
            logger.error(f"CSV results file not found: {csv_file}")
            return

        # 可視化のみを生成
        try:
            from molecule_nl_visualization import MoleculeNLVisualizationGenerator

            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=args.result_dir)

            logger.info("📊 Generating visualizations from existing results...")
            visualizer.generate_all_visualizations()
            logger.info("✅ Visualization completed successfully!")

            # メトリクスを表示
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                    logger.info(f"📊 Mean Perplexity: {metrics.get('mean_perplexity', 'N/A'):.3f}")
                    logger.info(
                        f"📈 Valid samples: {metrics.get('valid_samples', 'N/A')}/{metrics.get('total_samples', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"Error in visualization-only mode: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return

    # 通常の評価モード
    try:
        # 評価器の初期化
        evaluator = BERTMoleculeNLEvaluator(
            model_path=args.model_path,
            tokenizer_path="",  # 未使用
            device=args.device,
            max_length=args.max_length,
        )

        # データセット評価
        metrics, results_df = evaluator.evaluate_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )

        logger.info("Evaluation completed successfully!")
        logger.info(f"Mean Perplexity: {metrics['mean_perplexity']:.3f}")
        logger.info(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
