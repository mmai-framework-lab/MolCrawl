#!/usr/bin/env python3
"""
Molecule Natural Language モデルの評価スクリプト

このスクリプトは、訓練済みのGPT-2 molecule_nat_langモデルを使って
分子関連自然言語タスクの性能を検証します。
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk

# プロジェクトルートを追加

from molcrawl.utils.environment_check import check_learning_source_dir

from molcrawl.gpt2.model import GPT, GPTConfig

from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
from molcrawl.utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from molcrawl.utils.model_evaluator import ModelEvaluator

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class GPT2MoleculeNLEvaluator(ModelEvaluator):
    """Molecule NLデータを使用したモデル評価クラス"""

    def __init__(self, model_path, tokenizer_path="", device="cuda", max_length=1024):
        """
        初期化

        Args:
            model_path (str): 訓練済みモデルのパス
            tokenizer_path (str): 未使用（MoleculeNatLangTokenizerを内部で初期化）
            device (str): 使用デバイス
            max_length (int): 最大入力長
        """
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer_path = "molecule_nat_lang_internal"  # ダミーパス
        self.device = device

        # ModelEvaluatorのパス検証をスキップして直接初期化
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # トークナイザーとモデルを初期化
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()

        # 語彙サイズを設定
        self.vocab_size = getattr(self.tokenizer, "vocab_size", 32024)

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
        logger.info(f"Loading GPT2 model from {self.model_path}")
        return self._load_gpt2_model()

    def _load_gpt2_model(self):
        """訓練済みGPT2モデルの読み込み"""
        try:
            # チェックポイントの読み込み
            checkpoint = torch.load(self.model_path, map_location="cpu")
            logger.info("✅ Checkpoint loaded successfully")

            # モデル設定の取得
            if "config" in checkpoint:
                # 新しい形式: 設定が保存されている
                saved_config = checkpoint["config"]
                logger.info("📝 Using saved model configuration")

                # GPTConfigで使用可能なパラメータのみを抽出
                valid_params = {
                    "block_size",
                    "vocab_size",
                    "n_layer",
                    "n_head",
                    "n_embd",
                    "dropout",
                    "bias",
                }
                model_args = {k: v for k, v in saved_config.items() if k in valid_params}

                # 語彙サイズが設定にない場合は重みから推測
                if "vocab_size" not in model_args:
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                        if "transformer.wte.weight" in state_dict:
                            model_args["vocab_size"] = state_dict["transformer.wte.weight"].shape[0]
                            logger.info(f"📊 Detected vocab_size from weights: {model_args['vocab_size']}")

                logger.info(f"   - Filtered config keys: {list(model_args.keys())}")
                if "vocab_size" in model_args:
                    logger.info(f"   - Model vocab_size: {model_args['vocab_size']}")
                if self.tokenizer:
                    logger.info(f"   - Tokenizer vocab_size: {self.tokenizer.vocab_size}")
            else:
                # 古い形式: チェックポイントから語彙サイズを推測
                logger.warning("⚠️  No saved config found, using checkpoint weights for config")

                # チェックポイントから語彙サイズを推測
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif isinstance(checkpoint, dict) and "transformer.wte.weight" in checkpoint:
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # 埋め込み層から語彙サイズを取得
                vocab_size = 32024  # デフォルト値（チェックポイントから確認済み）
                if "transformer.wte.weight" in state_dict:
                    vocab_size = state_dict["transformer.wte.weight"].shape[0]
                    logger.info(f"📊 Detected vocab_size from checkpoint: {vocab_size}")
                elif "wte.weight" in state_dict:
                    vocab_size = state_dict["wte.weight"].shape[0]
                    logger.info(f"📊 Detected vocab_size from checkpoint: {vocab_size}")
                else:
                    logger.info(f"📊 Using default vocab_size: {vocab_size}")

                model_args = {
                    "block_size": 1024,
                    "vocab_size": vocab_size,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                    "dropout": 0.0,
                    "bias": True,
                }

            # モデルの作成
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

            # 重みの読み込み
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            # モデル統計の表示
            total_params = sum(p.numel() for p in model.parameters())
            logger.info("📊 Model Statistics:")
            logger.info(f"   - Total parameters: {total_params:,}")
            logger.info(f"   - Vocabulary size: {model_args.get('vocab_size', 'unknown')}")
            logger.info(f"   - Block size: {model_args.get('block_size', 'unknown')}")
            logger.info(f"   - Number of layers: {model_args.get('n_layer', 'unknown')}")
            logger.info(f"   - Embedding dimension: {model_args.get('n_embd', 'unknown')}")

            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """シーケンスをエンコード（抽象メソッドの実装）"""
        return self.encode_text(sequence)

    def encode_text(self, text, add_special_tokens=True):
        """テキストをトークンIDにエンコード"""
        try:
            # MoleculeNatLangTokenizerを使用してエンコード
            tokenized_result = self.tokenizer.tokenize_text(text)
            tokens = tokenized_result["input_ids"]

            # 最大長に調整
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
                logger.debug(f"Text truncated to {len(tokens)} tokens")

            if not tokens:
                logger.warning(f"Empty tokenization for text: {text[:50]}...")
                # パディングトークンのIDを使用
                tokens = [self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0]

            return torch.tensor(tokens, dtype=torch.long)

        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            # フォールバック：基本的なトークン化
            try:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"][0]
                return tokens
            except Exception as e2:
                logger.error(f"Fallback tokenization also failed: {e2}")
                return torch.tensor([0], dtype=torch.long)

    def calculate_perplexity(self, text_or_tokens):
        """テキストまたはトークンIDsのパープレキシティを計算

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
                    logger.info(f"Sequence too short for perplexity calculation: {len(tokens)} tokens")
                    return float("inf")

                # バッチ次元を追加してデバイスに転送
                tokens = tokens.unsqueeze(0).to(self.device)

                # 訓練時と同じように、入力と目標をシフト
                # x = tokens[:, :-1] (最後のトークンを除く)
                # y = tokens[:, 1:] (最初のトークンを除く)
                x = tokens[:, :-1]
                y = tokens[:, 1:]

                # モデルの予測
                logits, loss = self.model(x, targets=y)

                if loss is None:
                    logger.info("Model returned None for loss")
                    return float("inf")

                # パープレキシティは損失の指数
                perplexity = torch.exp(loss).item()

                return perplexity

            except Exception as e:
                logger.info(f"Error in perplexity calculation: {e}")
                import traceback

                logger.info(f"Traceback: {traceback.format_exc()}")
                return float("inf")

    def generate_text(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
        """テキスト生成"""
        self.model.eval()
        with torch.no_grad():
            tokens = self.encode_text(prompt)
            tokens = tokens.unsqueeze(0).to(self.device)

            for _ in range(max_new_tokens):
                if tokens.shape[1] >= self.max_length:
                    break

                # 予測
                logits, _ = self.model(tokens)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, top_k_indices, top_k_logits)

                # サンプリング
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # トークンを追加
                tokens = torch.cat([tokens, next_token], dim=1)

            # デコード
            generated_tokens = tokens[0].cpu().numpy().tolist()
            try:
                generated_text = self.tokenizer.decode(generated_tokens)
            except Exception as e:
                logger.warning(f"Decode failed in text generation: {e}")
                generated_text = f"[GENERATED TOKENS: {generated_tokens[:20]}...]"

            return generated_text

    def evaluate_dataset(
        self,
        dataset_path,
        output_dir="./reports/molecule_nat_lang_evaluation",
        sample_size=None,
    ):
        """
        Molecule NLデータセット全体の評価

        Args:
            dataset_path (str): データセットのパス
            output_dir (str): 出力ディレクトリ
            sample_size (int): サンプルサイズ（None=全データ）
        """
        print("DEBUG: evaluate_dataset method called!", flush=True)

        # タイムスタンプを追加してoutput_dirを更新
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🔬 Starting Molecule NL Model Evaluation")
        print("DEBUG: After first logger.info", flush=True)
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

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                avg_perplexity = np.mean(perplexities) if perplexities else float("inf")
                logger.info(f"   Progress: {idx}/{len(df)} texts processed, avg perplexity: {avg_perplexity:.2f}")

            try:
                # input_idsを直接使用（既にトークン化されているため）
                if "input_ids" in row:
                    input_ids = row["input_ids"]

                    # デバッグ: input_idsの型と内容を確認
                    if idx == 0:  # 最初のサンプルで詳細確認
                        logger.info("First sample analysis:")
                        logger.info(f"  input_ids type: {type(input_ids)}")
                        logger.info(f"  input_ids length: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")
                        if hasattr(input_ids, "__iter__"):
                            sample_ids = list(input_ids)[:10] if len(input_ids) > 10 else list(input_ids)
                            logger.info(f"  input_ids sample: {sample_ids}")
                            logger.info(f"  Min/Max IDs: {min(input_ids)}/{max(input_ids)}")

                    # numpy配列、リスト、タプルに対応
                    if hasattr(input_ids, "__len__") and len(input_ids) > 0:
                        # パープレキシティ計算（input_idsを直接渡す）
                        logger.info(f"Sample {idx}: Calculating perplexity for {len(input_ids)} tokens...")
                        perplexity = self.calculate_perplexity(input_ids)
                        logger.info(f"Sample {idx}: Perplexity = {perplexity}")

                        # テキストプレビュー用（表示のみ）
                        text_preview = row.get("input_text", f"[{len(input_ids)} tokens]")[:100]
                    else:
                        logger.warning(f"Sample {idx}: Empty or invalid input_ids")
                        continue
                else:
                    logger.warning(f"Sample {idx}: No input_ids found in row")
                    continue

                # 結果を記録
                result = {
                    "index": idx,
                    "text_length": len(text_preview),
                    "token_length": len(input_ids),
                    "perplexity": perplexity,
                    "log_perplexity": math.log(perplexity) if perplexity > 0 and perplexity != float("inf") else float("inf"),
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

        # 結果の保存と分析
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv"), index=False)

        logger.info(f"📊 Results DataFrame shape: {results_df.shape}")
        logger.info(f"   Columns: {list(results_df.columns)}")
        if len(results_df) > 0:
            logger.info(f"   Sample data:\n{results_df.head()}")

        # 性能指標の計算
        metrics = self._calculate_metrics(perplexities, results_df)

        # 結果の保存
        with open(os.path.join(output_dir, "molecule_nat_lang_evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 可視化（別クラスで処理）
        self._create_visualizations_with_separate_class(results_df, output_dir)

        # レポート生成
        self._generate_report(metrics, results_df, output_dir)

        logger.info(f"📁 Results saved to: {output_dir}")
        return metrics, results_df

    def _create_visualizations_with_separate_class(self, results_df, output_dir):
        """分離された可視化クラスを使用して可視化を生成"""
        try:
            # CSV結果ファイルのパスを生成
            csv_file = os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv")

            # 可視化クラスをインポート
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            # 可視化器を初期化
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # 全ての可視化を生成
            visualizer.generate_all_visualizations()

            # HTMLレポートも生成
            visualizer.create_html_report()

            logger.info("✅ Visualizations created using separate visualization class")

        except Exception as e:
            logger.warning(f"⚠️  Visualization generation failed: {e}")
            logger.info("📊 Falling back to basic visualization")
            # フォールバック：基本的な可視化のみ実行
            self._create_basic_visualization(results_df, output_dir)
        """分離された可視化クラスを使用して可視化を生成"""
        try:
            # CSV結果ファイルのパスを生成
            csv_file = os.path.join(output_dir, "molecule_nat_lang_detailed_results.csv")

            # 可視化クラスをインポート
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            # 可視化器を初期化
            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=output_dir)

            # 全ての可視化を生成
            visualizer.generate_all_visualizations()

            # HTMLレポートも生成
            visualizer.create_html_report()

            logger.info("✅ Visualizations created using separate visualization class")

        except Exception as e:
            logger.warning(f"⚠️  Visualization generation failed: {e}")
            logger.info("📊 Falling back to basic visualization")
            # フォールバック：基本的な可視化のみ実行
            self._create_basic_visualization(results_df, output_dir)

    def _create_basic_visualization(self, results_df, output_dir):
        """⚠️ DEPRECATED: 可視化は molecule_nat_lang_visualization.py を使用してください"""
        logger.warning("⚠️  Inline visualization is deprecated.")
        logger.info("Please use: python scripts/evaluation/gpt2/molecule_nat_lang_visualization.py --result-dir <output_dir>")
        logger.info("Skipping inline visualization.")

    def _calculate_metrics(self, perplexities, results_df):
        """性能指標の計算"""
        if not perplexities:
            logger.warning("No valid perplexities found")
            return {
                "total_samples": len(results_df),
                "valid_samples": 0,
                "mean_perplexity": float("inf"),
                "median_perplexity": float("inf"),
                "std_perplexity": float("inf"),
                "min_perplexity": float("inf"),
                "max_perplexity": float("inf"),
            }

        perplexities = np.array(perplexities)

        metrics = {
            "total_samples": len(results_df),
            "valid_samples": len(perplexities),
            "mean_perplexity": float(np.mean(perplexities)),
            "median_perplexity": float(np.median(perplexities)),
            "std_perplexity": float(np.std(perplexities)),
            "min_perplexity": float(np.min(perplexities)),
            "max_perplexity": float(np.max(perplexities)),
            "percentile_25": float(np.percentile(perplexities, 25)),
            "percentile_75": float(np.percentile(perplexities, 75)),
            "mean_text_length": float(results_df["text_length"].mean()) if "text_length" in results_df.columns else 0.0,
            "mean_token_length": float(results_df["token_length"].mean()) if "token_length" in results_df.columns else 0.0,
        }

        return metrics

    def _generate_report(self, metrics, results_df, output_dir):
        """評価レポートの生成"""
        report_path = os.path.join(output_dir, "molecule_nat_lang_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("Molecule Natural Language Model - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"🕐 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("🧬 Model Type: GPT-2 for Molecule Natural Language\n")
            f.write("📊 Evaluation Method: Perplexity-based language modeling assessment\n\n")

            f.write("📈 Dataset Summary:\n")
            f.write(f"   • Total samples evaluated: {metrics['total_samples']}\n")
            f.write(f"   • Valid samples: {metrics['valid_samples']}\n")
            f.write(f"   • Processing success rate: {metrics['valid_samples'] / metrics['total_samples']:.1%}\n\n")

            f.write("🎯 Performance Metrics:\n")
            f.write(f"   • Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Median Perplexity: {metrics.get('median_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Std Perplexity: {metrics.get('std_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Min Perplexity: {metrics.get('min_perplexity', float('inf')):.3f}\n")
            f.write(f"   • Max Perplexity: {metrics.get('max_perplexity', float('inf')):.3f}\n\n")

            f.write("📊 Text Statistics:\n")
            f.write(f"   • Mean Text Length: {metrics.get('mean_text_length', 0):.1f} characters\n")
            f.write(f"   • Mean Token Length: {metrics.get('mean_token_length', 0):.1f} tokens\n")
            f.write(f"   • 25th Percentile Perplexity: {metrics.get('percentile_25', float('inf')):.3f}\n")
            f.write(f"   • 75th Percentile Perplexity: {metrics.get('percentile_75', float('inf')):.3f}\n\n")

            # パフォーマンス解釈
            mean_ppl = metrics.get("mean_perplexity", float("inf"))
            if mean_ppl < 10:
                performance_interpretation = "🟢 Excellent language modeling performance"
            elif mean_ppl < 50:
                performance_interpretation = "🟡 Good language modeling performance"
            elif mean_ppl < 200:
                performance_interpretation = "🟠 Moderate language modeling performance"
            else:
                performance_interpretation = "🔴 Poor language modeling performance"

            f.write(f"📊 Overall Assessment: {performance_interpretation}\n")
            f.write(f"   Mean Perplexity: {mean_ppl:.3f}\n\n")

            f.write("💡 Key Insights:\n")
            f.write("   • Lower perplexity indicates better language modeling capability\n")
            f.write("   • This model specializes in molecule-related natural language\n")
            f.write("   • Perplexity reflects the model's uncertainty in predicting next tokens\n")
            f.write("   • Results should be compared with domain-specific baselines\n")


def main():
    parser = argparse.ArgumentParser(description="Molecule NL model evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt2-output/molecule_nat_lang-small/ckpt.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to test dataset directory")
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
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # 画像再生成オプション
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing results (skip evaluation)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory containing existing evaluation results (CSV files) for visualization",
    )

    args = parser.parse_args()

    # LEARNING_SOURCE_DIRの設定
    learning_source_dir = check_learning_source_dir()

    # 画像のみ生成モード
    if args.visualize_only:
        if args.result_dir is None:
            print("Error: --result-dir must be specified when using --visualize-only")
            sys.exit(1)

        if not os.path.exists(args.result_dir):
            print(f"Error: Result directory not found: {args.result_dir}")
            sys.exit(1)

        # ログ設定
        logger = setup_evaluation_logging(Path(args.result_dir), "molecule_nat_lang_visualization")

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        logger.info("🎨 Visualization-only mode")
        logger.info(f"📁 Loading results from: {args.result_dir}")

        try:
            # CSVファイルを読み込み
            csv_file = os.path.join(args.result_dir, "molecule_nat_lang_detailed_results.csv")
            if not os.path.exists(csv_file):
                logger.error(f"❌ CSV file not found: {csv_file}")
                sys.exit(1)

            results_df = pd.read_csv(csv_file)
            logger.info(f"✅ Loaded {len(results_df)} results from CSV")

            # メトリクスJSONを読み込み（存在する場合）
            json_file = os.path.join(args.result_dir, "molecule_nat_lang_evaluation_results.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metrics = json.load(f)
                logger.info("✅ Loaded metrics from JSON")
            else:
                # CSVから再計算
                logger.info("ℹ️  Metrics JSON not found, recalculating from CSV...")
                valid_perplexities = results_df[results_df["perplexity"] != float("inf")]["perplexity"].tolist()
                metrics = {
                    "total_samples": len(results_df),
                    "valid_samples": len(valid_perplexities),
                    "mean_perplexity": float(np.mean(valid_perplexities)) if valid_perplexities else float("inf"),
                    "median_perplexity": float(np.median(valid_perplexities)) if valid_perplexities else float("inf"),
                    "std_perplexity": float(np.std(valid_perplexities)) if valid_perplexities else float("inf"),
                    "min_perplexity": float(np.min(valid_perplexities)) if valid_perplexities else float("inf"),
                    "max_perplexity": float(np.max(valid_perplexities)) if valid_perplexities else float("inf"),
                }

            # 可視化生成クラスを使用
            from molecule_nat_lang_visualization import MoleculeNLVisualizationGenerator

            visualizer = MoleculeNLVisualizationGenerator(results_file=csv_file, output_dir=args.result_dir)

            # すべての可視化を生成
            logger.info("🎨 Generating visualizations...")
            visualizer.generate_all_visualizations()

            logger.info("✅ Visualization completed successfully!")
            logger.info(f"📊 Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}")
            logger.info(f"📈 Valid samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")

        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

        return

    # 通常の評価モード
    # デフォルトパスの設定
    if args.dataset_path is None:
        args.dataset_path = f"{learning_source_dir}/molecule_nat_lang/training_ready_hf_dataset/test"

    # データセットパスの存在確認
    if not os.path.exists(args.dataset_path):
        print(f"❌ ERROR: Dataset path does not exist: {args.dataset_path}")
        print(f"Expected structure: {learning_source_dir}/molecule_nat_lang/training_ready_hf_dataset/test")
        print("")
        print("Please verify that:")
        print(f"1. LEARNING_SOURCE_DIR='{learning_source_dir}' is correct")
        print("2. The molecule_nat_lang dataset has been processed")
        sys.exit(1)

    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "molecule_nat_lang", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), "molecule_nat_lang_evaluation")

    # ログレベルの設定（デバッグモード）
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    logger.info("Starting Molecule NL model evaluation...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # 評価器の初期化（トークナイザーパスは不要）
        evaluator = GPT2MoleculeNLEvaluator(
            model_path=args.model_path,
            tokenizer_path="",  # MoleculeNatLangTokenizerは内部で初期化
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
        logger.info(f"Mean Perplexity: {metrics.get('mean_perplexity', float('inf')):.3f}")
        logger.info(f"Valid samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
