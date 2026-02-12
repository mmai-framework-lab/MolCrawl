#!/usr/bin/env python3
"""
BERT版ProteinGymデータベースを使用したprotein sequenceモデルの精度検証スクリプト

このスクリプトは、訓練済みのBERT protein sequenceモデルを使って
ProteinGymデータベースのタンパク質変異適合性を評価する精度を検証します。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_name_from_path,
    get_model_type_from_path,
    setup_evaluation_logging,
)
from utils.model_evaluator import ModelEvaluator

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class BERTProteinGymEvaluator(ModelEvaluator):
    """BERT版ProteinGymデータを使用したモデル評価クラス"""

    def __init__(self, model_path: str, tokenizer_path: Optional[str], device: str = "cuda") -> None:
        """
        初期化

        Args:
            model_path (str): 訓練済みBERTモデルのパス
            tokenizer_path (str): SentencePieceトークナイザーのパス
            device (str): 使用デバイス
        """
        # 親クラスの初期化
        super().__init__(model_path, tokenizer_path, device)

        # サブクラス固有の初期化
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()

    def _init_tokenizer(self) -> Any:
        """protein_sequence用のトークナイザーを初期化（抽象メソッドの実装）"""
        try:
            # protein_sequence用のEsmSequenceTokenizerを使用
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
            from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

            logger.info("Initializing EsmSequenceTokenizer for protein_sequence")
            tokenizer = EsmSequenceTokenizer()
            self.vocab_size = len(tokenizer.get_vocab())
            logger.info(f"EsmSequenceTokenizer initialized with vocab_size: {self.vocab_size}")

            # 特殊トークンIDの取得（EsmSequenceTokenizerの仕様に合わせる）
            vocab = tokenizer.get_vocab()
            self.pad_token_id = vocab.get("<pad>", 1)
            self.unk_token_id = vocab.get("<unk>", 3)
            self.mask_token_id = vocab.get("<mask>", 32)
            self.cls_token_id = vocab.get("<cls>", 0)
            self.sep_token_id = vocab.get("<eos>", 2)

            logger.info(f"Special tokens - CLS: {self.cls_token_id}, SEP: {self.sep_token_id}, MASK: {self.mask_token_id}")

            # vocab_sizeが想定と異なる場合の対処
            if self.vocab_size != 40:
                logger.warning(f"EsmSequenceTokenizer vocab_size ({self.vocab_size}) != expected (40)")
                logger.info("Using tokenizer with current vocab_size")

            self.use_esm_tokenizer = True
            return tokenizer

        except ImportError as e:
            logger.warning(f"Could not import EsmSequenceTokenizer: {e}")
            logger.info("Falling back to simple amino acid tokenizer")
            return self._init_simple_amino_acid_tokenizer()

    def _init_model(self) -> BertForMaskedLM:
        """モデルの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading BERT model from {self.model_path}")
        return self._load_model()

    def _init_simple_amino_acid_tokenizer(self) -> None:
        """シンプルなアミノ酸トークナイザーを初期化（フォールバック用）"""
        # EsmSequenceTokenizerと同じvocab構成
        vocab_list = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
            "|",
            "<mask>",
        ]
        # モデルのvocab_size=40に合わせるため、パディングトークンを追加
        while len(vocab_list) < 40:
            vocab_list.append(f"<pad_{len(vocab_list) - 33}>")

        # 語彙の構築
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)  # 40

        # 特殊トークンID
        self.cls_token_id = 0  # <cls>
        self.pad_token_id = 1  # <pad>
        self.sep_token_id = 2  # <eos>
        self.unk_token_id = 3  # <unk>
        self.mask_token_id = 32  # <mask>

        logger.info(f"Simple amino acid tokenizer initialized with vocab_size: {self.vocab_size}")

        class SimpleTokenizer:
            """シンプルなアミノ酸トークナイザー（フォールバック用）"""

            def __init__(self, vocab: Dict[str, int]) -> None:
                self.vocab: Dict[str, int] = vocab
                self.id_to_token: Dict[int, str] = {idx: token for token, idx in vocab.items()}
                self.unk_token_id: int = 3

            def encode(self, sequence: str) -> List[int]:
                """アミノ酸配列をトークンIDに変換"""
                tokens: List[int] = []
                for char in sequence.upper():
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.unk_token_id)
                return tokens

            def get_vocab(self) -> Dict[str, int]:
                return self.vocab

        self.tokenizer = SimpleTokenizer(self.vocab)
        self.use_esm_tokenizer = False

    def _load_model(self) -> BertForMaskedLM:
        """訓練済みBERTモデルの読み込み"""
        try:
            # Safetensorsファイルのチェック
            safetensors_path = os.path.join(self.model_path, "model.safetensors")
            pytorch_path = os.path.join(self.model_path, "pytorch_model.bin")

            if os.path.exists(safetensors_path):
                logger.info("Loading from safetensors format")
                from safetensors.torch import load_file

                # 設定ファイルの読み込み
                config_path = os.path.join(self.model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    config = BertConfig(**config_dict)
                else:
                    logger.info("Config file not found, using default BERT config")
                    config = BertConfig(
                        vocab_size=self.vocab_size,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        max_position_embeddings=1024,
                    )

                # モデルの作成
                model = BertForMaskedLM(config)

                # Safetensorsから重みを読み込み
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)

                logger.info("✅ Successfully loaded trained BERT model with safetensors")

            elif os.path.exists(pytorch_path):
                logger.info("Loading from PyTorch format")
                model = BertForMaskedLM.from_pretrained(self.model_path)

            else:
                raise FileNotFoundError(f"No model file found in {self.model_path}")

            model.to(self.device)
            model.eval()

            # モデル情報のログ出力
            total_params = sum(p.numel() for p in model.parameters())
            config = model.config

            logger.info("📊 Model Statistics:")
            logger.info(f"   - Total parameters: {total_params:,}")
            logger.info(f"   - Hidden size: {config.hidden_size}")
            logger.info(f"   - Number of layers: {config.num_hidden_layers}")
            logger.info(f"   - Attention heads: {config.num_attention_heads}")
            logger.info(f"   - Max sequence length: {config.max_position_embeddings}")

            return model

        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

    def encode_sequence(self, sequence: str, max_length: int = 512) -> torch.Tensor:
        """アミノ酸配列をトークンIDにエンコード"""
        # 配列を適切にフォーマット（大文字変換、無効文字の除去など）
        sequence = sequence.upper().replace("X", "").replace("-", "").replace("*", "")

        # EsmSequenceTokenizerまたはSimpleTokenizerを使用
        if self.use_esm_tokenizer:
            # EsmSequenceTokenizerを使用
            if hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(sequence)
            else:
                # Callableな場合
                result = self.tokenizer(sequence)
                if isinstance(result, dict) and "input_ids" in result:
                    tokens = result["input_ids"]
                elif hasattr(result, "input_ids"):
                    tokens = result.input_ids
                else:
                    tokens = result
        else:
            # SimpleTokenizerを使用
            tokens = self.tokenizer.encode(sequence)

        # tokensがリストでない場合の対処
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens) if tokens else []

        # 長すぎる場合は切り詰め
        if len(tokens) > max_length - 2:  # CLS, SEP用に2つ確保
            tokens = tokens[: max_length - 2]

        # BERT形式に変換: [CLS] + tokens + [SEP]
        bert_tokens = [self.cls_token_id] + tokens + [self.sep_token_id]

        # デバッグ用ログ
        logger.debug(f"Original sequence length: {len(sequence)}")
        logger.debug(f"Encoded tokens length: {len(bert_tokens)}")

        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            # 空の場合は最小形式を返す
            bert_tokens = [self.cls_token_id, self.unk_token_id, self.sep_token_id]

        return torch.tensor(bert_tokens, dtype=torch.long)

    def get_masked_lm_score(
        self,
        sequence: str,
        mutation_pos: Optional[int] = None,
        context_length: int = 512,
    ) -> float:
        """
        マスク言語モデルスコアを計算

        Args:
            sequence (str): アミノ酸配列
            mutation_pos (int): 変異位置（オプション）
            context_length (int): 評価に使用するコンテキスト長

        Returns:
            float: MLMスコア（対数尤度の平均）
        """
        with torch.no_grad():
            # 配列をエンコード
            tokens = self.encode_sequence(sequence, context_length)

            # バッチ次元を追加してデバイスに転送
            input_ids = tokens.unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # BERTモデルで予測
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 対数確率を計算
            log_probs = F.log_softmax(logits, dim=-1)

            # 各トークンの尤度を計算（CLSとSEPを除く）
            token_log_probs: List[float] = []
            for i in range(1, len(tokens) - 1):  # CLS(0)とSEP(最後)を除く
                token_id = tokens[i].item()
                token_log_prob = log_probs[0, i, token_id]
                token_log_probs.append(token_log_prob.item())

            # 平均対数尤度を返す
            if token_log_probs:
                return np.mean(token_log_probs)
            else:
                return 0.0

    def get_variant_fitness_score(
        self,
        wildtype_seq: str,
        mutant_seq: str,
        context_length: int = 512,
    ) -> Dict[str, float]:
        """
        変異のフィットネススコアを計算（BERT版）

        Args:
            wildtype_seq (str): 野生型配列
            mutant_seq (str): 変異型配列
            context_length (int): 評価に使用するコンテキスト長

        Returns:
            dict: 詳細なスコア情報
        """
        try:
            # 野生型と変異型のMLMスコアを計算
            wt_score = self.get_masked_lm_score(wildtype_seq, context_length=context_length)
            mut_score = self.get_masked_lm_score(mutant_seq, context_length=context_length)

            # フィットネススコア = 変異型 - 野生型
            # 正の値 = 変異が有益、負の値 = 変異が有害
            fitness_score = mut_score - wt_score

            # 配列表現の類似度も計算
            similarity = self.get_sequence_similarity(wildtype_seq, mutant_seq, context_length)

            return {
                "fitness_score": fitness_score,
                "wildtype_mlm_score": wt_score,
                "mutant_mlm_score": mut_score,
                "sequence_similarity": similarity,
                "bert_pathogenicity_score": self._calculate_pathogenicity_score(fitness_score, similarity),
            }

        except Exception as e:
            logger.warning(f"Error calculating fitness score: {e}")
            return {
                "fitness_score": 0.0,
                "wildtype_mlm_score": 0.0,
                "mutant_mlm_score": 0.0,
                "sequence_similarity": 1.0,
                "bert_pathogenicity_score": 0.5,
            }

    def get_sequence_similarity(self, seq1: str, seq2: str, context_length: int = 512) -> float:
        """配列表現のコサイン類似度を計算"""
        with torch.no_grad():
            # 両配列をエンコード
            tokens1 = self.encode_sequence(seq1, context_length)
            tokens2 = self.encode_sequence(seq2, context_length)

            # パディングで長さを揃える
            max_len = max(len(tokens1), len(tokens2))
            if len(tokens1) < max_len:
                tokens1 = F.pad(tokens1, (0, max_len - len(tokens1)), value=self.pad_token_id)
            if len(tokens2) < max_len:
                tokens2 = F.pad(tokens2, (0, max_len - len(tokens2)), value=self.pad_token_id)

            # バッチ化してBERTに入力
            input_ids = torch.stack([tokens1, tokens2]).to(self.device)
            attention_mask = (input_ids != self.pad_token_id).float()

            # 隠れ状態を取得
            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            # [CLS]トークンの表現を使用
            repr1 = hidden_states[0, 0, :]  # 最初の配列の[CLS]
            repr2 = hidden_states[1, 0, :]  # 二番目の配列の[CLS]

            # コサイン類似度を計算
            similarity = F.cosine_similarity(repr1.unsqueeze(0), repr2.unsqueeze(0))

            return similarity.item()

    def _calculate_pathogenicity_score(self, fitness_score: float, similarity: float) -> float:
        """病原性スコアを計算（0-1の範囲）"""
        # フィットネススコアと類似度を組み合わせて病原性スコアを算出
        # 低いフィットネス + 低い類似度 = 高い病原性

        # フィットネススコアを正規化（シグモイド関数使用）
        fitness_norm = 1 / (1 + np.exp(fitness_score))

        # 類似度を逆転（低い類似度 = 高い変化）
        dissimilarity = 1 - similarity

        # 重み付き平均（フィットネススコアにより重きを置く）
        pathogenicity = 0.7 * fitness_norm + 0.3 * dissimilarity

        return pathogenicity

    def load_proteingym_data(self, proteingym_file: str) -> pd.DataFrame:
        """
        ProteinGymデータの読み込み

        Args:
            proteingym_file (str): ProteinGymデータファイルのパス

        Returns:
            pd.DataFrame: ProteinGymデータ
        """
        logger.info(f"Loading ProteinGym data from {proteingym_file}")

        # ファイル形式に応じて読み込み
        if proteingym_file.endswith(".csv"):
            df = pd.read_csv(proteingym_file)
        elif proteingym_file.endswith(".tsv"):
            df = pd.read_csv(proteingym_file, sep="\t")
        elif proteingym_file.endswith(".json"):
            df = pd.read_json(proteingym_file)
        else:
            raise ValueError(f"Unsupported file format: {proteingym_file}")

        # 必要なカラムの確認
        required_columns = ["mutated_sequence", "DMS_score"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")

        # target_seqがない場合は、mutated_sequenceから推定
        if "target_seq" not in df.columns and "mutant" in df.columns:
            logger.info("Inferring target_seq from mutated_sequence and mutant information")
            df = self._infer_target_sequence(df)

        logger.info(f"Loaded {len(df)} ProteinGym variants")
        logger.info(f"DMS_score statistics:\n{df['DMS_score'].describe()}")

        return df

    def _infer_target_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """mutated_sequenceとmutant情報から野生型配列を推定"""

        def reverse_mutation(mutated_seq, mutant):
            if pd.isna(mutant) or mutant == "WT":
                return mutated_seq

            # 単一変異の場合のみ対応
            if ":" not in str(mutant):
                try:
                    orig_aa = mutant[0]
                    pos = int(mutant[1:-1]) - 1  # 0-indexedに変換
                    mut_aa = mutant[-1]

                    # mutated_sequenceのmut_aaをorig_aaに戻す
                    wt_seq = list(mutated_seq)
                    if pos < len(wt_seq) and wt_seq[pos] == mut_aa:
                        wt_seq[pos] = orig_aa
                        return "".join(wt_seq)
                except (ValueError, IndexError, AttributeError):
                    pass

            return mutated_seq

        df["target_seq"] = df.apply(
            lambda row: reverse_mutation(row["mutated_sequence"], row.get("mutant", "")),
            axis=1,
        )
        return df

    def evaluate_model(
        self,
        proteingym_data: pd.DataFrame,
        batch_size: int = 16,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        モデルの評価実行

        Args:
            proteingym_data (pd.DataFrame): ProteinGymデータ
            batch_size (int): バッチサイズ
            sample_size (int): 評価するサンプル数（Noneで全件）

        Returns:
            dict: 評価結果
        """
        logger.info("Starting BERT model evaluation on ProteinGym data")

        # サンプリング
        if sample_size and sample_size < len(proteingym_data):
            proteingym_data = proteingym_data.sample(n=sample_size, random_state=42)
            logger.info(f"Using sample of {sample_size} variants")

        predictions: List[float] = []
        true_scores: List[float] = []
        detailed_results: List[Dict[str, Any]] = []

        # バッチ処理で評価
        processed: int = 0
        errors: int = 0
        last_percent: int = -1
        total_variants: int = len(proteingym_data)
        total_batches: int = math.ceil(total_variants / batch_size)

        logger.info(f"Total variants to evaluate: {total_variants}")
        logger.info(f"Batch size: {batch_size} (total batches: {total_batches})")

        for batch_index, i in enumerate(
            tqdm(
                range(0, total_variants, batch_size),
                desc="Evaluating variants",
                unit="batch",
            ),
            start=1,
        ):
            # 進捗がログに残るよう、一定間隔で出力
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                logger.info(
                    f"Batch progress: {batch_index}/{total_batches} "
                    f"({batch_index / total_batches * 100:.1f}%)"
                )

            batch = proteingym_data.iloc[i : i + batch_size]

            for idx, row in batch.iterrows():
                try:
                    # 野生型と変異型配列の取得
                    if "target_seq" in row and pd.notna(row["target_seq"]):
                        wt_seq = row["target_seq"]
                        mut_seq = row["mutated_sequence"]
                    else:
                        # target_seqがない場合の処理
                        wt_seq = row["mutated_sequence"]  # 仮の処理
                        mut_seq = row["mutated_sequence"]
                        logger.warning(f"No target_seq found for variant {idx}")

                    # フィットネススコアを計算
                    result = self.get_variant_fitness_score(wt_seq, mut_seq)

                    predictions.append(result["fitness_score"])
                    true_scores.append(row["DMS_score"])

                    # 詳細結果を記録
                    detailed_results.append(
                        {
                            "variant_id": idx,
                            "mutant": row.get("mutant", ""),
                            "protein_name": row.get("protein_name", ""),
                            "true_score": row["DMS_score"],
                            "predicted_fitness": result["fitness_score"],
                            "wildtype_mlm_score": result["wildtype_mlm_score"],
                            "mutant_mlm_score": result["mutant_mlm_score"],
                            "sequence_similarity": result["sequence_similarity"],
                            "bert_pathogenicity_score": result["bert_pathogenicity_score"],
                        }
                    )

                    processed += 1

                    if total_variants > 0:
                        percent: int = int(processed / total_variants * 100)
                        if percent != last_percent:
                            logger.info(
                                f"Step: evaluation Progress: {percent}% "
                                f"({processed}/{total_variants} variants processed)"
                            )
                            last_percent = percent

                except Exception as e:
                    logger.warning(f"Error processing variant {idx}: {e}")
                    errors += 1
                    continue

        logger.info("✅ Processing completed!")
        logger.info(f"   - Successfully processed: {processed} variants")
        logger.info(f"   - Processing errors: {errors}")

        # 評価指標を計算
        predictions = np.array(predictions)
        true_scores = np.array(true_scores)

        results = self._calculate_metrics(true_scores, predictions)
        results["detailed_results"] = detailed_results
        results["processed_variants"] = processed
        results["errors"] = errors

        logger.info("BERT model evaluation completed")
        return results

    def _calculate_metrics(self, true_scores: np.ndarray, predicted_scores: np.ndarray) -> Dict[str, Any]:
        """評価指標を計算"""
        from scipy.stats import pearsonr, spearmanr

        # 相関係数
        spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
        pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)

        # 平均絶対誤差
        mae = np.mean(np.abs(true_scores - predicted_scores))

        # 平均二乗誤差の平方根
        rmse = np.sqrt(np.mean((true_scores - predicted_scores) ** 2))

        results = {
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "mae": mae,
            "rmse": rmse,
            "n_variants": len(true_scores),
            "true_score_stats": {
                "mean": np.mean(true_scores),
                "std": np.std(true_scores),
                "min": np.min(true_scores),
                "max": np.max(true_scores),
            },
            "predicted_score_stats": {
                "mean": np.mean(predicted_scores),
                "std": np.std(predicted_scores),
                "min": np.min(predicted_scores),
                "max": np.max(predicted_scores),
            },
        }

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """評価結果を保存"""
        logger.info(f"Saving results to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # 主要結果をJSON保存
        main_results = {k: v for k, v in results.items() if k != "detailed_results"}
        main_results = self._make_serializable(main_results)

        with open(os.path.join(output_dir, "bert_proteingym_results.json"), "w") as f:
            json.dump(main_results, f, indent=2)

        # 詳細結果をCSV保存
        if "detailed_results" in results:
            detailed_df = pd.DataFrame(results["detailed_results"])
            detailed_df.to_csv(
                os.path.join(output_dir, "bert_proteingym_detailed_results.csv"),
                index=False,
            )

        # レポート作成
        self._create_report(results, output_dir)

    def _create_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """評価レポートを作成"""
        report_path = os.path.join(output_dir, "bert_proteingym_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write("Independent BERT Protein Sequence Model - ProteinGym Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"🕐 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("🧬 Model Type: BERT for Masked Language Modeling\n")
            f.write("📊 Evaluation Method: Independent fitness assessment\n\n")

            f.write("📈 Dataset Summary:\n")
            f.write(f"   • Total variants evaluated: {results['n_variants']}\n")
            f.write(f"   • Successfully processed: {results['processed_variants']}\n")
            f.write(f"   • Processing errors: {results['errors']}\n\n")

            f.write("🎯 Performance Metrics:\n")
            f.write(f"   • Spearman correlation: {results['spearman_correlation']:.4f} (p={results['spearman_p_value']:.4e})\n")
            f.write(f"   • Pearson correlation: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4e})\n")
            f.write(f"   • MAE: {results['mae']:.4f}\n")
            f.write(f"   • RMSE: {results['rmse']:.4f}\n\n")

            f.write("🧠 BERT-Specific Analysis:\n")
            f.write(
                f"   • True Score Range: {results['true_score_stats']['min']:.4f} to {results['true_score_stats']['max']:.4f}\n"
            )
            f.write(
                f"   • Predicted Score Range: {results['predicted_score_stats']['min']:.4f} to {results['predicted_score_stats']['max']:.4f}\n"
            )
            f.write(
                f"   • True Score Mean: {results['true_score_stats']['mean']:.4f} ± {results['true_score_stats']['std']:.4f}\n"
            )
            f.write(
                f"   • Predicted Score Mean: {results['predicted_score_stats']['mean']:.4f} ± {results['predicted_score_stats']['std']:.4f}\n\n"
            )

            f.write("🔍 BERT Model Interpretation:\n")
            f.write("   • MLM Scores: Higher values indicate better sequence likelihood\n")
            f.write("   • Fitness Score: Mutant MLM - Wildtype MLM (positive = beneficial)\n")
            f.write("   • Sequence Similarity: Cosine similarity of [CLS] representations\n")
            f.write("   • Pathogenicity Score: Combined fitness and similarity assessment\n\n")

            # パフォーマンス評価
            spearman = results["spearman_correlation"]
            if spearman > 0.7:
                assessment = "🟢 Excellent performance"
            elif spearman > 0.5:
                assessment = "🟡 Good performance"
            elif spearman > 0.3:
                assessment = "🟠 Moderate performance"
            else:
                assessment = "🔴 Poor performance"

            f.write(f"📊 Overall Assessment: {assessment}\n")
            f.write(f"   Spearman correlation: {spearman:.4f}\n\n")

            f.write("💡 Key Insights:\n")
            f.write("   • This evaluation is independent of GPT2 or other generative models\n")
            f.write("   • BERT's bidirectional attention enables context-aware variant assessment\n")
            f.write("   • MLM predictions provide direct evidence of sequence disruption\n")
            f.write("   • Results reflect BERT's understanding of protein sequence patterns\n")

    def _make_serializable(self, obj: Any) -> Any:
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


def create_sample_proteingym_data(output_file: str) -> None:
    """
    サンプルProteinGymデータを作成（テスト用）
    """
    logger.info(f"Creating sample ProteinGym data: {output_file}")

    # サンプルデータ（実際のProteinGymデータの形式に基づく）
    sample_data = [
        {
            "mutant": "A1V",
            "mutated_sequence": "VLKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "target_seq": "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "DMS_score": 0.85,
            "protein_name": "TEST_PROTEIN_1",
        },
        {
            "mutant": "L2P",
            "mutated_sequence": "APKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "target_seq": "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "DMS_score": 0.15,
            "protein_name": "TEST_PROTEIN_1",
        },
        {
            "mutant": "K3R",
            "mutated_sequence": "ALRGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "target_seq": "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "DMS_score": 0.75,
            "protein_name": "TEST_PROTEIN_1",
        },
        {
            "mutant": "WT",
            "mutated_sequence": "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "target_seq": "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT",
            "DMS_score": 1.0,
            "protein_name": "TEST_PROTEIN_1",
        },
    ]

    df = pd.DataFrame(sample_data)
    df.to_csv(output_file, index=False)
    logger.info(f"Sample data created with {len(df)} variants")


def get_protein_tokenizer_path() -> Optional[str]:
    """
    protein_sequence用のトークナイザーパスを取得
    protein_sequenceはEsmSequenceTokenizerを使用するため、Noneを返す

    Returns:
        None: protein_sequenceはSentencePieceを使用しない
    """
    # protein_sequenceはEsmSequenceTokenizerを使用するため、
    # SentencePieceトークナイザーは不要
    logger.info("protein_sequence uses EsmSequenceTokenizer, not SentencePiece")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT ProteinGym evaluation for protein sequence model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained BERT model checkpoint",
    )
    parser.add_argument(
        "--proteingym_data",
        type=str,
        required=True,
        help="Path to ProteinGym data file (CSV/TSV/JSON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not provided)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of variants to evaluate (None for all)",
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample ProteinGym data for testing",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (auto-detect if not provided)",
    )

    args = parser.parse_args()

    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, "bert_proteingym", model_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), "bert_proteingym_evaluation")

    # サンプルデータ作成モード
    if args.create_sample_data:
        create_sample_proteingym_data(args.proteingym_data)
        logger.info("Sample data created. Run again without --create_sample_data to evaluate.")
        return

    try:
        # トークナイザーパスの取得
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        else:
            tokenizer_path = get_protein_tokenizer_path()

        # protein_sequenceはEsmSequenceTokenizerを使用するため、tokenizer_pathはNoneでも可
        if tokenizer_path and tokenizer_path != "None" and not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # Noneの場合は使用しない
        if tokenizer_path == "None":
            tokenizer_path = None

        # 評価器の初期化
        evaluator = BERTProteinGymEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
        )

        # ProteinGymデータの読み込み
        proteingym_data = evaluator.load_proteingym_data(args.proteingym_data)

        # モデル評価の実行
        results = evaluator.evaluate_model(proteingym_data, batch_size=args.batch_size, sample_size=args.sample_size)

        # 結果の表示
        logger.info("=== BERT Evaluation Results ===")
        logger.info(f"Spearman correlation: {results['spearman_correlation']:.4f} (p={results['spearman_p_value']:.4e})")
        logger.info(f"Pearson correlation: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4e})")
        logger.info(f"MAE: {results['mae']:.4f}")
        logger.info(f"RMSE: {results['rmse']:.4f}")
        logger.info(f"Number of variants: {results['n_variants']}")

        # 結果の保存
        evaluator.save_results(results, args.output_dir)

        logger.info(f"BERT evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"BERT evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
