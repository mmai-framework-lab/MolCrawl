#!/usr/bin/env python3
"""
ProteinGymデータベースを使用したprotein sequenceモデルの精度検証スクリプト

このスクリプトは、訓練済みのGPT-2 protein sequenceモデルを使って
ProteinGymデータベースのタンパク質変異適合性を評価する精度を検証します。
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import logging

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "gpt2"))

from model import GPT, GPTConfig
from utils.evaluation_output import (
    get_evaluation_output_dir,
    get_model_type_from_path,
    get_model_name_from_path,
    setup_evaluation_logging,
)
from utils.model_evaluator import ModelEvaluator

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)


class ProteinGymEvaluator(ModelEvaluator):
    """ProteinGymデータを使用したモデル評価クラス"""

    def __init__(self, model_path, tokenizer_path=None, device="cuda"):
        """
        初期化

        Args:
            model_path (str): 訓練済みモデルのパス
            tokenizer_path (str): トークナイザーのパス（protein_sequenceでは不要）
            device (str): 使用デバイス
        """
        # 親クラスの初期化（tokenizer_pathがNoneの場合はmodel_pathを使用）
        # トークナイザーとモデルは自動的に初期化される
        super().__init__(model_path, tokenizer_path or model_path, device)

    def _init_tokenizer(self):
        """protein_sequence用のトークナイザーを初期化（抽象メソッドの実装）"""
        try:
            # protein_sequence用のEsmSequenceTokenizerを使用
            import sys
            import os

            sys.path.append(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
            )
            from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

            logger.info("Initializing EsmSequenceTokenizer for protein_sequence")
            tokenizer = EsmSequenceTokenizer()
            self.vocab_size = len(tokenizer.get_vocab())
            logger.info(
                f"EsmSequenceTokenizer initialized with vocab_size: {self.vocab_size}"
            )

            # モデルが40のvocab_sizeで学習されている一方、EsmSequenceTokenizerは33のvocab_sizeの場合の対処
            if self.vocab_size != 40:
                logger.info(
                    f"EsmSequenceTokenizer vocab_size ({self.vocab_size}) != model vocab_size (40)"
                )
                logger.info(
                    "Using SimpleTokenizer with padding tokens to match model vocab_size"
                )
                return self._init_simple_amino_acid_tokenizer()

            return tokenizer

        except ImportError as e:
            logger.warning(f"Could not import EsmSequenceTokenizer: {e}")
            logger.info("Falling back to simple amino acid tokenizer")
            return self._init_simple_amino_acid_tokenizer()

    def _init_model(self):
        """モデルの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading model from {self.model_path}")
        return self._load_model()

    def _init_simple_amino_acid_tokenizer(self):
        """シンプルなアミノ酸トークナイザーを初期化（vocab_size=40）"""
        # EsmSequenceTokenizerと同じvocab構成 + モデル用にパディング
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
        self.id_to_token = {idx: token for idx, token in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)  # 40

        logger.info(
            f"Simple amino acid tokenizer initialized with vocab_size: {self.vocab_size}"
        )

        class SimpleTokenizer:
            """シンプルなアミノ酸トークナイザー（フォールバック用）"""

            def __init__(self):
                # EsmSequenceTokenizerと同じvocab構成
                vocab = [
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
                # トレーニング時のvocab_size=40に合わせるため、パディングを追加
                while len(vocab) < 40:
                    vocab.append(f"<pad_{len(vocab) - 33}>")  # パディングトークン

                self.vocab = vocab
                self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
                self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
                self.vocab_size = len(self.vocab)  # 40
                self.unk_token_id = self.token_to_id["<unk>"]

            def encode(self, sequence):
                """アミノ酸配列をトークンIDに変換"""
                tokens = []
                for char in sequence.upper():
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.unk_token_id)
                return tokens

            def get_vocab(self):
                return self.token_to_id

        tokenizer = SimpleTokenizer()
        self.use_protein_tokenizer = True
        # 実際のvocab_sizeを更新
        self.vocab_size = tokenizer.vocab_size

        return tokenizer

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

    def encode_sequence(self, sequence):
        """アミノ酸配列をトークンIDにエンコード"""
        # 配列を適切にフォーマット（大文字変換、無効文字の除去など）
        sequence = sequence.upper().replace("X", "").replace("-", "").replace("*", "")

        if self.use_protein_tokenizer:
            # EsmSequenceTokenizerまたはSimpleTokenizerを使用
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
            # SentencePieceでエンコード（フォールバック）
            tokens = self.tokenizer.encode(sequence)

        # デバッグ用ログ
        logger.debug(f"Original sequence length: {len(sequence)}")
        logger.debug(f"Encoded tokens length: {len(tokens)}")
        logger.debug(f"First 10 chars: {sequence[:10]}")
        logger.debug(f"First 5 tokens: {tokens[:5] if tokens else 'Empty'}")

        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            # 空の場合は未知トークンを返す
            if self.use_protein_tokenizer:
                tokens = (
                    [self.tokenizer.unk_token_id]
                    if hasattr(self.tokenizer, "unk_token_id")
                    else [3]
                )  # <unk> token
            else:
                tokens = (
                    [self.tokenizer.unk_id()]
                    if hasattr(self.tokenizer, "unk_id")
                    else [0]
                )

        return torch.tensor(tokens, dtype=torch.long)

    def get_variant_fitness_score(self, wildtype_seq, mutant_seq, context_length=512):
        """
        変異のフィットネススコアを計算

        Args:
            wildtype_seq (str): 野生型配列
            mutant_seq (str): 変異型配列
            context_length (int): 評価に使用するコンテキスト長

        Returns:
            float: 変異のフィットネススコア（相対的な尤度の差）
        """
        with torch.no_grad():
            # 野生型と変異型をエンコード
            wt_tokens = self.encode_sequence(wildtype_seq)
            mut_tokens = self.encode_sequence(mutant_seq)

            # 最小長を確保（1以上）
            if len(wt_tokens) == 0:
                logger.warning(
                    "Wildtype sequence tokenization resulted in empty tokens"
                )
                return 0.0
            if len(mut_tokens) == 0:
                logger.warning("Mutant sequence tokenization resulted in empty tokens")
                return 0.0

            # コンテキスト長に調整
            if len(wt_tokens) > context_length:
                wt_tokens = wt_tokens[:context_length]
            if len(mut_tokens) > context_length:
                mut_tokens = mut_tokens[:context_length]

            # バッチ次元を追加してデバイスに転送
            wt_tokens = wt_tokens.unsqueeze(0).to(self.device)
            mut_tokens = mut_tokens.unsqueeze(0).to(self.device)

            logger.debug(
                f"Model input shapes - wt: {wt_tokens.shape}, mut: {mut_tokens.shape}"
            )

            # モデルの予測確率を取得
            wt_logits, _ = self.model(wt_tokens)
            mut_logits, _ = self.model(mut_tokens)

            logger.debug(
                f"Model output shapes - wt: {wt_logits.shape}, mut: {mut_logits.shape}"
            )

            # 対数尤度を計算
            if wt_logits.size(1) == wt_tokens.size(1):
                wt_log_prob = F.log_softmax(wt_logits, dim=-1)
                mut_log_prob = F.log_softmax(mut_logits, dim=-1)

                # 各トークンの尤度を計算
                wt_likelihood = self._calculate_sequence_likelihood(
                    wt_tokens, wt_log_prob
                )
                mut_likelihood = self._calculate_sequence_likelihood(
                    mut_tokens, mut_log_prob
                )
            else:
                # トークンごとに処理
                wt_likelihood = self._calculate_likelihood_token_by_token(wt_tokens)
                mut_likelihood = self._calculate_likelihood_token_by_token(mut_tokens)

            # フィットネススコア = 変異型の尤度 - 野生型の尤度
            # 高いスコア = より良いフィットネス
            fitness_score = mut_likelihood - wt_likelihood

            return fitness_score.item()

    def _calculate_sequence_likelihood(self, tokens, log_probs):
        """配列の対数尤度を計算"""
        # 入力チェック
        if tokens.size(1) <= 1:
            logger.warning(
                f"Sequence too short for likelihood calculation: {tokens.shape}"
            )
            return torch.tensor(0.0, device=tokens.device)

        if log_probs.size(1) == 0:
            logger.warning(f"Empty log_probs: {log_probs.shape}")
            return torch.tensor(0.0, device=tokens.device)

        # 最後のトークンを除く（予測対象がないため）
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        pred_log_probs = log_probs[:, :-1, :]

        # サイズの再確認
        if pred_log_probs.size(1) != target_tokens.size(1):
            logger.warning(
                f"Size mismatch: pred_log_probs={pred_log_probs.shape}, target_tokens={target_tokens.shape}"
            )
            return torch.tensor(0.0, device=tokens.device)

        # 各位置での正解トークンの対数確率を取得
        token_log_probs = pred_log_probs.gather(2, target_tokens.unsqueeze(2)).squeeze(
            2
        )

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
                log_probs = F.log_softmax(
                    logits[:, -1:, :], dim=-1
                )  # 最後の位置の予測のみ

                # 正解トークンの対数確率
                token_log_prob = log_probs.gather(2, target.unsqueeze(2)).squeeze()
                total_log_prob += token_log_prob.item()
                count += 1

        return torch.tensor(
            total_log_prob / count if count > 0 else 0.0, device=tokens.device
        )

    def load_proteingym_data(self, proteingym_file):
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
            logger.info(
                "Inferring target_seq from mutated_sequence and mutant information"
            )
            df = self._infer_target_sequence(df)

        logger.info(f"Loaded {len(df)} ProteinGym variants")
        logger.info(f"DMS_score statistics:\n{df['DMS_score'].describe()}")

        return df

    def _infer_target_sequence(self, df):
        """mutated_sequenceとmutant情報から野生型配列を推定"""

        # 簡単な実装：最初の変異を逆変換して野生型を推定
        def reverse_mutation(mutated_seq, mutant):
            if pd.isna(mutant) or mutant == "WT":
                return mutated_seq

            # 単一変異の場合のみ対応
            if ":" not in mutant:
                try:
                    orig_aa = mutant[0]
                    pos = int(mutant[1:-1]) - 1  # 0-indexedに変換
                    mut_aa = mutant[-1]

                    # mutated_sequenceのmut_aaをorig_aaに戻す
                    wt_seq = list(mutated_seq)
                    if pos < len(wt_seq) and wt_seq[pos] == mut_aa:
                        wt_seq[pos] = orig_aa
                        return "".join(wt_seq)
                except:
                    pass

            return mutated_seq

        df["target_seq"] = df.apply(
            lambda row: reverse_mutation(
                row["mutated_sequence"], row.get("mutant", "")
            ),
            axis=1,
        )
        return df

    def evaluate_model(self, proteingym_data, batch_size=32):
        """
        モデルの評価実行

        Args:
            proteingym_data (pd.DataFrame): ProteinGymデータ
            batch_size (int): バッチサイズ

        Returns:
            dict: 評価結果
        """
        logger.info(
            f"Starting model evaluation on ProteinGym data ({len(proteingym_data)} variants)"
        )
        logger.info(f"Available columns: {list(proteingym_data.columns)}")
        logger.info(f"Sample data:\n{proteingym_data.head(3)}")

        predictions = []
        true_scores = []
        fitness_scores = []

        # バッチ処理で評価
        for i in range(0, len(proteingym_data), batch_size):
            batch = proteingym_data.iloc[i : i + batch_size]

            for idx, row in batch.iterrows():
                try:
                    # フィットネススコアを計算
                    if "target_seq" in row and pd.notna(row["target_seq"]):
                        wt_seq = row["target_seq"]
                        mut_seq = row["mutated_sequence"]
                    else:
                        # target_seqがない場合は、mutated_sequenceを使用
                        logger.warning(
                            f"Row {idx}: target_seq not found, using mutated_sequence for both WT and mutant"
                        )
                        wt_seq = row["mutated_sequence"]  # 仮の処理
                        mut_seq = row["mutated_sequence"]

                    # DMS_scoreの確認
                    if pd.isna(row["DMS_score"]):
                        logger.warning(f"Row {idx}: DMS_score is NaN, skipping")
                        continue

                    # 配列の確認
                    if pd.isna(wt_seq) or pd.isna(mut_seq):
                        logger.warning(
                            f"Row {idx}: Missing sequence data (wt: {pd.isna(wt_seq)}, mut: {pd.isna(mut_seq)})"
                        )
                        continue

                    score = self.get_variant_fitness_score(wt_seq, mut_seq)

                    fitness_scores.append(score)
                    true_scores.append(row["DMS_score"])

                    # 最初のバリアントは詳細ログ
                    if len(fitness_scores) == 1:
                        logger.info(
                            f"First variant processed successfully: score={score:.4f}, DMS_score={row['DMS_score']:.4f}"
                        )

                    logger.debug(
                        f"Processed variant {len(fitness_scores)}/{len(proteingym_data)}"
                    )

                except Exception as e:
                    logger.warning(f"Row {idx}: Error processing variant: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())
                    continue

            # より頻繁に進捗をログ出力
            batch_num = i // batch_size + 1
            if batch_num % 5 == 0 or batch_num == 1:
                logger.info(
                    f"Processed batch {batch_num}: {len(fitness_scores)}/{len(proteingym_data)} variants successfully evaluated"
                )

        # 評価指標を計算
        logger.info(
            f"Successfully processed {len(fitness_scores)} out of {len(proteingym_data)} variants"
        )

        if len(fitness_scores) < 2:
            logger.error(
                f"Insufficient data for correlation calculation. Need at least 2 samples, got {len(fitness_scores)}"
            )
            logger.error("Check if:")
            logger.error("  1. ProteinGym data file contains valid sequences")
            logger.error(
                "  2. Required columns ('mutated_sequence', 'DMS_score') are present"
            )
            logger.error("  3. Sequences can be properly tokenized")
            raise ValueError(
                f"Need at least 2 valid samples for evaluation, got {len(fitness_scores)}"
            )

        fitness_scores = np.array(fitness_scores)
        true_scores = np.array(true_scores)

        results = self._calculate_metrics(true_scores, fitness_scores)

        logger.info("Model evaluation completed")
        return results

    def _calculate_metrics(self, true_scores, predicted_scores):
        """評価指標を計算"""
        from scipy.stats import spearmanr, pearsonr

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


def create_sample_proteingym_data(
    output_file, balanced=True, positive_samples=1000, negative_samples=1000
):
    """
    サンプルProteinGymデータを作成（テスト用）

    Args:
        output_file (str): 出力ファイルパス
        balanced (bool): バランスデータを作成するか
        positive_samples (int): 陽性サンプル数
        negative_samples (int): 陰性サンプル数
    """
    logger.info(f"Creating sample ProteinGym data: {output_file}")

    if balanced:
        logger.info(
            f"Creating balanced dataset: {positive_samples} positive + {negative_samples} negative samples"
        )

        # 陽性サンプル（高いDMS_score）を生成
        positive_data = []
        base_sequence = "ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT"
        amino_acids = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]

        np.random.seed(42)  # 再現性のため

        for i in range(positive_samples):
            pos = np.random.randint(1, len(base_sequence))
            orig_aa = base_sequence[pos - 1]
            new_aa = np.random.choice([aa for aa in amino_acids if aa != orig_aa])

            mut_sequence = list(base_sequence)
            mut_sequence[pos - 1] = new_aa
            mut_sequence = "".join(mut_sequence)

            positive_data.append(
                {
                    "mutant": f"{orig_aa}{pos}{new_aa}",
                    "mutated_sequence": mut_sequence,
                    "target_seq": base_sequence,
                    "DMS_score": np.random.uniform(0.6, 1.0),  # 陽性：0.6-1.0
                    "protein_name": "TEST_PROTEIN_POSITIVE",
                }
            )

        # 陰性サンプル（低いDMS_score）を生成
        negative_data = []
        for i in range(negative_samples):
            pos = np.random.randint(1, len(base_sequence))
            orig_aa = base_sequence[pos - 1]
            new_aa = np.random.choice([aa for aa in amino_acids if aa != orig_aa])

            mut_sequence = list(base_sequence)
            mut_sequence[pos - 1] = new_aa
            mut_sequence = "".join(mut_sequence)

            negative_data.append(
                {
                    "mutant": f"{orig_aa}{pos}{new_aa}",
                    "mutated_sequence": mut_sequence,
                    "target_seq": base_sequence,
                    "DMS_score": np.random.uniform(0.0, 0.4),  # 陰性：0.0-0.4
                    "protein_name": "TEST_PROTEIN_NEGATIVE",
                }
            )

        # データを結合してシャッフル
        sample_data = positive_data + negative_data
        np.random.shuffle(sample_data)

    else:
        # 従来のサンプルデータ（少数）
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

    if balanced:
        threshold = 0.5  # 中間値
        positive_count = len(df[df["DMS_score"] >= threshold])
        negative_count = len(df[df["DMS_score"] < threshold])
        logger.info(
            f"Sample balanced data created: {len(df)} total ({positive_count} positive, {negative_count} negative)"
        )
    else:
        logger.info(f"Sample data created with {len(df)} variants")


def get_protein_tokenizer_path():
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


def main():
    parser = argparse.ArgumentParser(
        description="ProteinGym evaluation for protein sequence model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
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
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample ProteinGym data for testing",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for evaluation"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (auto-detect if not provided)",
    )

    # バランスサンプルデータ作成用オプション
    parser.add_argument(
        "--balanced_samples",
        action="store_true",
        help="Create balanced sample data (positive and negative)",
    )
    parser.add_argument(
        "--sample_positive_count",
        type=int,
        default=1000,
        help="Number of positive samples in created data (default: 1000)",
    )
    parser.add_argument(
        "--sample_negative_count",
        type=int,
        default=1000,
        help="Number of negative samples in created data (default: 1000)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing, default: None = all)",
    )

    args = parser.parse_args()

    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(
            model_type, "proteingym", model_name
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), "proteingym_evaluation")

    # サンプルデータ作成モード
    if args.create_sample_data:
        create_sample_proteingym_data(
            output_file=args.proteingym_data,
            balanced=args.balanced_samples,
            positive_samples=args.sample_positive_count,
            negative_samples=args.sample_negative_count,
        )
        if args.balanced_samples:
            logger.info(
                f"Balanced sample data created ({args.sample_positive_count} positive + {args.sample_negative_count} negative). Run again without --create_sample_data to evaluate."
            )
        else:
            logger.info(
                "Sample data created. Run again without --create_sample_data to evaluate."
            )
        return

    try:
        # トークナイザーパスの取得
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        else:
            tokenizer_path = get_protein_tokenizer_path()

        # 評価器の初期化
        evaluator = ProteinGymEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
        )

        # ProteinGymデータの読み込み
        proteingym_data = evaluator.load_proteingym_data(args.proteingym_data)

        # サンプル数の制限（テスト用）
        if args.max_samples is not None and len(proteingym_data) > args.max_samples:
            logger.info(
                f"Limiting evaluation to first {args.max_samples} samples (out of {len(proteingym_data)})"
            )
            proteingym_data = proteingym_data.head(args.max_samples)

        # モデル評価の実行
        results = evaluator.evaluate_model(proteingym_data, batch_size=args.batch_size)

        # 結果の表示
        logger.info("=== Evaluation Results ===")
        logger.info(
            f"Spearman correlation: {results['spearman_correlation']:.4f} (p={results['spearman_p_value']:.4e})"
        )
        logger.info(
            f"Pearson correlation: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4e})"
        )
        logger.info(f"MAE: {results['mae']:.4f}")
        logger.info(f"RMSE: {results['rmse']:.4f}")
        logger.info(f"Number of variants: {results['n_variants']}")

        # 結果の保存
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        evaluator.save_results(results, results_file)

        # 詳細レポートの作成
        report_file = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_file, "w") as f:
            f.write("ProteinGym Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Data: {args.proteingym_data}\n")
            f.write(f"Tokenizer: {tokenizer_path}\n")
            f.write(f"Total variants evaluated: {results['n_variants']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(
                f"  Spearman correlation: {results['spearman_correlation']:.4f} (p={results['spearman_p_value']:.4e})\n"
            )
            f.write(
                f"  Pearson correlation: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4e})\n"
            )
            f.write(f"  MAE: {results['mae']:.4f}\n")
            f.write(f"  RMSE: {results['rmse']:.4f}\n\n")
            f.write("Score Statistics:\n")
            f.write(
                f"  True scores - mean: {results['true_score_stats']['mean']:.4f}, std: {results['true_score_stats']['std']:.4f}\n"
            )
            f.write(
                f"  Predicted scores - mean: {results['predicted_score_stats']['mean']:.4f}, std: {results['predicted_score_stats']['std']:.4f}\n"
            )

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
