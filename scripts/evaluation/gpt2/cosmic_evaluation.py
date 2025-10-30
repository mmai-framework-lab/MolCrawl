#!/usr/bin/env python3
"""
COSMICデータを使用したgenome sequenceモデル評価スクリプト

COSMICデータベースの癌関連変異データを使用して、
genome sequenceモデルの変異病原性予測性能を評価します。
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import json
from pathlib import Path

# プロジェクトルートを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'gpt2'))

from config.paths import get_genome_tokenizer_path, get_gpt2_output_path
from model import GPT, GPTConfig
from utils.evaluation_output import get_evaluation_output_dir, get_model_type_from_path, get_model_name_from_path, setup_evaluation_logging
from utils.model_evaluator import ModelEvaluator

# ログ設定は後でsetup_evaluation_loggingで行う
logger = logging.getLogger(__name__)

class COSMICEvaluator(ModelEvaluator):
    """COSMICデータを使用したモデル評価クラス"""
    
    def __init__(self, model_path, tokenizer_path=None, device=None):
        """
        初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
            tokenizer_path (str): トークナイザーのパス
            device (str): 使用するデバイス ('cuda' or 'cpu')
        """
        # 親クラスの初期化
        super().__init__(
            model_path, 
            tokenizer_path or get_genome_tokenizer_path(), 
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # サブクラス固有の初期化
        self.tokenizer = self._init_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size()
        self.model = self._init_model()
        
        logger.info(f"Model loaded successfully. Config: {self.model.config}")
    
    def _init_tokenizer(self):
        """トークナイザーの初期化（抽象メソッドの実装）"""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        return spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        
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
        """モデルをロード"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # モデル設定の復元
        model_args = checkpoint.get('model_args', {})
        config = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=model_args.get('block_size', 1024),
            n_layer=model_args.get('n_layer', 12),
            n_head=model_args.get('n_head', 12),
            n_embd=model_args.get('n_embd', 768),
            dropout=0.0,  # 評価時はdropoutを無効
            bias=model_args.get('bias', True)
        )
        
        # モデルの作成と重みの読み込み
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def encode_sequence(self, sequence):
        """DNA配列をトークンIDにエンコード"""
        # 配列を適切にフォーマット
        sequence = sequence.upper().replace('N', '').replace('-', '')
        
        # SentencePieceでエンコード
        tokens = self.tokenizer.encode(sequence)
        
        if not tokens:
            logger.warning(f"Empty tokenization for sequence: {sequence[:50]}...")
            tokens = [self.tokenizer.unk_id()] if hasattr(self.tokenizer, 'unk_id') else [0]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def get_oncogenic_probability(self, reference_seq, variant_seq, context_length=512):
        """
        変異の癌原性確率を計算
        
        Args:
            reference_seq (str): 参照配列
            variant_seq (str): 変異配列
            context_length (int): 評価に使用するコンテキスト長
            
        Returns:
            float: 変異の癌原性確率
        """
        with torch.no_grad():
            # 参照配列と変異配列をエンコード
            ref_tokens = self.encode_sequence(reference_seq)
            var_tokens = self.encode_sequence(variant_seq)
            
            # 最小長チェック
            if len(ref_tokens) == 0 or len(var_tokens) == 0:
                return 0.0
            
            # コンテキスト長に調整
            if len(ref_tokens) > context_length:
                ref_tokens = ref_tokens[:context_length]
            if len(var_tokens) > context_length:
                var_tokens = var_tokens[:context_length]
            
            # バッチ次元を追加してデバイスに転送
            ref_tokens = ref_tokens.unsqueeze(0).to(self.device)
            var_tokens = var_tokens.unsqueeze(0).to(self.device)
            
            # 尤度を計算
            ref_likelihood = self._calculate_likelihood_token_by_token(ref_tokens)
            var_likelihood = self._calculate_likelihood_token_by_token(var_tokens)
            
            # 尤度比から癌原性確率を計算
            likelihood_ratio = var_likelihood - ref_likelihood
            oncogenic_prob = torch.sigmoid(likelihood_ratio * 10).item()  # スケーリング
            
            return oncogenic_prob
    
    def _calculate_likelihood_token_by_token(self, tokens):
        """トークンごとに尤度を計算"""
        if tokens.size(1) <= 1:
            return torch.tensor(0.0, device=tokens.device)
        
        total_log_prob = 0.0
        count = 0
        
        # 各位置での条件付き確率を計算
        for i in range(1, tokens.size(1)):
            context = tokens[:, :i]
            target = tokens[:, i:i+1]
            
            with torch.no_grad():
                logits, _ = self.model(context)
                log_probs = F.log_softmax(logits[:, -1:, :], dim=-1)
                
                # 正解トークンの対数確率
                token_log_prob = log_probs.gather(2, target.unsqueeze(2)).squeeze()
                total_log_prob += token_log_prob.item()
                count += 1
        
        return torch.tensor(total_log_prob / count if count > 0 else 0.0, device=tokens.device)
    
    def load_cosmic_data(self, data_path):
        """
        COSMICデータを読み込み
        
        Args:
            data_path (str): COSMICデータファイルのパス
            
        Returns:
            pd.DataFrame: 処理されたCOSMICデータ
        """
        logger.info(f"Loading COSMIC data from {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} COSMIC variants")
        
        # 必要なカラムの確認
        required_columns = ['Reference_sequence', 'Variant_sequence', 'Cancer_significance']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {df.columns.tolist()}")
        
        # 癌原性ラベルの標準化
        df = self._standardize_cancer_significance(df)
        
        logger.info(f"Cancer significance distribution:\n{df['oncogenic'].value_counts()}")
        
        return df
    
    def _standardize_cancer_significance(self, df):
        """癌原性の臨床的意義を標準化"""
        
        def classify_oncogenicity(significance):
            significance = str(significance).lower()
            
            # 癌原性パターン
            oncogenic_terms = ['pathogenic', 'oncogenic', 'likely_pathogenic', 'driver']
            # 非癌原性パターン
            benign_terms = ['benign', 'likely_benign', 'neutral', 'passenger']
            
            if any(term in significance for term in oncogenic_terms):
                return 1  # 癌原性
            elif any(term in significance for term in benign_terms):
                return 0  # 非癌原性
            else:
                return None  # 不明（評価から除外）
        
        df['oncogenic'] = df['Cancer_significance'].apply(classify_oncogenicity)
        
        # 不明なものを除外
        df = df.dropna(subset=['oncogenic'])
        df['oncogenic'] = df['oncogenic'].astype(int)
        
        return df
    
    def evaluate_model(self, cosmic_data, batch_size=16):
        """
        モデルの評価を実行
        
        Args:
            cosmic_data (pd.DataFrame): COSMICデータ
            batch_size (int): バッチサイズ
            
        Returns:
            dict: 評価結果
        """
        logger.info("Starting model evaluation on COSMIC data")
        
        oncogenicity_scores = []
        true_labels = []
        
        for idx, row in cosmic_data.iterrows():
            try:
                # 癌原性確率を計算
                score = self.get_oncogenic_probability(
                    row['Reference_sequence'],
                    row['Variant_sequence']
                )
                
                oncogenicity_scores.append(score)
                true_labels.append(row['oncogenic'])
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(cosmic_data)} variants")
                
            except Exception as e:
                logger.warning(f"Error processing variant {idx}: {e}")
                continue
        
        if not oncogenicity_scores:
            raise ValueError("No valid oncogenicity scores computed")
        
        # 配列に変換
        oncogenicity_scores = np.array(oncogenicity_scores)
        true_labels = np.array(true_labels)
        
        # 最適な閾値を決定
        optimal_threshold = self._find_optimal_threshold(oncogenicity_scores, true_labels)
        
        # 予測ラベルを生成
        predicted_labels = (oncogenicity_scores >= optimal_threshold).astype(int)
        
        # 評価指標を計算
        results = self._calculate_metrics(
            true_labels, predicted_labels, oncogenicity_scores, optimal_threshold
        )
        
        logger.info("Model evaluation completed")
        
        return results
    
    def _find_optimal_threshold(self, scores, labels):
        """ROC曲線からF1スコアを最大化する閾値を決定"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # F1スコアを最大化する閾値を探す
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _calculate_metrics(self, true_labels, predicted_labels, scores, threshold):
        """評価指標を計算"""
        
        # 基本指標
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except ValueError:
            roc_auc = 0.5
        
        # PR-AUC
        try:
            pr_auc = average_precision_score(true_labels, scores)
        except ValueError:
            pr_auc = 0.5
        
        # 混同行列
        cm = confusion_matrix(true_labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Sensitivity (Recall) と Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'optimal_threshold': threshold,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': len(true_labels),
            'oncogenic_samples': int(np.sum(true_labels)),
            'non_oncogenic_samples': int(len(true_labels) - np.sum(true_labels))
        }
        
        return results
    
    def save_results(self, results, output_dir):
        """結果を保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で詳細結果を保存
        results_file = output_dir / 'cosmic_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # テキスト形式で要約を保存
        report_file = output_dir / 'cosmic_evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write("COSMIC Oncogenicity Prediction Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {results['total_samples']}\n")
            f.write(f"Oncogenic samples: {results['oncogenic_samples']}\n")
            f.write(f"Non-oncogenic samples: {results['non_oncogenic_samples']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1-score: {results['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC: {results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {results['pr_auc']:.4f}\n")
            f.write(f"  Sensitivity: {results['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {results['specificity']:.4f}\n")
            f.write(f"  Optimal threshold: {results['optimal_threshold']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives: {results['true_positives']}\n")
            f.write(f"  False Positives: {results['false_positives']}\n")
            f.write(f"  True Negatives: {results['true_negatives']}\n")
            f.write(f"  False Negatives: {results['false_negatives']}\n")
        
        logger.info(f"Results saved to {output_dir}")
        
        return results_file, report_file

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='COSMIC-based genome sequence model evaluation')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--cosmic_data', required=True, help='Path to COSMIC evaluation dataset')
    parser.add_argument('--output_dir', default=None, help='Output directory (auto-generated if not provided)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを自動生成または指定されたものを使用
    if args.output_dir is None:
        model_type = get_model_type_from_path(args.model_path)
        model_name = get_model_name_from_path(args.model_path)
        args.output_dir = get_evaluation_output_dir(model_type, 'cosmic', model_name)
    
    # ログ設定
    logger = setup_evaluation_logging(Path(args.output_dir), 'cosmic_evaluation')
    
    try:
        # トークナイザーパスの取得
        tokenizer_path = get_genome_tokenizer_path()
        
        # 評価器の初期化
        evaluator = COSMICEvaluator(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            device=args.device
        )
        
        # COSMICデータの読み込み
        cosmic_data = evaluator.load_cosmic_data(args.cosmic_data)
        
        # モデル評価
        results = evaluator.evaluate_model(cosmic_data, batch_size=args.batch_size)
        
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
        evaluator.save_results(results, args.output_dir)
        
        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
