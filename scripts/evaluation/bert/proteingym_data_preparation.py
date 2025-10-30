#!/usr/bin/env python3
"""
BERT版ProteinGymデータセットの前処理スクリプト

BERTモデル用にProteinGymデータセットを前処理し、
評価に適した形式で保存します。
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
import zipfile
import logging
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import json
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/bert_proteingym_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BERTProteinGymDataProcessor:
    """BERT用ProteinGymデータセットの前処理クラス"""
    
    # ProteinGym v1.3データセットの公式URL
    PROTEINGYM_URLS = {
        # DMS (Deep Mutational Scanning) データ - メイン評価用
        'substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip',
        'reference_substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv',
    }
    
    def __init__(self, output_dir='./bert_proteingym_data'):
        """
        初期化
        
        Args:
            output_dir (str): 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログディレクトリの作成
        Path('logs').mkdir(exist_ok=True)
    
    def download_file(self, url, filename=None):
        """
        ファイルをダウンロード
        
        Args:
            url (str): ダウンロードURL
            filename (str): 保存ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: ダウンロードしたファイルのパス
        """
        if filename is None:
            filename = Path(urlparse(url).path).name
        
        filepath = self.output_dir / filename
        
        logger.info(f"Downloading {url} to {filepath}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)
        
        logger.info(f"Downloaded: {filepath}")
        return str(filepath)
    
    def extract_zip(self, zip_path):
        """
        ZIPファイルを展開
        
        Args:
            zip_path (str): ZIPファイルのパス
            
        Returns:
            str: 展開されたディレクトリのパス
        """
        extract_dir = self.output_dir / Path(zip_path).stem
        extract_dir.mkdir(exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted to: {extract_dir}")
        return str(extract_dir)
    
    def load_proteingym_data(self, data_dir, reference_csv=None):
        """
        ProteinGymデータを読み込み
        
        Args:
            data_dir (str): データディレクトリ
            reference_csv (str): 参照CSVファイル（オプション）
            
        Returns:
            dict: データセット辞書
        """
        logger.info(f"Loading ProteinGym data from {data_dir}")
        
        data_dir = Path(data_dir)
        datasets = {}
        
        # ProteinGym構造を確認
        dms_dir = None
        possible_dirs = [
            data_dir / "DMS_ProteinGym_substitutions" / "DMS_ProteinGym_substitutions",
            data_dir / "DMS_ProteinGym_substitutions",
            data_dir
        ]
        
        for possible_dir in possible_dirs:
            if possible_dir.exists() and any(possible_dir.glob('*.csv')):
                dms_dir = possible_dir
                logger.info(f"Found ProteinGym data in: {dms_dir}")
                break
        
        if not dms_dir:
            logger.error(f"No ProteinGym CSV files found in {data_dir} or subdirectories")
            return datasets
        
        # DMS CSVファイルを読み込み
        csv_files = list(dms_dir.glob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataset_name = csv_file.stem
                datasets[dataset_name] = df
                logger.info(f"Loaded {dataset_name}: {len(df)} variants")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        # 参照情報を読み込み（オプション）
        if reference_csv and os.path.exists(reference_csv):
            try:
                ref_df = pd.read_csv(reference_csv)
                datasets['reference'] = ref_df
                logger.info(f"Loaded reference data: {len(ref_df)} assays")
            except Exception as e:
                logger.warning(f"Failed to load reference CSV: {e}")
        
        return datasets
    
    def preprocess_for_bert(self, datasets, max_variants_per_assay=1000):
        """
        BERT評価用にデータを前処理
        
        Args:
            datasets (dict): データセット辞書
            max_variants_per_assay (int): アッセイごとの最大変異数
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        logger.info("Starting BERT-specific preprocessing")
        
        all_variants = []
        
        for dataset_name, df in datasets.items():
            if dataset_name == 'reference':
                continue
            
            logger.info(f"Processing dataset: {dataset_name}")
            
            # 必要なカラムの確認
            required_columns = ['mutated_sequence', 'DMS_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in {dataset_name}: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                continue
            
            # データの基本的なフィルタリング
            df_clean = df.dropna(subset=required_columns).copy()
            
            # DMS_scoreの有効性チェック
            df_clean = df_clean[
                (df_clean['DMS_score'].notna()) & 
                (np.isfinite(df_clean['DMS_score']))
            ].copy()
            
            # 配列長の妥当性チェック
            df_clean = df_clean[
                (df_clean['mutated_sequence'].str.len() > 10) & 
                (df_clean['mutated_sequence'].str.len() < 2000)  # BERTの制限を考慮
            ].copy()
            
            # アッセイごとのサンプリング
            if len(df_clean) > max_variants_per_assay:
                df_clean = df_clean.sample(n=max_variants_per_assay, random_state=42)
                logger.info(f"Sampled {max_variants_per_assay} variants from {dataset_name}")
            
            # データセット名を追加
            df_clean['assay_name'] = dataset_name
            
            # 野生型配列の推定（必要に応じて）
            if 'target_seq' not in df_clean.columns:
                df_clean = self._infer_wildtype_sequences(df_clean)
            
            all_variants.append(df_clean)
            logger.info(f"Processed {len(df_clean)} variants from {dataset_name}")
        
        if not all_variants:
            raise ValueError("No valid datasets found")
        
        # 全データを結合
        combined_df = pd.concat(all_variants, ignore_index=True)
        
        # 最終的なクリーニング
        combined_df = self._final_cleaning(combined_df)
        
        logger.info(f"Total processed variants: {len(combined_df)}")
        return combined_df
    
    def _infer_wildtype_sequences(self, df):
        """
        野生型配列を推定
        
        Args:
            df (pd.DataFrame): 変異データ
            
        Returns:
            pd.DataFrame: 野生型配列が追加されたデータ
        """
        logger.info("Inferring wildtype sequences")
        
        def infer_wildtype(mutated_seq, mutant_info):
            """単一変異から野生型を推定"""
            if pd.isna(mutant_info) or mutant_info == 'WT':
                return mutated_seq
            
            # 単一変異の場合のみ対応 (例: A1V)
            if isinstance(mutant_info, str) and len(mutant_info) >= 3:
                try:
                    orig_aa = mutant_info[0]
                    pos = int(mutant_info[1:-1]) - 1  # 0-indexedに変換
                    mut_aa = mutant_info[-1]
                    
                    # 変異配列を野生型に戻す
                    wt_seq = list(mutated_seq)
                    if 0 <= pos < len(wt_seq) and wt_seq[pos] == mut_aa:
                        wt_seq[pos] = orig_aa
                        return ''.join(wt_seq)
                except:
                    pass
            
            return mutated_seq
        
        # mutantカラムがある場合
        if 'mutant' in df.columns:
            df['target_seq'] = df.apply(
                lambda row: infer_wildtype(row['mutated_sequence'], row.get('mutant', '')), 
                axis=1
            )
        else:
            # mutant情報がない場合は、mutated_sequenceをそのまま使用
            df['target_seq'] = df['mutated_sequence']
            logger.warning("No mutant information found, using mutated_sequence as target_seq")
        
        return df
    
    def _final_cleaning(self, df):
        """
        最終的なデータクリーニング
        
        Args:
            df (pd.DataFrame): 前処理データ
            
        Returns:
            pd.DataFrame: クリーニング済みデータ
        """
        logger.info("Final data cleaning")
        
        # 重複除去
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['mutated_sequence'], keep='first')
        after_dedup = len(df)
        logger.info(f"Removed {before_dedup - after_dedup} duplicate sequences")
        
        # アミノ酸配列の妥当性チェック
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        def is_valid_protein_sequence(seq):
            """有効なタンパク質配列かチェック"""
            if not isinstance(seq, str):
                return False
            return all(aa in valid_aa for aa in seq.upper())
        
        valid_seq_mask = df['mutated_sequence'].apply(is_valid_protein_sequence)
        df = df[valid_seq_mask].copy()
        logger.info(f"Retained {len(df)} variants with valid amino acid sequences")
        
        # DMS_scoreの統計情報をログ出力
        logger.info(f"DMS_score statistics:\n{df['DMS_score'].describe()}")
        
        return df
    
    def save_bert_ready_data(self, df, filename='bert_proteingym_dataset.csv'):
        """
        BERT用データとして保存
        
        Args:
            df (pd.DataFrame): 前処理済みデータ
            filename (str): 保存ファイル名
        """
        filepath = self.output_dir / filename
        
        # CSV形式で保存
        df.to_csv(filepath, index=False)
        logger.info(f"Saved BERT-ready dataset: {filepath}")
        
        # JSON形式でも保存（メタデータ付き）
        json_data = {
            'metadata': {
                'total_variants': len(df),
                'unique_assays': df['assay_name'].nunique() if 'assay_name' in df.columns else 1,
                'dms_score_range': [float(df['DMS_score'].min()), float(df['DMS_score'].max())],
                'avg_sequence_length': float(df['mutated_sequence'].str.len().mean()),
                'processing_date': datetime.now().isoformat()
            },
            'data': df.to_dict('records')[:100]  # 最初の100件のみJSONに含める
        }
        
        json_filepath = filepath.with_suffix('.json')
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved metadata and sample data: {json_filepath}")
        
        # 統計レポートも作成
        self._create_statistics_report(df)
    
    def _create_statistics_report(self, df):
        """
        統計レポートを作成
        
        Args:
            df (pd.DataFrame): データセット
        """
        report_file = self.output_dir / 'bert_proteingym_statistics.txt'
        
        with open(report_file, 'w') as f:
            f.write("BERT ProteinGym Dataset Statistics Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total variants: {len(df)}\n")
            f.write(f"Unique assays: {df['assay_name'].nunique() if 'assay_name' in df.columns else 'N/A'}\n\n")
            
            f.write("DMS Score Distribution:\n")
            f.write(f"{df['DMS_score'].describe()}\n\n")
            
            f.write("Sequence Length Distribution:\n")
            f.write(f"{df['mutated_sequence'].str.len().describe()}\n\n")
            
            if 'assay_name' in df.columns:
                f.write("Top 10 Assays by Variant Count:\n")
                for assay, count in df['assay_name'].value_counts().head(10).items():
                    f.write(f"  {assay}: {count}\n")
                f.write("\n")
            
            f.write("Sample Sequences (first 5):\n")
            for i, seq in enumerate(df['mutated_sequence'].head(5)):
                f.write(f"  {i+1}: {seq[:50]}{'...' if len(seq) > 50 else ''}\n")
        
        logger.info(f"Statistics report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='BERT ProteinGym data preprocessing')
    parser.add_argument('--output_dir', type=str, default='./bert_proteingym_data',
                       help='Output directory for processed data')
    parser.add_argument('--download', action='store_true',
                       help='Download ProteinGym data')
    parser.add_argument('--max_variants_per_assay', type=int, default=1000,
                       help='Maximum variants per assay')
    parser.add_argument('--sample_only', action='store_true',
                       help='Create sample dataset only')
    parser.add_argument('--data_dir', type=str, default='./bert_proteingym_data',
                       help='Directory containing ProteinGym data')
    
    args = parser.parse_args()
    
    processor = BERTProteinGymDataProcessor(args.output_dir)
    
    try:
        if args.sample_only:
            # サンプルデータセットを作成
            logger.info("Creating sample dataset")
            sample_data = [
                {
                    'mutant': 'A1V',
                    'mutated_sequence': 'VLKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'target_seq': 'ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'DMS_score': 0.85,
                    'assay_name': 'SAMPLE_PROTEIN'
                },
                {
                    'mutant': 'L2P',
                    'mutated_sequence': 'APKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'target_seq': 'ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'DMS_score': 0.15,
                    'assay_name': 'SAMPLE_PROTEIN'
                },
                {
                    'mutant': 'WT',
                    'mutated_sequence': 'ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'target_seq': 'ALKGDLSGLTQVKSGQDKGLTRVKDDLSVLTQVKSGQDKGLT',
                    'DMS_score': 1.0,
                    'assay_name': 'SAMPLE_PROTEIN'
                }
            ]
            
            df_sample = pd.DataFrame(sample_data)
            processor.save_bert_ready_data(df_sample, 'bert_proteingym_sample.csv')
            logger.info("Sample dataset created")
            return
        
        if args.download:
            # データのダウンロード
            logger.info("Downloading ProteinGym data")
            
            # 置換変異データをダウンロード
            zip_file = processor.download_file(
                processor.PROTEINGYM_URLS['substitutions']
            )
            
            # 参照データをダウンロード
            ref_file = processor.download_file(
                processor.PROTEINGYM_URLS['reference_substitutions']
            )
            
            # ZIPを展開
            extract_dir = processor.extract_zip(zip_file)
            
            # データを読み込み
            datasets = processor.load_proteingym_data(extract_dir, ref_file)
        else:
            # 既存データを使用
            data_dir = args.data_dir
            if not Path(data_dir).exists():
                raise ValueError(f"Data directory not found: {data_dir}")
            
            datasets = processor.load_proteingym_data(data_dir)
        
        # BERT用に前処理
        processed_df = processor.preprocess_for_bert(
            datasets, 
            max_variants_per_assay=args.max_variants_per_assay
        )
        
        # 保存
        processor.save_bert_ready_data(processed_df)
        
        logger.info("BERT ProteinGym data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()