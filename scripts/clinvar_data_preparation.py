#!/usr/bin/env python3
"""
ClinVarデータダウンロード・前処理スクリプト

ClinVarデータベースから病原性変異データをダウンロードし、
genome sequenceモデルの評価に適した形式に前処理します。
"""

import sys
import os
import argparse
import pandas as pd
import requests
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/clinvar_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinVarProcessor:
    """ClinVarデータの取得・前処理クラス"""
    
    def __init__(self, output_dir='./clinvar_data'):
        """
        初期化
        
        Args:
            output_dir (str): 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ClinVarダウンロードURL
        self.clinvar_urls = {
            'variant_summary': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz',
            'submission_summary': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz'
        }
    
    def download_clinvar_data(self, data_type='variant_summary'):
        """
        ClinVarデータをダウンロード
        
        Args:
            data_type (str): ダウンロードするデータタイプ
            
        Returns:
            str: ダウンロードしたファイルのパス
        """
        if data_type not in self.clinvar_urls:
            raise ValueError(f"Unknown data type: {data_type}")
        
        url = self.clinvar_urls[data_type]
        filename = self.output_dir / f"{data_type}.txt.gz"
        
        logger.info(f"Downloading {data_type} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {filename}")
        return str(filename)
    
    def extract_and_load_data(self, gzip_file):
        """
        圧縮ファイルを展開してDataFrameとして読み込み
        
        Args:
            gzip_file (str): gzipファイルのパス
            
        Returns:
            pd.DataFrame: 読み込まれたデータ
        """
        logger.info(f"Extracting and loading {gzip_file}")
        
        with gzip.open(gzip_file, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep='\t', low_memory=False)
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def filter_single_nucleotide_variants(self, df):
        """
        単一ヌクレオチド変異（SNV）のみをフィルタリング
        
        Args:
            df (pd.DataFrame): ClinVarデータ
            
        Returns:
            pd.DataFrame: フィルタリングされたデータ
        """
        logger.info(f"Starting with {len(df)} total variants")
        
        # 利用可能な変異タイプを確認
        logger.info(f"Available variant types: {df['Type'].value_counts().head(10)}")
        
        # 変異タイプでフィルタリング
        snv_types = ['single nucleotide variant', 'SNV']
        df_snv = df[df['Type'].isin(snv_types)].copy()
        logger.info(f"After type filtering: {len(df_snv)} variants")
        
        # "na"値を除外して有効なアリル情報のみをフィルタリング
        valid_alleles = (
            (df_snv['ReferenceAllele'].notna()) &
            (df_snv['AlternateAllele'].notna()) &
            (df_snv['ReferenceAllele'] != 'na') &
            (df_snv['AlternateAllele'] != 'na') &
            (df_snv['ReferenceAllele'] != '') &
            (df_snv['AlternateAllele'] != '') &
            (df_snv['ReferenceAllele'] != '-') &
            (df_snv['AlternateAllele'] != '-')
        )
        
        df_snv = df_snv[valid_alleles].copy()
        logger.info(f"After removing 'na' and invalid alleles: {len(df_snv)} variants")
        
        # 元のフィルタリング条件
        if len(df_snv) > 0:
            valid_ref_len = df_snv['ReferenceAllele'].str.len() == 1
            valid_alt_len = df_snv['AlternateAllele'].str.len() == 1
            valid_ref_base = df_snv['ReferenceAllele'].isin(['A', 'T', 'G', 'C'])
            valid_alt_base = df_snv['AlternateAllele'].isin(['A', 'T', 'G', 'C'])
            
            logger.info(f"Valid reference length: {valid_ref_len.sum()}")
            logger.info(f"Valid alternate length: {valid_alt_len.sum()}")
            logger.info(f"Valid reference bases: {valid_ref_base.sum()}")
            logger.info(f"Valid alternate bases: {valid_alt_base.sum()}")
            
            df_snv = df_snv[valid_ref_len & valid_alt_len & valid_ref_base & valid_alt_base].copy()
            logger.info(f"After allele filtering: {len(df_snv)} variants")
            
            # フィルタリング後のサンプル値を確認
            logger.info(f"Final ReferenceAllele values: {df_snv['ReferenceAllele'].value_counts()}")
            logger.info(f"Final AlternateAllele values: {df_snv['AlternateAllele'].value_counts()}")
        
        return df_snv
    
    def filter_by_clinical_significance(self, df):
        """
        臨床的意義でフィルタリング
        
        Args:
            df (pd.DataFrame): ClinVarデータ
            
        Returns:
            pd.DataFrame: フィルタリングされたデータ
        """
        logger.info(f"Starting clinical significance filtering with {len(df)} variants")
    
        # 実際のClinicalSignificance値を確認
        logger.info(f"Available clinical significance values:")
        for value, count in df['ClinicalSignificance'].value_counts().head(20).items():
            logger.info(f"  {value}: {count}")
    
        clear_classifications = [
            'Pathogenic', 'Likely pathogenic',
            'Benign', 'Likely benign'
        ]
        
        df_filtered = df[df['ClinicalSignificance'].isin(clear_classifications)].copy()
        logger.info(f"After clinical significance filtering: {len(df_filtered)} variants")
        
        return df_filtered
    
    def get_reference_sequences(self, df, sequence_length=50):
        """
        参照配列を取得（簡略版 - 実際の実装では外部APIを使用）
        
        Args:
            df (pd.DataFrame): ClinVarデータ
            sequence_length (int): 取得する配列長
            
        Returns:
            pd.DataFrame: 参照配列が追加されたデータ
        """
        logger.info("Generating reference sequences (mock implementation)")
        
        # 実際の実装では、Ensembl REST APIやNCBI APIを使用して
        # 実際の参照配列を取得する必要があります
        # ここではサンプル実装として、ランダムな配列を生成します
        
        import random
        
        def generate_mock_sequence(length):
            """モック用のDNA配列を生成"""
            bases = ['A', 'T', 'G', 'C']
            return ''.join(random.choices(bases, k=length))
        
        def create_variant_sequence(ref_seq, position, ref_allele, alt_allele):
            """変異配列を作成"""
            # 配列の中央付近に変異を配置
            mid_pos = len(ref_seq) // 2
            variant_seq = ref_seq[:mid_pos] + alt_allele + ref_seq[mid_pos+1:]
            return variant_seq
        
        reference_sequences = []
        variant_sequences = []
        
        for _, row in df.iterrows():
            # モック参照配列を生成
            ref_seq = generate_mock_sequence(sequence_length)
            
            # 変異配列を作成
            var_seq = create_variant_sequence(
                ref_seq, 
                sequence_length // 2,  # 中央位置
                row['ReferenceAllele'],
                row['AlternateAllele']
            )
            
            reference_sequences.append(ref_seq)
            variant_sequences.append(var_seq)
        
        df_with_sequences = df.copy()
        df_with_sequences['reference_sequence'] = reference_sequences
        df_with_sequences['variant_sequence'] = variant_sequences
        
        logger.info("Reference sequences generated (mock)")
        return df_with_sequences
    
    def prepare_evaluation_dataset(self, df, max_samples_per_class=1000):
        """
        評価用データセットを準備
        
        Args:
            df (pd.DataFrame): 前処理されたClinVarデータ
            max_samples_per_class (int): クラスごとの最大サンプル数
            
        Returns:
            pd.DataFrame: 評価用データセット
        """
        logger.info("Preparing evaluation dataset")
        
        # 病原性/非病原性の二値分類に変換
        pathogenic_terms = ['Pathogenic', 'Likely pathogenic']
        benign_terms = ['Benign', 'Likely benign']
        
        df_pathogenic = df[df['ClinicalSignificance'].isin(pathogenic_terms)].copy()
        df_benign = df[df['ClinicalSignificance'].isin(benign_terms)].copy()
        
        # サンプル数を制限
        if len(df_pathogenic) > max_samples_per_class:
            df_pathogenic = df_pathogenic.sample(n=max_samples_per_class, random_state=42)
        
        if len(df_benign) > max_samples_per_class:
            df_benign = df_benign.sample(n=max_samples_per_class, random_state=42)
        
        # 結合
        df_eval = pd.concat([df_pathogenic, df_benign], ignore_index=True)
        
        # 標準化されたラベルを追加
        df_eval['pathogenic'] = df_eval['ClinicalSignificance'].apply(
            lambda x: 1 if x in pathogenic_terms else 0
        )
        
        # 必要なカラムのみを保持
        columns_to_keep = [
            'VariationID', 'GeneSymbol', 'ClinicalSignificance', 'pathogenic',
            'Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele',
            'reference_sequence', 'variant_sequence'
        ]
        
        df_eval = df_eval[columns_to_keep].copy()
        
        # データをシャッフル
        df_eval = df_eval.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Prepared evaluation dataset with {len(df_eval)} variants")
        logger.info(f"Pathogenic: {len(df_pathogenic)}, Benign: {len(df_benign)}")
        
        return df_eval
    
    def save_dataset(self, df, filename):
        """
        データセットを保存
        
        Args:
            df (pd.DataFrame): 保存するデータセット
            filename (str): ファイル名
        """
        filepath = self.output_dir / filename
        
        # 複数の形式で保存
        df.to_csv(str(filepath).replace('.csv', '.csv'), index=False)
        df.to_json(str(filepath).replace('.csv', '.json'), orient='records', indent=2)
        
        logger.info(f"Dataset saved to {filepath} (CSV and JSON formats)")
    
    def create_statistics_report(self, df):
        """
        データセットの統計レポートを作成
        
        Args:
            df (pd.DataFrame): 評価用データセット
        """
        report_file = self.output_dir / 'dataset_statistics.txt'
        
        with open(report_file, 'w') as f:
            f.write("ClinVar Dataset Statistics Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total variants: {len(df)}\n")
            f.write(f"Pathogenic variants: {len(df[df['pathogenic'] == 1])}\n")
            f.write(f"Benign variants: {len(df[df['pathogenic'] == 0])}\n\n")
            
            f.write("Clinical Significance Distribution:\n")
            for sig, count in df['ClinicalSignificance'].value_counts().items():
                f.write(f"  {sig}: {count}\n")
            f.write("\n")
            
            f.write("Chromosome Distribution (top 10):\n")
            for chrom, count in df['Chromosome'].value_counts().head(10).items():
                f.write(f"  {chrom}: {count}\n")
            f.write("\n")
            
            f.write("Gene Distribution (top 10):\n")
            for gene, count in df['GeneSymbol'].value_counts().head(10).items():
                f.write(f"  {gene}: {count}\n")
            f.write("\n")
            
            f.write("Reference/Alternate Allele Distribution:\n")
            f.write("Reference Alleles:\n")
            for allele, count in df['ReferenceAllele'].value_counts().items():
                f.write(f"  {allele}: {count}\n")
            f.write("Alternate Alleles:\n")
            for allele, count in df['AlternateAllele'].value_counts().items():
                f.write(f"  {allele}: {count}\n")
        
        logger.info(f"Statistics report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='ClinVar data preprocessing for genome sequence evaluation')
    parser.add_argument('--output_dir', type=str, default='./clinvar_data',
                       help='Output directory for processed data')
    parser.add_argument('--download', action='store_true',
                       help='Download ClinVar data from NCBI')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples per class')
    parser.add_argument('--sequence_length', type=int, default=100,
                       help='Length of reference sequences to generate')
    parser.add_argument('--input_file', type=str,
                       help='Input ClinVar file (if not downloading)')
    
    args = parser.parse_args()
    
    # ログディレクトリを作成
    os.makedirs('logs', exist_ok=True)
    
    processor = ClinVarProcessor(args.output_dir)
    
    try:
        # データの取得
        if args.download:
            logger.info("Downloading ClinVar data")
            gzip_file = processor.download_clinvar_data('variant_summary')
            df = processor.extract_and_load_data(gzip_file)
        elif args.input_file:
            logger.info(f"Loading data from {args.input_file}")
            if args.input_file.endswith('.gz'):
                df = processor.extract_and_load_data(args.input_file)
            else:
                df = pd.read_csv(args.input_file, sep='\t', low_memory=False)
        else:
            raise ValueError("Either --download or --input_file must be specified")
        
        logger.info(f"Initial dataset size: {len(df)}")
        
        # データの前処理
        logger.info("Starting data preprocessing")
        
        # 単一ヌクレオチド変異のフィルタリング
        df_snv = processor.filter_single_nucleotide_variants(df)
        
        # 臨床的意義でフィルタリング
        df_filtered = processor.filter_by_clinical_significance(df_snv)
        
        # 参照配列の取得（モック実装）
        df_with_sequences = processor.get_reference_sequences(
            df_filtered, 
            sequence_length=args.sequence_length
        )
        
        # 評価用データセットの準備
        df_eval = processor.prepare_evaluation_dataset(
            df_with_sequences,
            max_samples_per_class=args.max_samples
        )
        
        # データセットの保存
        processor.save_dataset(df_eval, 'clinvar_evaluation_dataset.csv')
        
        # 統計レポートの作成
        processor.create_statistics_report(df_eval)
        
        logger.info("Data preprocessing completed successfully")
        logger.info(f"Final dataset saved with {len(df_eval)} variants")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
