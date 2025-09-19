#!/usr/bin/env python3
"""
OMIM Data Preparation Script
============================

OMIM (Online Mendelian Inheritance in Man) データベースの遺伝性疾患情報を
ゲノム配列モデルの評価用に前処理するスクリプト

主な機能:
- OMIMサンプルデータの生成
- 実際のOMIMデータの処理（認証済みアクセス）
- 遺伝性疾患関連変異とベニン変異の分類
- モデル評価用データセットの作成
- データ品質チェック

注意: 実際のOMIMデータ使用にはライセンスが必要です
"""

import os
import sys
import random
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# プロジェクトルートを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 実データプロセッサをインポート
try:
    from omim_real_data_processor import process_omim_real_data
except ImportError:
    process_omim_real_data = None

def setup_logging(output_dir: str) -> logging.Logger:
    """ログ設定をセットアップ"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"omim_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class OMIMDataGenerator:
    """OMIM評価用サンプルデータ生成クラス"""
    
    def __init__(self, sequence_length: int = 100, logger: Optional[logging.Logger] = None):
        self.sequence_length = sequence_length
        self.logger = logger or logging.getLogger(__name__)
        
        # OMIM関連遺伝子の例（実際のOMIMから取得した代表的な遺伝子）
        self.disease_genes = [
            'BRCA1', 'BRCA2', 'TP53', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
            'PALB2', 'CHEK2', 'ATM', 'BARD1', 'BRIP1', 'RAD51C', 'RAD51D',
            'CDH1', 'PTEN', 'STK11', 'NF1', 'NF2', 'VHL', 'RB1', 'WT1',
            'CDKN2A', 'CDK4', 'BAP1', 'MITF', 'POT1', 'ACD', 'TERF2IP',
            'TERT', 'DCC', 'SMAD4', 'BMPR1A', 'MUTYH', 'NTHL1', 'POLE',
            'POLD1', 'EPCAM', 'PMS1', 'AXIN2', 'GREM1', 'SCG5', 'RNF43'
        ]
        
        # OMIM表現型分類
        self.phenotype_categories = {
            'autosomal_dominant': 0.3,
            'autosomal_recessive': 0.25,
            'x_linked': 0.15,
            'mitochondrial': 0.05,
            'complex': 0.25
        }
        
        # 病原性レベル
        self.pathogenicity_levels = {
            'pathogenic': 1,
            'likely_pathogenic': 1,
            'uncertain_significance': 0,
            'likely_benign': 0,
            'benign': 0
        }

    def generate_sequence(self, is_pathogenic: bool = True) -> str:
        """ゲノム配列を生成"""
        nucleotides = ['A', 'T', 'G', 'C']
        
        if is_pathogenic:
            # 病原性変異: より多くの変異を含む
            sequence = ''.join(random.choices(nucleotides, k=self.sequence_length))
            # 特定の位置に病原性変異パターンを挿入
            mutation_positions = random.sample(range(self.sequence_length), 
                                             min(5, self.sequence_length // 20))
            sequence_list = list(sequence)
            for pos in mutation_positions:
                # フレームシフトや停止コドンを模倣
                sequence_list[pos] = random.choice(['T', 'A'])  # より病原性の高い変異
            sequence = ''.join(sequence_list)
        else:
            # ベニン変異: より保守的な配列
            sequence = ''.join(random.choices(nucleotides, 
                                            weights=[0.3, 0.3, 0.2, 0.2], 
                                            k=self.sequence_length))
        
        return sequence

    def generate_omim_entry(self, entry_id: int) -> Dict:
        """単一のOMIMエントリを生成"""
        is_pathogenic = random.random() < 0.7  # 70%が病原性
        
        gene = random.choice(self.disease_genes)
        phenotype_type = random.choices(
            list(self.phenotype_categories.keys()),
            weights=list(self.phenotype_categories.values())
        )[0]
        
        if is_pathogenic:
            pathogenicity = random.choices(
                ['pathogenic', 'likely_pathogenic'],
                weights=[0.7, 0.3]
            )[0]
        else:
            pathogenicity = random.choices(
                ['benign', 'likely_benign', 'uncertain_significance'],
                weights=[0.5, 0.3, 0.2]
            )[0]
        
        # OMIM ID形式 (6桁の数字)
        omim_id = f"{entry_id + 100000:06d}"
        
        return {
            'omim_id': omim_id,
            'gene_symbol': gene,
            'sequence': self.generate_sequence(is_pathogenic),
            'phenotype_type': phenotype_type,
            'pathogenicity': pathogenicity,
            'is_disease_causing': self.pathogenicity_levels[pathogenicity],
            'chromosome': random.choice([str(i) for i in range(1, 23)] + ['X', 'Y']),
            'position': random.randint(1000000, 200000000),
            'inheritance_pattern': phenotype_type,
            'clinical_significance': pathogenicity,
            'mim_number': omim_id,
            'disease_name': f"Hereditary {gene.lower()} disorder",
            'molecular_basis': 'Point mutation' if random.random() < 0.6 else 'Deletion/Duplication'
        }

    def generate_dataset(self, num_samples: int) -> pd.DataFrame:
        """OMIMデータセットを生成"""
        self.logger.info(f"Creating sample OMIM data with {num_samples} samples")
        
        data = []
        for i in range(num_samples):
            entry = self.generate_omim_entry(i)
            data.append(entry)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{num_samples} OMIM entries")
        
        df = pd.DataFrame(data)
        
        # データバランスの調整
        pathogenic_count = df['is_disease_causing'].sum()
        benign_count = len(df) - pathogenic_count
        
        self.logger.info(f"Generated dataset statistics:")
        self.logger.info(f"  Disease-causing variants: {pathogenic_count}")
        self.logger.info(f"  Benign variants: {benign_count}")
        self.logger.info(f"  Total samples: {len(df)}")
        
        return df

def prepare_omim_data(
    output_dir: str,
    num_samples: int = 1000,
    sequence_length: int = 100,
    seed: int = 42
) -> str:
    """OMIM評価データを準備"""
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # ログ設定
    logger = setup_logging(output_dir)
    logger.info("Starting OMIM data preparation")
    
    # シード設定
    random.seed(seed)
    np.random.seed(seed)
    
    # データ生成
    generator = OMIMDataGenerator(sequence_length=sequence_length, logger=logger)
    df = generator.generate_dataset(num_samples)
    
    # データ保存
    output_file = os.path.join(data_dir, 'omim_evaluation_dataset.csv')
    df.to_csv(output_file, index=False)
    
    logger.info(f"Sample OMIM data saved to {output_file}")
    logger.info(f"Data distribution: {df['is_disease_causing'].value_counts().to_dict()}")
    
    # メタデータ保存
    metadata = {
        'total_samples': len(df),
        'disease_causing': int(df['is_disease_causing'].sum()),
        'benign': int((df['is_disease_causing'] == 0).sum()),
        'sequence_length': sequence_length,
        'generation_date': datetime.now().isoformat(),
        'inheritance_patterns': df['inheritance_pattern'].value_counts().to_dict(),
        'pathogenicity_distribution': df['pathogenicity'].value_counts().to_dict()
    }
    
    metadata_file = os.path.join(data_dir, 'omim_metadata.json')
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("OMIM data preparation completed")
    return output_file

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='OMIM Data Preparation for Genome Sequence Evaluation')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for prepared data')
    parser.add_argument('--mode', type=str, choices=['sample', 'real'], default='sample',
                       help='Data mode: sample (generated data) or real (actual OMIM data)')
    parser.add_argument('--config', type=str,
                       help='Path to real data config file (required for real mode)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate for sample mode (default: 1000)')
    parser.add_argument('--sequence_length', type=int, default=100,
                       help='Length of genome sequences (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force_download', action='store_true',
                       help='Force download for real mode even if files exist')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'real':
            # 実データモード
            if not args.config:
                raise ValueError("Real data mode requires --config parameter")
            
            if process_omim_real_data is None:
                raise ImportError("Real data processor not available. Check omim_real_data_processor.py")
            
            print("Processing real OMIM data...")
            output_file = process_omim_real_data(
                config_path=args.config,
                output_dir=args.output_dir,
                force_download=args.force_download
            )
            print(f"Real OMIM data processing completed!")
            
        else:
            # サンプルデータモード
            print("Generating sample OMIM data...")
            output_file = prepare_omim_data(
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                sequence_length=args.sequence_length,
                seed=args.seed
            )
            print(f"Sample OMIM data preparation completed!")
        
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during OMIM data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
