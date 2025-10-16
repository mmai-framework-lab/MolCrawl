#!/usr/bin/env python3
"""
AI モデル評価スクリプト用の出力ディレクトリ管理ユーティリティ

LEARNING_SOURCE_DIR環境変数に基づいて、構造化された評価レポートディレクトリを生成する。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_learning_source_dir():
    """
    LEARNING_SOURCE_DIR環境変数を取得し、デフォルト値を設定
    
    Returns:
        Path: LEARNING_SOURCE_DIRのパス
    """
    learning_source_dir = os.getenv('LEARNING_SOURCE_DIR')
    
    if not learning_source_dir:
        # プロジェクトルートから推定
        project_root = Path(__file__).parent.parent.parent
        learning_source_dir = project_root / "learning_source_202508"
        logger.warning(f"LEARNING_SOURCE_DIR not set, using default: {learning_source_dir}")
    
    return Path(learning_source_dir)

def get_evaluation_output_dir(model_type, evaluation_type, model_name=None, timestamp=None):
    """
    評価レポート用の出力ディレクトリパスを生成
    
    Args:
        model_type (str): モデルタイプ ('genome_sequence', 'protein_sequence', etc.)
        evaluation_type (str): 評価タイプ ('proteingym', 'clinvar', 'protein_classification', etc.)
        model_name (str, optional): モデル名（指定しない場合は自動生成）
        timestamp (str, optional): タイムスタンプ（指定しない場合は現在時刻）
    
    Returns:
        Path: 評価結果出力ディレクトリのパス
        
    Example:
        get_evaluation_output_dir('genome_sequence', 'clinvar')
        -> {LEARNING_SOURCE_DIR}/genome_sequence/report/clinvar_20241015_143022
        
        get_evaluation_output_dir('protein_sequence', 'proteingym', 'bert_medium')
        -> {LEARNING_SOURCE_DIR}/protein_sequence/report/proteingym_bert_medium_20241015_143022
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    learning_source_dir = get_learning_source_dir()
    
    # モデルタイプディレクトリ（genome_sequence, protein_sequence等）
    model_type_dir = learning_source_dir / model_type
    
    # reportディレクトリ
    report_dir = model_type_dir / "report"
    
    # 評価タイプとモデル名を含むディレクトリ名を生成
    if model_name:
        dir_name = f"{evaluation_type}_{model_name}_{timestamp}"
    else:
        dir_name = f"{evaluation_type}_{timestamp}"
    
    output_dir = report_dir / dir_name
    
    # ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created evaluation output directory: {output_dir}")
    
    return output_dir

def get_model_type_from_path(model_path):
    """
    モデルパスからモデルタイプを推定
    
    Args:
        model_path (str): モデルのパス
        
    Returns:
        str: 推定されたモデルタイプ
    """
    model_path_str = str(model_path).lower()
    
    if 'genome' in model_path_str:
        return 'genome_sequence'
    elif 'protein' in model_path_str:
        return 'protein_sequence'
    elif 'compound' in model_path_str:
        return 'compounds'
    elif 'rna' in model_path_str:
        return 'rna'
    elif 'molecule' in model_path_str:
        return 'molecule_nl'
    else:
        return 'general'

def get_model_name_from_path(model_path):
    """
    モデルパスからモデル名を推定
    
    Args:
        model_path (str): モデルのパス
        
    Returns:
        str: 推定されたモデル名
    """
    model_path = Path(model_path)
    
    # パスの最後の部分からモデル名を推定
    if model_path.is_dir():
        model_name = model_path.name
    else:
        model_name = model_path.stem
    
    # 共通のプレフィックス/サフィックスを除去
    model_name = model_name.replace('runs_train_', '')
    model_name = model_name.replace('bert_', '')
    model_name = model_name.replace('gpt2_', '')
    
    return model_name

def setup_evaluation_logging(output_dir, script_name):
    """
    評価スクリプト用のログ設定
    
    Args:
        output_dir (Path): 出力ディレクトリ
        script_name (str): スクリプト名
        
    Returns:
        logging.Logger: 設定されたロガー
    """
    log_file = output_dir / f"{script_name}.log"
    
    # ログフォーマット
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # ロガー設定
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_evaluation_summary(output_dir, evaluation_info):
    """
    評価サマリーファイルを作成
    
    Args:
        output_dir (Path): 出力ディレクトリ
        evaluation_info (dict): 評価情報
    """
    summary_file = output_dir / "evaluation_summary.json"
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(evaluation_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation summary saved to: {summary_file}")

if __name__ == "__main__":
    # テスト実行
    print("Testing evaluation output directory generation...")
    
    # テスト1: genome_sequence + clinvar
    output_dir1 = get_evaluation_output_dir('genome_sequence', 'clinvar')
    print(f"Test 1: {output_dir1}")
    
    # テスト2: protein_sequence + proteingym + model name
    output_dir2 = get_evaluation_output_dir('protein_sequence', 'proteingym', 'bert_medium')
    print(f"Test 2: {output_dir2}")
    
    # テスト3: モデルタイプ推定
    model_type = get_model_type_from_path("runs_train_bert_genome_sequence")
    print(f"Test 3: {model_type}")
    
    # テスト4: モデル名推定
    model_name = get_model_name_from_path("runs_train_bert_protein_sequence_medium")
    print(f"Test 4: {model_name}")