#!/usr/bin/env python3
"""
プロジェクト全体で使用するパス設定の定数定義
"""

import os

# データセット保存先ディレクトリの定数定義
LEARNING_SOURCE_DIR = 'learning_source_202508'

# プロジェクトルートディレクトリの取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_refseq_tokenizer_path():
    return os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR, 'refseq', 'spm_tokenizer.model')

# 各データセットの基本パス
def get_dataset_path(dataset_type, relative_path=""):
    """
    データセットのパスを取得する関数
    
    Args:
        dataset_type (str): データセットタイプ ('uniprot', 'refseq', 'cellxgene')
        relative_path (str): データセット内の相対パス
        
    Returns:
        str: 完全なパス
    """
    base_path = os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR, dataset_type)
    if relative_path:
        return os.path.join(base_path, relative_path)
    return base_path

# よく使用されるパスの定数
PROTEIN_SEQUENCE_DIR = LEARNING_SOURCE_DIR + '/protein_sequence'
GENOME_SEQUENCE_DIR = LEARNING_SOURCE_DIR + '/genome_sequence'
RNA_DATASET_DIR = LEARNING_SOURCE_DIR + '/rna'
MOLECULE_NL_DATASET_DIR = LEARNING_SOURCE_DIR + '/molecule_nl'
COMPOUNDS_DIR = LEARNING_SOURCE_DIR + '/compounds'
UNIPROT_DATASET_DIR = get_dataset_path('protein_sequence', 'training_ready_hf_dataset')
REFSEQ_DATASET_DIR = get_dataset_path('genome_sequence/refseq', 'training_ready_hf_dataset')
CELLXGENE_DATASET_DIR = get_dataset_path('rna/cellxgene', 'training_ready_hf_dataset')
COMPOUNDS_DATASET_DIR = get_dataset_path('compounds', 'training_ready_hf_dataset')

REFSEQ_TOKENIZER_PATH = get_dataset_path('refseq', 'spm_tokenizer.model')

# 絶対パス版（WebアプリケーションやAPIで使用）
ABSOLUTE_LEARNING_SOURCE_PATH = os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR)

# GPT-2モデル出力先ディレクトリの基本パス
GPT2_OUTPUT_BASE_DIR = 'gpt2-output'

def get_gpt2_output_path(domain, model_size):
    """
    GPT-2モデルの出力パスを取得する関数
    
    Args:
        domain (str): ドメイン名 ('protein_sequence', 'genome_sequence', 'rna', 'compounds', 'molecule_nl')
        model_size (str): モデルサイズ ('small', 'medium', 'large', 'xl', 'ex-large')
        
    Returns:
        str: GPT-2出力ディレクトリのパス
    """
    # model_sizeの標準化
    if model_size == 'xl':
        size_suffix = 'ex-large'
    else:
        size_suffix = model_size
    
    return os.path.join(GPT2_OUTPUT_BASE_DIR, f"{domain}-{size_suffix}")

# よく使用されるGPT-2出力パスの定数
def get_gpt2_tensorboard_path(domain, model_size):
    """GPT-2 TensorBoard出力パスを取得"""
    return get_gpt2_output_path(domain, model_size)

def get_gpt2_model_output_path(domain, model_size):
    """GPT-2モデル出力パスを取得"""
    return get_gpt2_output_path(domain, model_size)
