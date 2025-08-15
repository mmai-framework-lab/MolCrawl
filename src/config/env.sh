#!/bin/bash
# 環境変数設定用スクリプト
# 使用方法: source src/config/env.sh

export LEARNING_SOURCE_DIR="learning_source_202508"
export UNIPROT_DATASET_DIR="$LEARNING_SOURCE_DIR/uniprot/training_ready_hf_dataset"
export REFSEQ_DATASET_DIR="$LEARNING_SOURCE_DIR/refseq/training_ready_hf_dataset"
export CELLXGENE_DATASET_DIR="$LEARNING_SOURCE_DIR/cellxgene/training_ready_hf_dataset"
export COMPOUNDS_DATASET_DIR="$LEARNING_SOURCE_DIR/compounds/training_ready_hf_dataset"
export MOLECULE_NL_DATASET_DIR="$LEARNING_SOURCE_DIR/molecule_nl/training_ready_hf_dataset"
export REFSEQ_TOKENIZER_PATH="$LEARNING_SOURCE_DIR/refseq/spm_tokenizer.model"

echo "環境変数が設定されました:"
echo "  LEARNING_SOURCE_DIR=$LEARNING_SOURCE_DIR"
echo "  UNIPROT_DATASET_DIR=$UNIPROT_DATASET_DIR"
echo "  REFSEQ_DATASET_DIR=$REFSEQ_DATASET_DIR"
echo "  CELLXGENE_DATASET_DIR=$CELLXGENE_DATASET_DIR"
echo "  COMPOUNDS_DATASET_DIR=$COMPOUNDS_DATASET_DIR"
echo "  MOLECULE_NL_DATASET_DIR=$MOLECULE_NL_DATASET_DIR"
echo "  REFSEQ_TOKENIZER_PATH=$REFSEQ_TOKENIZER_PATH"
