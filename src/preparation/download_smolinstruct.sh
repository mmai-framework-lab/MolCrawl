#!/bin/bash
#
# SMolInstruct Dataset Download Script
# 
# このスクリプトはHugging Face CLIを使用してSMolInstructデータセットを
# ローカルにダウンロードします。
#
# 使用方法:
#   bash scripts/preparation/download_smolinstruct.sh [output_directory]
#
# 必要条件:
#   - huggingface_hub Pythonパッケージ
#   - インターネット接続
#   - (オプション) Hugging Face認証トークン
#

set -e  # エラーで停止

# デフォルトの出力ディレクトリ
DEFAULT_OUTPUT_DIR="molecule_nl/osunlp/SMolInstruct"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

# 色付きログ
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# LEARNING_SOURCE_DIRの確認
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    log_error "LEARNING_SOURCE_DIR environment variable is not set!"
    log_error "Please set it before running this script:"
    log_error "  export LEARNING_SOURCE_DIR='learning_source_202508'"
    exit 1
fi

log_info "Using LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"

# 完全な出力パスを構築
FULL_OUTPUT_DIR="${LEARNING_SOURCE_DIR}/${OUTPUT_DIR}"
log_info "Download destination: $FULL_OUTPUT_DIR"

# ディレクトリが既に存在するかチェック
if [ -d "$FULL_OUTPUT_DIR" ] && [ "$(ls -A $FULL_OUTPUT_DIR)" ]; then
    log_warning "Directory $FULL_OUTPUT_DIR already exists and is not empty."
    read -p "Do you want to remove it and re-download? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing directory..."
        rm -rf "$FULL_OUTPUT_DIR"
    else
        log_info "Keeping existing data. Exiting."
        exit 0
    fi
fi

# 出力ディレクトリを作成
mkdir -p "$FULL_OUTPUT_DIR"

# Pythonとhuggingface_hubの確認
log_info "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    log_error "Python3 is not installed or not in PATH"
    exit 1
fi

# huggingface_hubパッケージの確認
log_info "Checking huggingface_hub package..."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    log_warning "huggingface_hub package not found. Installing..."
    pip install huggingface_hub
fi

# Hugging Face認証の確認（オプション）
log_info "Checking Hugging Face authentication..."
if python3 -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
    log_success "Hugging Face authentication found"
else
    log_warning "No Hugging Face authentication found"
    log_info "Some datasets may require authentication. If download fails, run:"
    log_info "  huggingface-cli login"
fi

# Pythonスクリプトでダウンロード
log_info "Starting dataset download..."
log_info "This may take a while (dataset is several GB)..."

python3 << 'PYTHON_SCRIPT'
import os
import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 環境変数から出力ディレクトリを取得
learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
output_dir = os.environ.get("OUTPUT_DIR", "molecule_nl/osunlp/SMolInstruct")
full_output_dir = Path(learning_source_dir) / output_dir

logger.info(f"Downloading to: {full_output_dir}")

try:
    # データセットをダウンロード
    snapshot_download(
        repo_id="osunlp/SMolInstruct",
        repo_type="dataset",
        local_dir=str(full_output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=True
    )
    
    logger.info("Dataset download completed successfully!")
    logger.info(f"Dataset saved to: {full_output_dir}")
    
    # ダウンロードされたファイルを確認
    files = list(full_output_dir.rglob("*"))
    logger.info(f"Downloaded {len(files)} files/directories")
    
    # data.zipファイルを展開
    data_zip = full_output_dir / "data.zip"
    if data_zip.exists():
        logger.info(f"Found data.zip, extracting...")
        import zipfile
        
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall(full_output_dir)
        
        logger.info(f"Successfully extracted data.zip")
        
        # 展開後のファイルを確認
        extracted_files = list(full_output_dir.rglob("*.parquet"))
        logger.info(f"Found {len(extracted_files)} parquet files after extraction")
    else:
        logger.warning("No data.zip file found, skipping extraction")
    
except Exception as e:
    logger.error(f"Failed to download dataset: {e}")
    logger.error("\nTroubleshooting:")
    logger.error("1. Check your internet connection")
    logger.error("2. Try running: huggingface-cli login")
    logger.error("3. Check if the dataset is accessible: https://huggingface.co/datasets/osunlp/SMolInstruct")
    sys.exit(1)

PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    log_success "Dataset downloaded successfully!"
    log_info "Location: $FULL_OUTPUT_DIR"
    log_info ""
    log_info "Next steps:"
    log_info "1. Run the preparation script:"
    log_info "   python scripts/preparation/preparation_script_molecule_related_nat_lang.py \\"
    log_info "     assets/configs/molecule_nl.yaml"
else
    log_error "Dataset download failed!"
    exit 1
fi
