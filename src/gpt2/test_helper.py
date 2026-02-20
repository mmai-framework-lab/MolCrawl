"""
GPT2チェックポイントテスト用のヘルパースクリプト
各ドメインのチェックポイントを自動的に検出してテストします。

# チェックポイントをリストアップ
python gpt2/test_helper.py --search_dir=runs_train_bert_*/checkpoints --list_only

# 特定のチェックポイントをテスト
python gpt2/test_helper.py --checkpoint_path=path/to/ckpt.pt

# 全チェックポイントを自動テスト
python gpt2/test_helper.py --search_dir=runs_* --auto_run

"""

import argparse
import glob
import json
import os
from pathlib import Path


def find_checkpoint_files(search_dir):
    """チェックポイントファイルを検索"""
    checkpoint_patterns = [
        "**/ckpt.pt",
        "**/checkpoint.pt",
        "**/pytorch_model.bin",
        "**/model.safetensors",
        "**/*checkpoint*.pt",
    ]

    checkpoints = []
    for pattern in checkpoint_patterns:
        found = glob.glob(os.path.join(search_dir, pattern), recursive=True)
        checkpoints.extend(found)

    return list(set(checkpoints))  # 重複を除去


def get_domain_info():
    """各ドメインの情報を返す"""
    import os
    import sys

    from config.paths import COMPOUNDS_DATASET_DIR, MOLECULE_NL_DATASET_DIR

    return {
        "compounds": {
            "vocab_path": "assets/molecules/vocab.txt",
            "dataset_dir": COMPOUNDS_DATASET_DIR,
        },
        "molecule_nl": {"vocab_path": None, "dataset_dir": MOLECULE_NL_DATASET_DIR},
        "genome": {
            "vocab_path": None,  # SentencePieceモデルパスが必要
            "dataset_dir": "outputs/genome_sequence/training_ready_hf_dataset",
        },
        "protein_sequence": {
            "vocab_path": None,
            "dataset_dir": "outputs/protein_sequence/training_ready_hf_dataset",
        },
        "rna": {
            "vocab_path": None,
            "dataset_dir": "outputs/rna/training_ready_hf_dataset",
        },
    }


def detect_domain_from_path(checkpoint_path):
    """チェックポイントパスからドメインを推測"""
    path_lower = checkpoint_path.lower()

    if "compound" in path_lower:
        return "compounds"
    elif "molecule" in path_lower and "nl" in path_lower:
        return "molecule_nl"
    elif "genome" in path_lower:
        return "genome"
    elif "protein" in path_lower:
        return "protein_sequence"
    elif "rna" in path_lower:
        return "rna"
    else:
        return None


def create_test_command(checkpoint_path, domain=None, output_dir=None, max_samples=500):
    """テストコマンドを生成"""
    domain_info = get_domain_info()

    # ドメインを推測
    if domain is None:
        domain = detect_domain_from_path(checkpoint_path)

    if domain is None:
        print(f"警告: {checkpoint_path} からドメインを推測できませんでした")
        domain = "unknown"

    # 出力ディレクトリを設定
    if output_dir is None:
        checkpoint_name = Path(checkpoint_path).parent.name
        output_dir = f"test_results_{domain}_{checkpoint_name}"

    # 基本コマンド
    cmd = [
        "python",
        "gpt2/test_checkpoint.py",
        f"--checkpoint_path={checkpoint_path}",
        f"--output_dir={output_dir}",
        f"--max_test_samples={max_samples}",
        "--convert_to_hf",
    ]

    # ドメイン特化の設定
    if domain in domain_info:
        cmd.append(f"--domain={domain}")

        vocab_path = domain_info[domain]["vocab_path"]
        if vocab_path and os.path.exists(vocab_path):
            cmd.append(f"--vocab_path={vocab_path}")

        dataset_dir = domain_info[domain]["dataset_dir"]
        if os.path.exists(dataset_dir):
            dataset_params = {"dataset_dir": dataset_dir}
            cmd.append(f"--test_dataset_params={json.dumps(dataset_params)}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="GPT2チェックポイントテスト用ヘルパー")
    parser.add_argument("--search_dir", default=".", help="チェックポイントを検索するディレクトリ")
    parser.add_argument("--checkpoint_path", help="特定のチェックポイントパス")
    parser.add_argument(
        "--domain",
        choices=["compounds", "molecule_nl", "genome", "protein_sequence", "rna"],
        help="強制的に指定するドメイン",
    )
    parser.add_argument("--output_dir", help="出力ディレクトリ")
    parser.add_argument("--max_samples", type=int, default=500, help="テストサンプル数")
    parser.add_argument("--auto_run", action="store_true", help="自動実行する")
    parser.add_argument("--list_only", action="store_true", help="チェックポイントをリストアップのみ")

    args = parser.parse_args()

    if args.checkpoint_path:
        # 特定のチェックポイントをテスト
        checkpoints = [args.checkpoint_path]
    else:
        # チェックポイントを検索
        print(f"チェックポイントを検索中: {args.search_dir}")
        checkpoints = find_checkpoint_files(args.search_dir)

    if not checkpoints:
        print("チェックポイントが見つかりませんでした。")
        return

    print(f"\n発見されたチェックポイント: {len(checkpoints)}")
    for i, cp in enumerate(checkpoints, 1):
        domain = detect_domain_from_path(cp) or "unknown"
        size_mb = os.path.getsize(cp) / 1024 / 1024 if os.path.exists(cp) else 0
        print(f"{i:2d}. {cp} [{domain}] ({size_mb:.1f} MB)")

    if args.list_only:
        return

    # 各チェックポイントのテストコマンドを生成
    for checkpoint in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"チェックポイント: {checkpoint}")

        domain = args.domain or detect_domain_from_path(checkpoint)
        if domain:
            print(f"検出ドメイン: {domain}")

        cmd = create_test_command(
            checkpoint,
            domain=args.domain,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )

        print("実行コマンド:")
        print(" ".join(cmd))

        if args.auto_run:
            print("\n実行中...")
            import subprocess

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ テスト成功")
                else:
                    print(f"✗ テスト失敗: {result.stderr}")
            except Exception as e:
                print(f"✗ 実行エラー: {e}")
        else:
            print("\n上記コマンドを実行してテストを開始してください。")


def create_test_configs():
    """各ドメイン用のテスト設定ファイルを作成"""
    configs_dir = Path("gpt2/test_configs")
    configs_dir.mkdir(exist_ok=True)

    domain_info = get_domain_info()

    for domain, info in domain_info.items():
        config_content = f"""# {domain.upper()}ドメイン用GPT2テスト設定

# 基本設定
domain = "{domain}"
max_test_samples = 1000
convert_to_hf = True

# データセット設定
"""

        if info["dataset_dir"]:
            config_content += f"""dataset_params = {{
    "dataset_dir": "{info["dataset_dir"]}"
}}
"""

        if info["vocab_path"]:
            config_content += f"""
# 語彙ファイル
vocab_path = "{info["vocab_path"]}"
"""

        config_content += f"""
# 出力設定
output_dir = "test_results_{domain}"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

        config_file = configs_dir / f"{domain}_test_config.py"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)

        print(f"設定ファイルを作成: {config_file}")


if __name__ == "__main__":
    # まず設定ファイルを作成
    print("テスト設定ファイルを作成中...")
    create_test_configs()
    print()

    # メイン処理
    main()
