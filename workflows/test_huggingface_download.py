#!/usr/bin/env python3
"""
Hugging Face Hub からモデルをダウンロードしてテストするスクリプト

このスクリプトは以下の機能を提供します：
1. Hugging Face Hub からモデルをダウンロード
2. チェックポイントの読み込み確認
3. モデル構造の検証
4. 簡単な推論テスト
5. テキスト生成テスト（オプション）

使用方法:
    python test_huggingface_download.py <repo_id> [options]

例:
    python test_huggingface_download.py deskull/rna-small-gpt2 --test-generate
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# プロジェクトのパスを追加
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


try:
    import torch
except ImportError:
    print("[ERROR] PyTorchがインストールされていません")
    print("インストール: pip install torch")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
except ImportError:
    print("[ERROR] huggingface_hub がインストールされていません")
    print("インストール: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Hugging Face Hub からモデルをダウンロードしてテスト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的なテスト（ダウンロードと読み込み確認）
  python test_huggingface_download.py deskull/rna-small-gpt2

  # 生成テスト付き
  python test_huggingface_download.py deskull/rna-small-gpt2 --test-generate

  # ローカルキャッシュを指定
  python test_huggingface_download.py deskull/rna-small-gpt2 --cache-dir ./models
        """,
    )

    parser.add_argument("repo_id", type=str, help="Hugging Face Hub のリポジトリID")
    parser.add_argument("--revision", type=str, default="main", help="ブランチ/タグ/コミット")
    parser.add_argument("--cache-dir", type=str, help="ダウンロード先のキャッシュディレクトリ")
    parser.add_argument("--checkpoint-file", type=str, default="ckpt.pt", help="チェックポイントファイル名")
    parser.add_argument("--device", type=str, default="auto", help="使用するデバイス (cpu, cuda, auto)")
    parser.add_argument("--test-generate", action="store_true", help="テキスト生成テストを実行")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["rna", "genome", "protein_sequence", "compounds", "molecule_nl"],
        help="モデルのドメイン（トークナイザー選択用）",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="生成する最大トークン数")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細な出力")

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """デバイスを決定"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def list_repo_contents(repo_id: str, revision: str = "main") -> list:
    """リポジトリの内容を一覧"""
    print(f"\n[INFO] リポジトリの内容を確認中: {repo_id}")
    try:
        files = list_repo_files(repo_id, revision=revision)
        print(f"[INFO] ファイル一覧 ({len(files)} 件):")
        for f in files:
            print(f"  - {f}")
        return files
    except Exception as e:
        print(f"[ERROR] リポジトリ内容の取得に失敗: {e}")
        return []


def download_checkpoint(repo_id: str, checkpoint_file: str, cache_dir: str = None, revision: str = "main") -> str:
    """チェックポイントファイルをダウンロード"""
    print(f"\n[INFO] チェックポイントをダウンロード中: {checkpoint_file}")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_file,
            cache_dir=cache_dir,
            revision=revision,
        )
        print(f"[SUCCESS] ダウンロード完了: {local_path}")
        return local_path
    except Exception as e:
        print(f"[ERROR] ダウンロードに失敗: {e}")
        return None


def download_all_files(repo_id: str, cache_dir: str = None, revision: str = "main") -> str:
    """リポジトリ全体をダウンロード"""
    print("\n[INFO] リポジトリ全体をダウンロード中...")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
        )
        print(f"[SUCCESS] ダウンロード完了: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"[ERROR] ダウンロードに失敗: {e}")
        return None


def load_checkpoint(checkpoint_path: str, device: str, verbose: bool = False):
    """チェックポイントを読み込み"""
    print("\n[INFO] チェックポイントを読み込み中...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # チェックポイントの内容を確認
        print("[INFO] チェックポイントのキー:")
        for key in checkpoint.keys():
            if key == "model":
                print(f"  - {key}: (state_dict with {len(checkpoint[key])} parameters)")
            elif isinstance(checkpoint[key], dict):
                print(f"  - {key}: (dict with {len(checkpoint[key])} items)")
            else:
                print(f"  - {key}: {checkpoint[key]}")

        # モデル設定を取得
        model_args = checkpoint.get("model_args", {})
        if model_args:
            print("\n[INFO] モデル設定:")
            for k, v in model_args.items():
                print(f"  - {k}: {v}")

        return checkpoint

    except Exception as e:
        print(f"[ERROR] チェックポイントの読み込みに失敗: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_model(checkpoint: dict, device: str):
    """チェックポイントからモデルを構築"""
    print("\n[INFO] モデルを構築中...")

    try:
        # GPTモデルをインポート
        from gpt2.model import GPT, GPTConfig

        model_args = checkpoint.get("model_args", {})
        if not model_args:
            print("[ERROR] model_args がチェックポイントに含まれていません")
            return None

        # GPTConfigを作成
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # 状態辞書をロード
        state_dict = checkpoint["model"]

        # 不要なプレフィックスを削除
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # モデル情報を表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[SUCCESS] モデル構築完了")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - 訓練可能パラメータ数: {trainable_params:,}")
        print(f"  - メモリ使用量: {total_params * 4 / (1024**2):.2f} MB (float32)")

        return model

    except Exception as e:
        print(f"[ERROR] モデルの構築に失敗: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_tokenizer(domain: str):
    """ドメイン用のトークナイザーをロード"""
    print(f"\n[INFO] トークナイザーをロード中 (domain: {domain})...")

    try:
        if domain == "rna":
            from rna.utils.bert_tokenizer import create_bert_rna_tokenizer

            tokenizer = create_bert_rna_tokenizer()
        elif domain == "genome":
            from genome_sequence.utils.tokenizer import create_genome_tokenizer

            tokenizer = create_genome_tokenizer()
        elif domain == "protein_sequence":
            from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

            tokenizer = create_bert_protein_tokenizer()
        elif domain == "compounds":
            from compounds.utils.tokenizer import CompoundsTokenizer

            vocab_file = str(PROJECT_ROOT / "assets" / "molecules" / "vocab.txt")
            tokenizer = CompoundsTokenizer(vocab_file, 256)
        elif domain == "molecule_nl":
            from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer

            tokenizer = MoleculeNatLangTokenizer()
        else:
            print(f"[WARNING] 未知のドメイン: {domain}")
            return None

        print("[SUCCESS] トークナイザーのロード完了")
        return tokenizer

    except Exception as e:
        print(f"[WARNING] トークナイザーのロードに失敗: {e}")
        return None


def test_forward_pass(model, device: str, vocab_size: int = 1000, seq_len: int = 128):
    """フォワードパスのテスト"""
    print("\n[INFO] フォワードパスをテスト中...")

    try:
        # ランダムな入力テンソルを作成
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        with torch.no_grad():
            # フォワードパス
            logits, loss = model(input_ids)

        print("[SUCCESS] フォワードパス成功")
        print(f"  - 入力形状: {input_ids.shape}")
        print(f"  - 出力形状: {logits.shape}")
        print(f"  - 出力の統計: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

        return True

    except Exception as e:
        print(f"[ERROR] フォワードパスに失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generate(model, tokenizer, device: str, max_tokens: int = 50):
    """テキスト生成テスト"""
    print("\n[INFO] テキスト生成をテスト中...")

    try:
        # 初期トークンを用意
        if tokenizer is not None:
            # トークナイザーがある場合、適切な開始トークンを使用
            if hasattr(tokenizer, "encode"):
                # サンプル入力（ドメインに応じて変更）
                sample_inputs = ["A", "G", "C", "U"]  # RNAの場合
                try:
                    input_ids = tokenizer.encode(sample_inputs[0])
                    if isinstance(input_ids, list):
                        input_ids = torch.tensor([input_ids], device=device)
                    else:
                        input_ids = input_ids.unsqueeze(0).to(device)
                except Exception:
                    input_ids = torch.tensor([[1]], device=device)  # フォールバック
            else:
                input_ids = torch.tensor([[1]], device=device)
        else:
            # トークナイザーがない場合、ランダムなトークンで開始
            input_ids = torch.randint(1, 100, (1, 1), device=device)

        print(f"  - 開始トークン: {input_ids.tolist()}")

        # 生成
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=40,
            )

        print("[SUCCESS] テキスト生成成功")
        print(f"  - 生成トークン数: {generated.shape[1]}")
        print(f"  - 生成されたトークンID: {generated[0, :20].tolist()}...")  # 最初の20トークン

        # デコード（トークナイザーがある場合）
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                decoded = tokenizer.decode(generated[0].tolist())
                print(f"  - デコード結果: {decoded[:100]}...")  # 最初の100文字
            except Exception as e:
                print(f"  - デコードエラー: {e}")

        return True

    except Exception as e:
        print(f"[ERROR] テキスト生成に失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_tests(
    repo_id: str,
    revision: str = "main",
    cache_dir: str = None,
    checkpoint_file: str = "ckpt.pt",
    device: str = "auto",
    test_generate: bool = False,
    domain: str = None,
    max_tokens: int = 50,
    verbose: bool = False,
) -> dict:
    """すべてのテストを実行"""

    results = {
        "repo_id": repo_id,
        "success": True,
        "tests": {},
    }

    device = get_device(device)
    print(f"\n{'='*60}")
    print("Hugging Face Hub モデルテスト")
    print(f"{'='*60}")
    print(f"リポジトリID: {repo_id}")
    print(f"デバイス: {device}")
    print(f"{'='*60}")

    # 1. リポジトリ内容を確認
    files = list_repo_contents(repo_id, revision)
    results["tests"]["list_files"] = len(files) > 0
    if not files:
        results["success"] = False
        return results

    # チェックポイントファイルを探す
    pt_files = [f for f in files if f.endswith(".pt")]
    if not pt_files:
        print("[ERROR] .pt ファイルが見つかりません")
        results["success"] = False
        return results

    # 指定されたファイル、またはckpt.pt、または最初の.ptファイル
    if checkpoint_file in files:
        target_file = checkpoint_file
    elif "ckpt.pt" in files:
        target_file = "ckpt.pt"
    else:
        target_file = pt_files[0]
    print(f"\n[INFO] 使用するチェックポイント: {target_file}")

    # 2. チェックポイントをダウンロード
    checkpoint_path = download_checkpoint(repo_id, target_file, cache_dir, revision)
    results["tests"]["download"] = checkpoint_path is not None
    if not checkpoint_path:
        results["success"] = False
        return results

    # 3. チェックポイントを読み込み
    checkpoint = load_checkpoint(checkpoint_path, device, verbose)
    results["tests"]["load_checkpoint"] = checkpoint is not None
    if not checkpoint:
        results["success"] = False
        return results

    # 4. モデルを構築
    model = load_model(checkpoint, device)
    results["tests"]["load_model"] = model is not None
    if not model:
        results["success"] = False
        return results

    # モデル設定からvocab_sizeを取得
    model_args = checkpoint.get("model_args", {})
    vocab_size = model_args.get("vocab_size", 1000)
    block_size = model_args.get("block_size", 128)

    # 5. フォワードパステスト
    forward_ok = test_forward_pass(model, device, vocab_size, min(block_size, 128))
    results["tests"]["forward_pass"] = forward_ok
    if not forward_ok:
        results["success"] = False

    # 6. 生成テスト（オプション）
    if test_generate:
        tokenizer = None
        if domain:
            tokenizer = load_tokenizer(domain)

        generate_ok = test_generate_func(model, tokenizer, device, max_tokens)
        results["tests"]["generate"] = generate_ok
        if not generate_ok:
            results["success"] = False

    return results


# test_generateと関数名が被らないようにリネーム
test_generate_func = test_generate


def print_summary(results: dict):
    """テスト結果のサマリーを表示"""
    print(f"\n{'='*60}")
    print("テスト結果サマリー")
    print(f"{'='*60}")

    for test_name, passed in results["tests"].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"{'='*60}")
    if results["success"]:
        print("[SUCCESS] すべてのテストに合格しました！")
    else:
        print("[FAILED] 一部のテストに失敗しました")
    print(f"{'='*60}")


def main():
    """メイン関数"""
    args = parse_args()

    results = run_tests(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        test_generate=args.test_generate,
        domain=args.domain,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    print_summary(results)

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
