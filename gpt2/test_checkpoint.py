"""
GPT2チェックポイントの包括的テスト・検証スクリプト

このスクリプトは以下の機能を提供します：
1. GPT2チェックポイントをHugging Face形式に変換
2. テストデータでの正答率検証
3. パープレキシティ計算
4. テキスト生成品質評価
5. モデルパフォーマンス統計
"""

import torch
import argparse
import time
import json
import numpy as np
import sys
import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# GPT2モデルクラスをインポート
sys.path.append(str(project_root / "gpt2"))
from model import GPT, GPTConfig
from core.dataset import PreparedDataset


def load_gpt2_checkpoint(checkpoint_path, device="cuda"):
    """
    カスタムGPT2チェックポイントをロードする
    """
    print(f"チェックポイントをロードしています: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # モデル設定を取得
        model_args = checkpoint["model_args"]
        print(f"モデル設定: {model_args}")

        # GPTConfigを作成
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # 状態辞書をロード
        state_dict = checkpoint["model"]

        # 不要なプレフィックスを削除
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print("✓ チェックポイントの読み込み成功")

        # 追加情報
        iter_num = checkpoint.get("iter_num", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        config = checkpoint.get("config", {})

        return model, model_args, iter_num, best_val_loss, config

    except Exception as e:
        print(f"✗ チェックポイントの読み込み中にエラーが発生しました: {e}")
        return None, None, None, None, None


def convert_to_hf_format(model, model_args, output_dir):
    """
    カスタムGPTモデルをHugging Face形式に変換する
    """
    print(f"Hugging Face形式に変換中: {output_dir}")

    try:
        # HF GPT2Configを作成
        hf_config = GPT2Config(
            vocab_size=model_args["vocab_size"],
            n_positions=model_args["block_size"],
            n_embd=model_args["n_embd"],
            n_layer=model_args["n_layer"],
            n_head=model_args["n_head"],
            use_cache=True,
            bos_token_id=0,
            eos_token_id=0,
        )

        # HF GPT2LMHeadModelを作成
        hf_model = GPT2LMHeadModel(hf_config)

        # 重みをマッピング
        state_dict_mapping = {}

        # Transformerブロックの重みをマッピング
        for i in range(model_args["n_layer"]):
            # Attention weights
            state_dict_mapping[f"transformer.h.{i}.attn.c_attn.weight"] = (
                f"transformer.h.{i}.attn.c_attn.weight"
            )
            state_dict_mapping[f"transformer.h.{i}.attn.c_proj.weight"] = (
                f"transformer.h.{i}.attn.c_proj.weight"
            )

            # LayerNorm weights
            state_dict_mapping[f"transformer.h.{i}.ln_1.weight"] = (
                f"transformer.h.{i}.ln_1.weight"
            )
            state_dict_mapping[f"transformer.h.{i}.ln_2.weight"] = (
                f"transformer.h.{i}.ln_2.weight"
            )

            # MLP weights
            state_dict_mapping[f"transformer.h.{i}.mlp.c_fc.weight"] = (
                f"transformer.h.{i}.mlp.c_fc.weight"
            )
            state_dict_mapping[f"transformer.h.{i}.mlp.c_proj.weight"] = (
                f"transformer.h.{i}.mlp.c_proj.weight"
            )

            # Bias terms (if exists)
            if model_args.get("bias", False):
                state_dict_mapping[f"transformer.h.{i}.attn.c_attn.bias"] = (
                    f"transformer.h.{i}.attn.c_attn.bias"
                )
                state_dict_mapping[f"transformer.h.{i}.attn.c_proj.bias"] = (
                    f"transformer.h.{i}.attn.c_proj.bias"
                )
                state_dict_mapping[f"transformer.h.{i}.ln_1.bias"] = (
                    f"transformer.h.{i}.ln_1.bias"
                )
                state_dict_mapping[f"transformer.h.{i}.ln_2.bias"] = (
                    f"transformer.h.{i}.ln_2.bias"
                )
                state_dict_mapping[f"transformer.h.{i}.mlp.c_fc.bias"] = (
                    f"transformer.h.{i}.mlp.c_fc.bias"
                )
                state_dict_mapping[f"transformer.h.{i}.mlp.c_proj.bias"] = (
                    f"transformer.h.{i}.mlp.c_proj.bias"
                )

        # 他の重要な重み
        state_dict_mapping["transformer.wte.weight"] = (
            "transformer.wte.weight"  # Token embeddings
        )
        state_dict_mapping["transformer.wpe.weight"] = (
            "transformer.wpe.weight"  # Position embeddings
        )
        state_dict_mapping["transformer.ln_f.weight"] = (
            "transformer.ln_f.weight"  # Final LayerNorm
        )
        state_dict_mapping["lm_head.weight"] = "lm_head.weight"  # Language model head

        if model_args.get("bias", False):
            state_dict_mapping["transformer.ln_f.bias"] = "transformer.ln_f.bias"

        # 重みをコピー
        hf_state_dict = {}
        model_state_dict = model.state_dict()

        for hf_key, custom_key in state_dict_mapping.items():
            if custom_key in model_state_dict:
                hf_state_dict[hf_key] = model_state_dict[custom_key].clone()
            else:
                print(f"警告: {custom_key} がモデルの状態辞書に見つかりません")

        # HFモデルに重みをロード
        hf_model.load_state_dict(hf_state_dict, strict=False)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        hf_model.save_pretrained(output_dir)
        hf_config.save_pretrained(output_dir)

        print(f"✓ Hugging Face形式での保存完了: {output_dir}")
        return hf_model, hf_config

    except Exception as e:
        print(f"✗ Hugging Face形式への変換中にエラー: {e}")
        return None, None


def load_domain_tokenizer(domain, vocab_path=None):
    """ドメイン特化のトークナイザーをロードする"""
    try:
        if domain == "compounds":
            from compounds.utils.tokenizer import CompoundsTokenizer

            vocab_file = vocab_path or "assets/molecules/vocab.txt"
            if not os.path.exists(vocab_file):
                print(f"語彙ファイルが見つかりません: {vocab_file}")
                return None
            return CompoundsTokenizer(vocab_file, 256)

        elif domain == "molecule_nl":
            from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer

            return MoleculeNatLangTokenizer()

        elif domain == "genome":
            from genome_sequence.utils.tokenizer import create_genome_tokenizer

            model_path = vocab_path
            return create_genome_tokenizer(model_path)

        elif domain == "protein_sequence":
            from protein_sequence.utils.bert_tokenizer import (
                create_bert_protein_tokenizer,
            )

            return create_bert_protein_tokenizer()

        elif domain == "rna":
            from rna.utils.bert_tokenizer import create_bert_rna_tokenizer

            return create_bert_rna_tokenizer()

        else:
            print(f"未知のドメイン: {domain}")
            return None

    except ImportError as e:
        print(f"ドメイン特化トークナイザーのインポートに失敗: {e}")
        return None
    except Exception as e:
        print(f"トークナイザーの初期化に失敗: {e}")
        return None


def create_simple_tokenizer(vocab_size):
    """
    簡単なトークナイザーを作成（デバッグ用）
    """
    try:
        # 簡単な語彙を作成
        vocab = {f"<token_{i}>": i for i in range(vocab_size)}
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

        for i, token in enumerate(special_tokens):
            vocab[token] = i

        # PreTrainedTokenizerFastで簡単なトークナイザーを作成
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        tokenizer_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

        # HF wrapper
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )

        return hf_tokenizer

    except Exception as e:
        print(f"簡単なトークナイザー作成エラー: {e}")
        return None


def evaluate_perplexity(
    model, dataset, tokenizer=None, max_samples=1000, device="cuda"
):
    """
    データセットでのパープレキシティを計算する
    """
    print(f"\n=== パープレキシティ評価 ===")
    print(f"評価サンプル数: {min(len(dataset), max_samples)}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(
            range(min(len(dataset), max_samples)), desc="パープレキシティ計算中"
        ):
            try:
                # データを取得
                tokens = dataset[i]
                if torch.is_tensor(tokens):
                    input_ids = tokens.unsqueeze(0).to(device)
                else:
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

                # 入力と目標を分離（正しいサイズ調整）
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                if inputs.size(1) == 0 or targets.size(1) == 0:  # 空の場合はスキップ
                    continue

                if inputs.size(1) != targets.size(1):  # サイズが一致しない場合の調整
                    min_len = min(inputs.size(1), targets.size(1))
                    inputs = inputs[:, :min_len]
                    targets = targets[:, :min_len]

                # モデルで予測
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # 損失を計算
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )

                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
                num_batches += 1

            except Exception as e:
                print(f"サンプル {i} でエラー: {e}")
                continue

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        print(f"✓ 平均損失: {avg_loss:.4f}")
        print(f"✓ パープレキシティ: {perplexity:.4f}")
        print(f"✓ 評価トークン数: {total_tokens:,}")

        return perplexity, avg_loss
    else:
        print("✗ 有効なデータが見つかりませんでした")
        return float("inf"), float("inf")


def generate_text_samples(
    model, tokenizer, device="cuda", num_samples=5, max_length=100
):
    """
    テキスト生成のサンプルを作成する
    """
    print(f"\n=== テキスト生成テスト ===")

    if tokenizer is None:
        print("トークナイザーが利用できないため、テキスト生成をスキップします")
        return []

    model.eval()
    generated_samples = []

    try:
        for i in range(num_samples):
            # 開始トークン
            if (
                hasattr(tokenizer, "bos_token_id")
                and tokenizer.bos_token_id is not None
            ):
                start_token = tokenizer.bos_token_id
            else:
                start_token = 0

            input_ids = torch.tensor([[start_token]]).to(device)

            with torch.no_grad():
                # 生成
                for _ in range(max_length):
                    outputs = model(input_ids)
                    logits = (
                        outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    )

                    # 次のトークンを予測
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits, dim=-1), 1
                    )

                    # 入力に追加
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

                    # 終了条件
                    if (
                        hasattr(tokenizer, "eos_token_id")
                        and next_token.item() == tokenizer.eos_token_id
                    ):
                        break

            # デコード
            if hasattr(tokenizer, "decode"):
                generated_text = tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                )
            else:
                generated_text = " ".join(
                    [f"token_{tid}" for tid in input_ids[0].tolist()]
                )

            generated_samples.append(generated_text)
            print(f"サンプル {i + 1}: {generated_text}")

    except Exception as e:
        print(f"テキスト生成エラー: {e}")

    return generated_samples


def calculate_accuracy_metrics(
    model, dataset, tokenizer=None, max_samples=500, device="cuda"
):
    """
    各種精度メトリクスを計算する
    """
    print(f"\n=== 精度メトリクス計算 ===")

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    top5_correct = 0

    with torch.no_grad():
        for i in tqdm(range(min(len(dataset), max_samples)), desc="精度計算中"):
            try:
                tokens = dataset[i]
                if torch.is_tensor(tokens):
                    input_ids = tokens.unsqueeze(0).to(device)
                else:
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                if inputs.size(1) == 0 or targets.size(1) == 0:  # 空の場合はスキップ
                    continue

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Top-1 精度
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.numel()

                # Top-5 精度
                top5_preds = torch.topk(logits, 5, dim=-1).indices
                top5_correct += sum(
                    [targets[i] in top5_preds[i] for i in range(len(targets))]
                )

            except Exception as e:
                continue

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        top5_accuracy = top5_correct / total_predictions

        print(
            f"✓ Top-1 精度: {accuracy:.4f} ({correct_predictions}/{total_predictions})"
        )
        print(f"✓ Top-5 精度: {top5_accuracy:.4f}")

        return accuracy, top5_accuracy
    else:
        print("✗ 精度計算用の有効なデータが見つかりませんでした")
        return 0.0, 0.0


def test_model_performance(model, model_args, device="cuda"):
    """
    モデルのパフォーマンス統計を取得する
    """
    print(f"\n=== モデルパフォーマンス統計 ===")

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    print(f"モデルサイズ: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"語彙サイズ: {model_args.get('vocab_size', 'Unknown'):,}")
    print(f"ブロックサイズ: {model_args.get('block_size', 'Unknown')}")
    print(f"レイヤー数: {model_args.get('n_layer', 'Unknown')}")
    print(f"ヘッド数: {model_args.get('n_head', 'Unknown')}")
    print(f"埋め込み次元: {model_args.get('n_embd', 'Unknown')}")
    print(f"使用デバイス: {device}")

    # GPU使用量の確認
    if torch.cuda.is_available() and "cuda" in device:
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU メモリ予約量: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "vocab_size": model_args.get("vocab_size"),
        "block_size": model_args.get("block_size"),
        "n_layer": model_args.get("n_layer"),
        "n_head": model_args.get("n_head"),
        "n_embd": model_args.get("n_embd"),
    }


def generate_test_report(checkpoint_path, results, output_dir):
    """テスト結果のレポートを生成"""
    report_path = Path(output_dir) / "gpt2_test_report.json"

    # Tensorなどをシリアライズ可能な形式に変換
    def make_serializable(obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(item) for item in obj)
        else:
            return obj

    report = {
        "checkpoint_path": checkpoint_path,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": make_serializable(results),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ テストレポートを保存しました: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GPT2チェックポイントの包括的テスト・検証スクリプト"
    )
    parser.add_argument(
        "--checkpoint_path", required=True, help="テストするチェックポイントのパス"
    )
    parser.add_argument(
        "--output_dir", default="gpt2_test_output", help="出力ディレクトリ"
    )
    parser.add_argument(
        "--convert_to_hf", action="store_true", help="Hugging Face形式に変換"
    )
    parser.add_argument(
        "--test_dataset_params",
        type=str,
        help="テストデータセットのパラメータ (JSON形式)",
    )
    parser.add_argument(
        "--domain",
        choices=["compounds", "molecule_nl", "genome", "protein_sequence", "rna"],
        help="使用するドメイン",
    )
    parser.add_argument("--vocab_path", help="語彙ファイルのパス")
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=1000,
        help="テストに使用する最大サンプル数",
    )
    parser.add_argument("--device", default="cuda", help="使用するデバイス")

    args = parser.parse_args()

    print("=== GPT2チェックポイント テスト・検証スクリプト ===")
    print(f"チェックポイント: {args.checkpoint_path}")
    print(f"出力ディレクトリ: {args.output_dir}")

    results = {}

    # チェックポイントをロード
    model, model_args, iter_num, best_val_loss, config = load_gpt2_checkpoint(
        args.checkpoint_path, args.device
    )

    if model is None:
        print("チェックポイントの読み込みに失敗したため、テストを終了します。")
        return

    results["checkpoint_info"] = {
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    # Hugging Face形式に変換（オプション）
    hf_model = None
    if args.convert_to_hf:
        hf_output_dir = Path(args.output_dir) / "hf_model"
        hf_model, hf_config = convert_to_hf_format(model, model_args, hf_output_dir)
        if hf_model:
            results["hf_conversion"] = "success"
            model = hf_model  # テストには変換後のモデルを使用
        else:
            results["hf_conversion"] = "failed"

    # トークナイザーをロード（オプション）
    tokenizer = None
    if args.domain:
        print(f"\nドメイン特化トークナイザーを読み込み中: {args.domain}")
        tokenizer = load_domain_tokenizer(args.domain, args.vocab_path)

    if tokenizer is None:
        print("シンプルなトークナイザーを作成中...")
        tokenizer = create_simple_tokenizer(model_args["vocab_size"])

    # テストデータセットをロード
    test_dataset = None
    if args.test_dataset_params:
        try:
            dataset_params = json.loads(args.test_dataset_params)
            test_dataset = PreparedDataset(**dataset_params, split="valid")
            print(f"✓ テストデータセットをロード: {len(test_dataset)} サンプル")
        except Exception as e:
            print(f"テストデータセットの読み込みエラー: {e}")

    if test_dataset is None:
        print("デフォルトのテストデータセットを作成中...")
        # ダミーデータセットを作成（適切なサイズ）
        dummy_data = [
            torch.randint(0, model_args["vocab_size"], (model_args["block_size"],))
            for _ in range(100)
        ]
        test_dataset = dummy_data

    try:
        # モデルパフォーマンス統計
        perf_stats = test_model_performance(model, model_args, args.device)
        results["performance_stats"] = perf_stats

        # パープレキシティ評価
        perplexity, avg_loss = evaluate_perplexity(
            model, test_dataset, tokenizer, args.max_test_samples, args.device
        )
        results["perplexity"] = perplexity
        results["avg_loss"] = avg_loss

        # 精度メトリクス
        accuracy, top5_accuracy = calculate_accuracy_metrics(
            model, test_dataset, tokenizer, args.max_test_samples, args.device
        )
        results["accuracy"] = accuracy
        results["top5_accuracy"] = top5_accuracy

        # テキスト生成サンプル
        generated_samples = generate_text_samples(model, tokenizer, args.device)
        results["generated_samples"] = generated_samples

        results["status"] = "success"

    except Exception as e:
        print(f"\nテスト中にエラーが発生しました: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    # レポート生成
    os.makedirs(args.output_dir, exist_ok=True)
    generate_test_report(args.checkpoint_path, results, args.output_dir)

    print("\n=== テスト完了 ===")
    print(f"結果の詳細は {args.output_dir}/gpt2_test_report.json をご確認ください。")


if __name__ == "__main__":
    main()
