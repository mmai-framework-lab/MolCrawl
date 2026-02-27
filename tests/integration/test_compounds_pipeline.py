"""
Compounds の統合テスト - 実際のモデルとデータパイプラインの検証
"""

import os

import pytest


@pytest.mark.integration
@pytest.mark.compound
class TestCompoundsEndToEnd:
    """Compounds の end-to-end 統合テスト"""

    def test_smiles_to_scaffold_pipeline(self):
        """SMILES → Scaffold の完全なパイプラインをテスト"""
        from molcrawl.compounds.utils.preprocessing import prepare_scaffolds

        # 実際の化合物例
        test_cases = [
            ("CCO", False),  # エタノール - 環なし
            ("c1ccccc1", True),  # ベンゼン - 環あり
            ("CC(=O)O", False),  # 酢酸 - 環なし
            ("INVALID_SMILES", False),  # 無効
            ("", False),  # 空 - 無効
        ]

        results = []
        for smiles, should_be_valid in test_cases:
            scaffold = prepare_scaffolds(smiles)
            is_valid = scaffold != ""

            results.append(
                {
                    "smiles": smiles,
                    "scaffold": scaffold,
                    "expected_valid": should_be_valid,
                    "actual_valid": is_valid,
                    "passed": is_valid == should_be_valid,
                }
            )

        # 結果を表示
        for result in results:
            print(
                f"SMILES: {result['smiles'][:20]:20s} | "
                f"Expected: {result['expected_valid']:5} | "
                f"Actual: {result['actual_valid']:5} | "
                f"{'✓' if result['passed'] else '✗'}"
            )

        # 全てのテストケースがパスしたことを確認
        assert all(r["passed"] for r in results), "Some test cases failed"

    def test_batch_smiles_processing(self):
        """大量のSMILESをバッチ処理できることを確認"""
        from molcrawl.compounds.utils.preprocessing import get_invalid_smiles_stats, prepare_scaffolds

        # 大量のSMILESデータをシミュレート
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CC(C)C",
            "C1=CC=C(C=C1)O",
        ] * 20  # 100個のSMILES

        scaffolds = []
        for smiles in test_smiles:
            scaffold = prepare_scaffolds(smiles)
            scaffolds.append(scaffold)

        # 統計を確認
        invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()

        print("\nBatch Processing Results:")
        print(f"  Processed: {len(test_smiles)} SMILES")
        print(f"  Valid scaffolds: {len([s for s in scaffolds if s != ''])}")
        print(f"  Invalid rate: {invalid_rate:.2f}%")

        # 全てのSMILESが処理された
        assert len(scaffolds) == len(test_smiles)
        # 環構造のSMILESのみscaffoldを持つ
        valid_count = len([s for s in scaffolds if s != ""])
        assert valid_count >= len(test_smiles) * 0.2  # 環構造は1/5程度


@pytest.mark.integration
@pytest.mark.compound
@pytest.mark.slow
class TestCompoundsBERTIntegration:
    """Compounds BERT モデルの統合テスト"""

    @pytest.fixture
    def bert_model_path(self):
        """BERT モデルのパスを返す（実際のパスに置き換える）"""
        # 環境変数から取得、または実際のパスを指定
        path = os.environ.get("COMPOUNDS_BERT_MODEL_PATH")
        if path and os.path.exists(path):
            return path
        pytest.skip("BERT model path not found. Set COMPOUNDS_BERT_MODEL_PATH environment variable.")

    def test_bert_model_loading(self, bert_model_path):
        """BERT モデルが正しくロードできることを確認"""
        from transformers import BertForMaskedLM

        try:
            model = BertForMaskedLM.from_pretrained(bert_model_path)
            assert model is not None
            print(f"✓ BERT model loaded successfully from {bert_model_path}")
        except Exception as e:
            pytest.fail(f"Failed to load BERT model: {e}")

    def test_bert_tokenizer_loading(self, bert_model_path):
        """BERT tokenizer が正しくロードできることを確認"""
        from molcrawl.compounds.utils.tokenizer import SmilesTokenizer

        vocab_path = os.path.join(bert_model_path, "vocab.txt")
        if not os.path.exists(vocab_path):
            pytest.skip(f"Vocab file not found at {vocab_path}")

        try:
            tokenizer = SmilesTokenizer(vocab_path)
            assert tokenizer is not None
            print("✓ Tokenizer loaded successfully")
            print(f"  Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            pytest.fail(f"Failed to load tokenizer: {e}")

    def test_bert_inference_pipeline(self, bert_model_path):
        """BERT モデルで推論が実行できることを確認"""
        import torch
        from transformers import BertForMaskedLM

        from molcrawl.compounds.utils.tokenizer import SmilesTokenizer

        vocab_path = os.path.join(bert_model_path, "vocab.txt")
        if not os.path.exists(vocab_path):
            pytest.skip(f"Vocab file not found at {vocab_path}")

        try:
            # モデルとトークナイザーをロード
            model = BertForMaskedLM.from_pretrained(bert_model_path)
            tokenizer = SmilesTokenizer(vocab_path)

            model.eval()

            # サンプルSMILES
            test_smiles = "CCO"  # エタノール

            # トークン化
            inputs = tokenizer(test_smiles, return_tensors="pt")

            # 推論
            with torch.no_grad():
                outputs = model(**inputs)

            assert outputs is not None
            assert outputs.logits is not None
            print("✓ BERT inference successful")
            print(f"  Input SMILES: {test_smiles}")
            print(f"  Output shape: {outputs.logits.shape}")

        except Exception as e:
            pytest.fail(f"BERT inference failed: {e}")


@pytest.mark.integration
@pytest.mark.compound
@pytest.mark.slow
class TestCompoundsGPT2Integration:
    """Compounds GPT2 モデルの統合テスト"""

    @pytest.fixture
    def gpt2_model_path(self):
        """GPT2 モデルのパスを返す（実際のパスに置き換える）"""
        path = os.environ.get("COMPOUNDS_GPT2_MODEL_PATH")
        if path and os.path.exists(path):
            return path
        pytest.skip("GPT2 model path not found. Set COMPOUNDS_GPT2_MODEL_PATH environment variable.")

    def test_gpt2_model_loading(self, gpt2_model_path):
        """GPT2 モデルが正しくロードできることを確認"""
        from transformers import GPT2LMHeadModel

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            assert model is not None
            print(f"✓ GPT2 model loaded successfully from {gpt2_model_path}")
        except Exception as e:
            pytest.fail(f"Failed to load GPT2 model: {e}")

    def test_gpt2_smiles_generation(self, gpt2_model_path):
        """GPT2 で SMILES を生成できることを確認"""
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)

            model.eval()

            # 開始トークン
            prompt = "C"

            # 入力を準備
            inputs = tokenizer(prompt, return_tensors="pt")

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=3,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # デコード
            generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            print("✓ GPT2 generation successful")
            print(f"  Prompt: {prompt}")
            print(f"  Generated {len(generated_smiles)} SMILES:")
            for i, smiles in enumerate(generated_smiles, 1):
                print(f"    {i}. {smiles[:50]}")

            assert len(generated_smiles) == 3

        except Exception as e:
            pytest.fail(f"GPT2 generation failed: {e}")

    def test_gpt2_generated_smiles_validity(self, gpt2_model_path):
        """生成された SMILES の妥当性を確認"""
        import torch
        from rdkit import Chem
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
            model.eval()

            # 複数の開始点から生成
            prompts = ["C", "c1", "CC"]

            all_generated = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                        num_return_sequences=5,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                all_generated.extend(generated)

            # 妥当性をチェック
            valid_count = 0
            for smiles in all_generated:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1

            validity_rate = (valid_count / len(all_generated)) * 100

            print("\n✓ SMILES Validity Check:")
            print(f"  Total generated: {len(all_generated)}")
            print(f"  Valid SMILES: {valid_count}")
            print(f"  Validity rate: {validity_rate:.1f}%")

            # 有効率が一定以上であることを確認（50%以上を期待）
            assert validity_rate >= 50, f"Validity rate too low: {validity_rate:.1f}%"

        except Exception as e:
            pytest.fail(f"Validity check failed: {e}")


@pytest.mark.integration
@pytest.mark.compound
class TestCompoundsDatasetIntegration:
    """Compounds データセットの統合テスト"""

    def test_dataset_loading(self, mock_compounds_dataset):
        """モックデータセットが正しくロードできることを確認"""
        import pandas as pd

        df = pd.read_csv(mock_compounds_dataset)

        assert "smiles" in df.columns
        assert len(df) > 0
        print(f"✓ Dataset loaded: {len(df)} compounds")

    def test_dataset_preprocessing(self, mock_compounds_dataset):
        """データセット前処理パイプラインをテスト"""
        import pandas as pd

        from molcrawl.compounds.utils.preprocessing import prepare_scaffolds

        df = pd.read_csv(mock_compounds_dataset)

        # Scaffoldを生成
        df["scaffold"] = df["smiles"].apply(prepare_scaffolds)

        # 統計
        valid_scaffolds = df[df["scaffold"] != ""]
        invalid_scaffolds = df[df["scaffold"] == ""]

        print("\n✓ Dataset Preprocessing:")
        print(f"  Total: {len(df)}")
        print(f"  Valid: {len(valid_scaffolds)}")
        print(f"  Invalid: {len(invalid_scaffolds)}")

        # 環構造を含むデータのみscaffoldが生成される
        assert len(valid_scaffolds) >= 1
