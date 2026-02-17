"""
Compounds 専用の fixtures とヘルパー関数
"""
import os
import tempfile

import pytest


@pytest.fixture
def sample_vocab_file():
    """テスト用のサンプルvocabファイルを作成"""
    # 基本的なSMILES文字とBERT特殊トークンを含むvocab
    vocab_content = """[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
C
c
N
n
O
o
S
s
P
p
F
Cl
Br
I
(
)
[
]
=
#
-
+
1
2
3
4
5
6
7
8
9
0
"""

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(vocab_content)
        temp_path = f.name

    yield temp_path

    # クリーンアップ
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_smiles_data():
    """テスト用のサンプルSMILESデータ"""
    return {
        "valid": [
            "CCO",  # エタノール
            "c1ccccc1",  # ベンゼン
            "CC(=O)O",  # 酢酸
            "CC(C)C",  # イソブタン
            "C1=CC=C(C=C1)O",  # フェノール
        ],
        "invalid": [
            "",  # 空
            ".",  # ドット
            "INVALID",  # 無効な文字
            "C(C(C",  # 括弧エラー
        ],
        "complex": [
            "C1=CC=C(C=C1)C(=O)O",  # 安息香酸
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # イブプロフェン（簡略版）
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # カフェイン
        ],
    }


@pytest.fixture
def mock_compounds_dataset(tmp_path):
    """テスト用のモックcompoundsデータセット"""
    import pandas as pd

    # サンプルデータ
    data = {
        "smiles": [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "INVALID",  # 無効なSMILES
            "CC(C)C",
        ],
        "label": [0, 1, 0, 1, 0],
    }

    df = pd.DataFrame(data)

    # 一時ファイルに保存
    file_path = tmp_path / "test_compounds.csv"
    df.to_csv(file_path, index=False)

    return file_path


def validate_smiles_output(smiles: str) -> bool:
    """
    生成されたSMILESの妥当性を検証するヘルパー関数

    Args:
        smiles: 検証するSMILES文字列

    Returns:
        bool: 妥当なSMILESの場合True
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def calculate_smiles_metrics(generated_smiles: list, reference_smiles: list = None) -> dict:
    """
    生成されたSMILESの品質メトリクスを計算

    Args:
        generated_smiles: 生成されたSMILES文字列のリスト
        reference_smiles: 参照用SMILES文字列のリスト（オプション）

    Returns:
        dict: メトリクス辞書
    """
    valid_count = sum(1 for s in generated_smiles if validate_smiles_output(s))
    total_count = len(generated_smiles)

    metrics = {
        "total": total_count,
        "valid": valid_count,
        "validity_rate": valid_count / total_count if total_count > 0 else 0,
    }

    return metrics
