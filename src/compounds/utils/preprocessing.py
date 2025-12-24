import logging
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

# RDKitの警告を抑制（無効なSMILES構造の警告が多数出るため）
# ただし、エラーはログに記録する
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)

# 無効なSMILESの統計を保持
_invalid_smiles_count = 0
_total_smiles_count = 0
_invalid_smiles_examples = []  # 無効なSMILESの例を保存（最大10件）


def get_invalid_smiles_stats():
    """
    無効なSMILESの統計を取得
    
    Returns:
        tuple: (無効なSMILES数, 総SMILES数, 無効率, 無効例のリスト)
    """
    if _total_smiles_count == 0:
        return 0, 0, 0.0, []
    invalid_rate = (_invalid_smiles_count / _total_smiles_count) * 100
    return _invalid_smiles_count, _total_smiles_count, invalid_rate, _invalid_smiles_examples


def prepare_scaffolds(smiles: str):
    """
    Prepare the scaffolds of a molecule.
    
    Args:
        smiles: SMILES文字列
        
    Returns:
        str: scaffold SMILES文字列、無効な場合は空文字列
        
    Note:
        ZINC20などの大規模データベースには以下の理由で無効なSMILESが含まれることがあります：
        1. 四級アンモニウム（N+）などのイオン構造の表記問題
        2. 異なるフォーマットからの変換エラー
        3. 特殊な立体化学の表現
        4. データベース作成時の自動処理エラー
        
        これらは通常、データベース全体の数%程度で許容範囲内です。
    """
    global _invalid_smiles_count, _total_smiles_count, _invalid_smiles_examples
    _total_smiles_count += 1
    
    if smiles == "." or not smiles:
        _invalid_smiles_count += 1
        if len(_invalid_smiles_examples) < 10:
            _invalid_smiles_examples.append(("empty or dot", smiles))
        return ""

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            _invalid_smiles_count += 1
            # 最初の10件の無効例を保存
            if len(_invalid_smiles_examples) < 10:
                _invalid_smiles_examples.append(("parse_failed", smiles[:100]))
            
            # 1000件ごとに統計をログ出力
            if _invalid_smiles_count % 1000 == 0:
                invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()
                logger.warning(
                    f"Invalid SMILES detected: {invalid_count}/{total_count} ({invalid_rate:.2f}%)"
                )
            return ""

        scaffold = GetScaffoldForMol(molecule)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception as e:
        _invalid_smiles_count += 1
        if len(_invalid_smiles_examples) < 10:
            _invalid_smiles_examples.append(("exception", f"{smiles[:100]} | Error: {str(e)[:50]}"))
        logger.debug(f"Error processing SMILES '{smiles[:50]}...': {e}")
        return ""
