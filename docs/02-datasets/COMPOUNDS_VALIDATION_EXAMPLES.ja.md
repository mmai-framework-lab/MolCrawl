# Compounds Validation - 実践例

## 実際の使用例：Compoundsの処理が正しいかをGitHub Actionsで検証

このドキュメントでは、作成したCI/CDシステムを使ってCompounds（化合物）処理を検証する具体的な手順を示します。

##  シナリオ1: SMILES validation を追加・変更した場合

### 状況 (シナリオ1)

`molcrawl/compounds/utils/preprocessing.py` のSMILES検証ロジックを改善したい。

### 手順 (シナリオ1)

#### 1. ローカルでテストを実行

```bash
# Compounds関連のテストを実行
pytest tests/unit/test_compounds.py::TestSmilesValidation -v

# 期待される出力:
# tests/unit/test_compounds.py::TestSmilesValidation::test_valid_smiles PASSED
# tests/unit/test_compounds.py::TestSmilesValidation::test_invalid_smiles PASSED
# tests/unit/test_compounds.py::TestSmilesValidation::test_complex_valid_smiles PASSED
```

#### 2. コードを変更

```python
# molcrawl/compounds/utils/preprocessing.py
def prepare_scaffolds(smiles: str):
    # 改善されたロジックを追加
    if not smiles or smiles.strip() == "":
        return ""

    # 新しい検証ロジック
    if len(smiles) > 1000:  # 異常に長いSMILESを拒否
        logger.warning(f"SMILES too long: {len(smiles)} characters")
        return ""

    # 既存のロジック...
```

#### 3. ローカルで再テスト

```bash
pytest tests/unit/test_compounds.py::TestSmilesValidation -v
```

#### 4. GitHubにプッシュ

```bash
git add molcrawl/compounds/utils/preprocessing.py
git commit -m "feat(compounds): add length validation for SMILES"
git push origin feature/improve-smiles-validation
```

#### 5. GitHub Actionsの結果を確認

```text
GitHub → Actions タブ → "Compounds Validation" ワークフロー

 unit-tests: SUCCESS
 smiles-validation: SUCCESS
   Invalid SMILES: 2/100 (2.00%)
   ✓ Invalid SMILES rate is acceptable
```

##  シナリオ2: 新しいtokenizerロジックを追加

### 状況 (シナリオ2)

特殊な化学構造（立体化学など）に対応するため、tokenizerを拡張したい。

### 手順 (シナリオ2)

#### 1. テストファーストで開発

```python
# tests/unit/test_compounds.py に追加
@pytest.mark.unit
@pytest.mark.compound
def test_stereochemistry_tokenization(self, sample_vocab_file):
    """立体化学表記が正しくトークン化されることを確認"""
    from molcrawl.compounds.utils.tokenizer import SmilesTokenizer

    # `C[C@H](O)C` のような立体化学を含むSMILES
    smiles = "C[C@H](O)C"

    tokenizer = SmilesTokenizer(sample_vocab_file)
    tokens = tokenizer.tokenize(smiles)

    # @ が正しく認識される
    assert "@" in tokens or "[C@H]" in tokens
```

#### 2. テストを実行（失敗するはず）

```bash
pytest tests/unit/test_compounds.py::test_stereochemistry_tokenization -v
# FAILED - 新機能が未実装なので失敗
```

#### 3. 機能を実装

```python
# molcrawl/compounds/utils/tokenizer.py
SMI_REGEX_PATTERN = r"""(
    \[[^\]]+]|        # 角括弧内
    Br?|Cl?|N|O|S|P|F|I|  # 原子
    b|c|n|o|s|p|      # 芳香族
    @|@@|             # 立体化学 ← 追加
    \(|\)|            # 括弧
    \.|=|#|-|\+|\\|\/|:|~|\?|>>?|\*|\$|
    \%[0-9]{2}|
    [0-9]
)"""
```

#### 4. テストを再実行（成功するはず）

```bash
pytest tests/unit/test_compounds.py::test_stereochemistry_tokenization -v
# PASSED
```

#### 5. 全体のテストを実行

```bash
pytest tests/unit/test_compounds.py -v
```

#### 6. GitHubで検証

```bash
git add molcrawl/compounds/utils/tokenizer.py tests/unit/test_compounds.py
git commit -m "feat(compounds): support stereochemistry in tokenizer"
git push

# GitHub Actionsが自動実行:
#  unit-tests
#  tokenization-tests
```

##  シナリオ3: Phase 1 BERT モデル検証

### 状況 (シナリオ3)

Compounds用BERTモデルをトレーニングした。Phase 1検証を実行したい。

### 手順 (シナリオ3)

#### 1. モデルパスを設定

```bash
# ローカルテスト用
export COMPOUNDS_BERT_MODEL_PATH=/path/to/trained/bert/model
```

#### 2. 統合テストを実行

```bash
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsBERTIntegration -v

# 期待される出力:
# test_bert_model_loading PASSED
#   ✓ BERT model loaded successfully
# test_bert_tokenizer_loading PASSED
#   ✓ Tokenizer loaded successfully
#   Vocab size: 500
# test_bert_inference_pipeline PASSED
#   ✓ BERT inference successful
#   Input SMILES: CCO
#   Output shape: torch.Size([1, 10, 500])
```

#### 3. 手動でPhase 1検証ワークフローを実行

```bash
gh workflow run compounds-validation.yml
```

または、GitHub Web UIから:

```text
Actions → Compounds Validation → Run workflow
```

#### 4. 結果を確認

```text
Artifacts:
- phase1-compounds-verification-report.md をダウンロード

内容:
# Compounds Phase 1 Verification Report
Date: 2026-01-05

## BERT Verification Status
-  Model initialization
-  Tokenization pipeline
-  Inference test

Status: PASSED
```

##  シナリオ4: GPT-2でSMILES生成品質を検証

### 状況 (シナリオ4)

GPT-2モデルで生成されるSMILESの品質を確認したい。

### 手順 (シナリオ4)

#### 1. 統合テストを実行

```bash
export COMPOUNDS_GPT2_MODEL_PATH=/path/to/trained/gpt2/model

pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration::test_gpt2_generated_smiles_validity -v
```

#### 2. 結果を確認

```text
✓ SMILES Validity Check:
  Total generated: 15
  Valid SMILES: 12
  Validity rate: 80.0%

PASSED
```

#### 3. 品質が低い場合のデバッグ

```bash
# より詳細なテストを実行
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration -v --tb=long

# 生成されたSMILESを確認:
# 1. CCO
# 2. c1ccccc1O
# 3. INVALID_STRUCTURE ← これが問題
```

#### 4. モデルを再トレーニングまたはハイパーパラメータ調整

```bash
# 温度パラメータを下げる（より保守的な生成）
# temperature: 0.8 → 0.6

# 再テスト
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration -v
```

##  シナリオ5: Pull Request での自動検証

### 状況 (シナリオ5)

チームメンバーがCompounds関連のコードを変更したPRを作成。

### 手順 (シナリオ5)

#### 1. PR作成

```bash
# フィーチャーブランチを作成
git checkout -b feature/add-new-smiles-feature
git add molcrawl/compounds/
git commit -m "feat: add new SMILES feature"
git push origin feature/add-new-smiles-feature

# GitHub でPRを作成
```

#### 2. 自動チェックが開始

```text
GitHub PR画面:
 Compounds Validation - unit-tests
 Compounds Validation - integration-tests
 Compounds Validation - smiles-validation
 Compounds Validation - tokenization-tests
 Compounds Validation - phase1-verification (running)
```

#### 3. レビュアーが結果を確認

```text
PR Check Details:

 All checks passed

Details:
- Unit Tests: 25/25 passed
- Integration Tests: 8/8 passed
- Invalid SMILES rate: 3.2% (acceptable)
- Tokenization: All tests passed

Artifacts:
 compounds-validation-summary.md
```

#### 4. マージ

```text
全チェックが → "Merge pull request" ボタンが有効化
```

##  実際の検証フロー図

```text
開発者がコード変更
    ↓
ローカルでpytest実行
    ↓
問題なし？
    ↓ Yes
Git push
    ↓
GitHub Actions 自動実行
    ├── Unit Tests (1-2分)
    ├── Integration Tests (3-5分)
    ├── SMILES Validation (30秒)
    ├── Tokenization Tests (30秒)
    └── Phase 1 Verification (オプション)
    ↓
全て？
    ↓ Yes
レポート生成・保存
    ↓
PRマージ可能
```

##  よくある使い方

### 1. 毎日のルーチン開発

```bash
# 朝: 最新を取得
git pull origin develop

# 機能追加
vim molcrawl/compounds/utils/preprocessing.py

# テスト追加
vim tests/unit/test_compounds.py

# ローカル検証
pytest tests/unit/test_compounds.py -v

# プッシュ（CIが自動実行）
git push
```

### 2. 週次モデル検証

```bash
# 毎週金曜日にモデルの統合テストを実行
gh workflow run compounds-validation.yml

# 結果をチーム会議で確認
```

### 3. リリース前の完全検証

```bash
# Phase 1完了時
gh workflow run phase-validation.yml -f phase=phase1-bert-verification
gh workflow run phase-validation.yml -f phase=phase1-gpt2-verification

# 全Compoundsテスト
gh workflow run compounds-validation.yml -f test_level=all
```

##  メトリクスの見方

### 合格基準の例

```yaml
Unit Tests:  100% pass required
SMILES Validation:  Invalid rate < 50%
GPT-2 Validity:  > 50% valid SMILES
Test Coverage:  > 80% (目標)
Test Duration:  < 5 minutes (目標)
```

### 警告が出た場合

```text
 Invalid SMILES rate: 45%
→ まだ許容範囲内だが、原因を調査すべき

 GPT-2 validity: 48%
→ 基準(50%)をわずかに下回る。モデル調整を検討

 Unit test failed: 2/25
→ 直ちに修正が必要
```

##  まとめ

この実践例から分かること：

1. **迅速なフィードバック**: ローカル → CI → レポートの流れで素早く問題発見
2. **品質基準の明確化**: Invalid SMILES率、妥当性率などの具体的指標
3. **自動化の価値**: Push時に自動検証、手動での詳細検証も可能
4. **チーム協業**: PRでの自動チェックでコードレビューを支援

**次のステップ**: [COMPOUNDS_VALIDATION_GUIDE.ja.md](COMPOUNDS_VALIDATION_GUIDE.ja.md) で詳細を確認
