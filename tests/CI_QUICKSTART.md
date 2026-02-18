# CI/CD Quick Start Guide

このガイドでは、プロジェクトのCI/CD システムの使い方を簡単に説明します。

## 🚀 基本的な使い方

### 1. コードをプッシュする

```bash
git add .
git commit -m "feat: implement DNA tokenizer"
git push origin feature/dna-tokenizer
```

→ 自動的にCIテスト（`ci-tests.yml`）が実行されます

### 2. フェーズ検証を実行する

#### Phase 1: BERT検証

```bash
gh workflow run phase-validation.yml -f phase=phase1-bert-verification
```

#### Phase 1: GPT2検証

```bash
gh workflow run phase-validation.yml -f phase=phase1-gpt2-verification
```

#### Phase 2: データセット準備検証

```bash
gh workflow run phase-validation.yml -f phase=phase2-dataset-prep
```

#### Phase 2: スクリプト検証

```bash
gh workflow run phase-validation.yml -f phase=phase2-script-verification
```

#### Phase 3: アルファ評価

```bash
gh workflow run phase-validation.yml -f phase=phase3-alpha-evaluation
```

### 3. ローカルでテストを実行する

```bash
# 全テスト実行
pytest

# Phase 1のテストのみ
pytest -m phase1

# DNAドメインのテストのみ
pytest -m dna

# カバレッジレポート付き
pytest --cov=src --cov-report=html
```

### 4. ドキュメントをビルドする

```bash
cd docs
make html
```

または、GitHub ActionsでREADMEをプッシュすると自動的にドキュメントがビルドされます。

### 5. パフォーマンスベンチマークを実行する

```bash
# 手動実行
gh workflow run benchmark.yml

# ローカル実行
pytest tests/benchmarks/ --benchmark-only
```

## 📊 ワークフロー一覧

| ワークフロー           | トリガー   | 目的                  |
| ---------------------- | ---------- | --------------------- |
| `ci-tests.yml`         | push, PR   | コード品質とテスト    |
| `ruff.yml`             | push, PR   | Pythonコードのlinting |
| `eslint.yml`           | push, PR   | JavaScriptのlinting   |
| `phase-validation.yml` | 手動       | フェーズ固有の検証    |
| `documentation.yml`    | push, 手動 | ドキュメント生成      |
| `benchmark.yml`        | 週次, 手動 | パフォーマンス測定    |
| `release.yml`          | タグ, 手動 | リリース準備          |

## 🔧 よくあるタスク

### 新しい機能を追加する時

1. フィーチャーブランチを作成

   ```bash
   git checkout -b feature/your-feature
   ```

2. テストを書く

   ```bash
   # tests/unit/test_your_feature.py
   ```

3. 実装する

   ```bash
   # src/your_module.py
   ```

4. ローカルでテスト

   ```bash
   pytest tests/unit/test_your_feature.py -v
   ```

5. プッシュしてPR作成

   ```bash
   git push origin feature/your-feature
   ```

### Phase完了時の手順

1. すべてのタスクが完了したことを確認
2. フェーズ検証ワークフローを実行

   ```bash
   gh workflow run phase-validation.yml -f phase=phase1-bert-verification
   ```

3. 結果を確認（GitHub Actionsタブ）
4. PHASE_PROGRESS.mdを更新
5. 次のフェーズに進む

### リリース準備

1. バージョンタグを作成

   ```bash
   git tag alpha-0.1.0
   git push origin alpha-0.1.0
   ```

2. 自動的にリリースワークフローが実行される

3. Draft Releaseを確認して公開

## 🐛 トラブルシューティング

### CIが失敗する

1. ローカルで同じテストを実行

   ```bash
   pytest tests/ -v
   ```

2. lintingエラーを修正

   ```bash
   ruff check . --fix
   ```

3. 型チェックエラーを確認

   ```bash
   mypy src/
   ```

### テストがスキップされる

- 依存関係がインストールされているか確認
- `pytest.skip()`の条件を確認

### ワークフローが見つからない

- `.github/workflows/`ディレクトリにファイルが存在するか確認
- YAMLシンタックスエラーがないか確認

## 📚 詳細情報

- 詳細なCI/CDガイド: [.github/CI_CD_GUIDE.md](.github/CI_CD_GUIDE.md)
- テストドキュメント: [tests/README.md](tests/README.md)
- フェーズ進捗: [PHASE_PROGRESS.md](PHASE_PROGRESS.md)

## 💡 ヒント

- `gh` CLIをインストールすると、コマンドラインからワークフローを簡単に実行できます

  ```bash
  brew install gh  # macOS
  # または https://cli.github.com/
  ```

- VS Code拡張機能「GitHub Actions」を使うと、エディタからワークフローを管理できます

- テストにマーカーを付けると、特定のテストだけを実行できます

  ```python
  @pytest.mark.phase1
  @pytest.mark.dna
  def test_dna_feature():
      pass
  ```
