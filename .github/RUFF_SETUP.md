# Ruff Linting Setup

このプロジェクトでは[Ruff](https://docs.astral.sh/ruff/)を使用してコード品質をチェックしています。

## GitHub Actions

`.github/workflows/ruff.yml`でCI/CDパイプラインに統合されています：

- **全体チェック**: すべてのルールでチェック（警告のみ、失敗させない）
- **厳格チェック**: 重大な問題（構文エラー、未定義変数など）のみでチェック（失敗する）

## ローカルでの使用

### インストール

```bash
pip install ruff
```

### チェック実行

```bash
# 全体チェック
ruff check .

# 特定のルールのみ
ruff check . --select F,E

# 自動修正可能な問題を修正
ruff check . --fix
```

### 設定

`pyproject.toml`に設定があります：

- **対象Pythonバージョン**: 3.9+
- **行長**: 128文字（blackと同じ）
- **有効なルール**:
  - E: pycodestyle errors
  - F: pyflakes（未定義変数、未使用インポートなど）
  - W: pycodestyle warnings
  - B: flake8-bugbear（一般的なバグパターン）

### 除外パターン

以下は自動的に除外されます：

- `miniconda/`
- `learning_source_*`
- `runs_train_*`
- `gpt2-output*`

## 特定のエラーを無視する

### ファイル全体で無視

```python
# ruff: noqa: E402
import sys
sys.path.append("...")
```

### 特定の行で無視

```python
result = some_function()  # noqa: F841
```

### 設定ファイルで無視

`pyproject.toml`の`[tool.ruff.lint.per-file-ignores]`セクションで設定できます。

## よくあるエラーコード

- **E402**: Module level import not at top of file
- **F401**: Imported but unused
- **F811**: Redefinition of unused
- **F821**: Undefined name
- **E722**: Do not use bare except
- **B006**: Do not use mutable data structures for argument defaults

## 参考リンク

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Rule Reference](https://docs.astral.sh/ruff/rules/)
