# ESLint & Prettier Setup for molcrawl-web

このプロジェクトでは**ESLint**と**Prettier**を使用してコード品質を管理しています。

## ツール構成

### ESLint
- **目的**: コード品質チェック、バグ検出、ベストプラクティスの強制
- **設定ファイル**: `.eslintrc.json`
- **ベース設定**: `react-app` (Create React App標準)

### Prettier
- **目的**: コードフォーマットの統一
- **設定ファイル**: `.prettierrc.json`

## ローカルでの使用

### インストール

```bash
cd molcrawl-web
npm install
```

### Lintチェック

```bash
# ESLintでチェック
npm run lint

# 自動修正可能な問題を修正
npm run lint:fix

# Prettierでフォーマットチェック
npm run format:check

# Prettierでフォーマット実行
npm run format
```

## GitHub Actions

`.github/workflows/eslint.yml`でCI/CDパイプラインに統合されています：

- JavaScriptファイルの変更時に自動実行
- 全体チェック（警告表示）
- 厳格チェック（警告もエラー扱い）

## ESLintルール

主要なルール：

### エラー（修正必須）
- `eqeqeq`: 常に`===`/`!==`を使用
- `curly`: 常に波括弧を使用
- `no-var`: `var`禁止、`let`/`const`を使用
- `prefer-const`: 再代入しない変数は`const`を使用
- `no-duplicate-imports`: 重複インポート禁止

### 警告
- `no-unused-vars`: 未使用変数（`_`で始まる変数は除外）
- `no-console`: `console.log`禁止（`console.warn`/`console.error`は許可）
- `react/no-array-index-key`: 配列インデックスをkeyに使用しない

### React特有
- `react/jsx-no-target-blank`: `target="_blank"`には`rel="noopener noreferrer"`を追加
- `react/no-danger`: `dangerouslySetInnerHTML`の使用に警告
- `react/prop-types`: PropTypes検証は不要（TypeScript推奨）

## Prettierルール

- **セミコロン**: あり
- **クォート**: ダブルクォート
- **行長**: 100文字
- **インデント**: スペース2個
- **末尾カンマ**: ES5スタイル

## VSCode統合

`.vscode/settings.json`に以下を追加すると、保存時に自動フォーマットされます：

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": true
    }
  },
  "[javascriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": true
    }
  }
}
```

必要な拡張機能：
- ESLint (`dbaeumer.vscode-eslint`)
- Prettier (`esbenp.prettier-vscode`)

## エラーを無視する

### 特定の行を無視

```javascript
// eslint-disable-next-line no-console
console.log("デバッグ用");

const unused = "test"; // eslint-disable-line no-unused-vars
```

### ファイル全体を無視

```javascript
/* eslint-disable no-console */
console.log("このファイルではconsole.logを許可");
```

### 特定のルールを無視

```javascript
/* eslint-disable react/no-array-index-key */
const items = arr.map((item, index) => (
  <div key={index}>{item}</div>
));
/* eslint-enable react/no-array-index-key */
```

## トラブルシューティング

### キャッシュのクリア

```bash
cd molcrawl-web
rm -rf node_modules/.cache
npm run lint
```

### 依存関係の再インストール

```bash
cd molcrawl-web
rm -rf node_modules package-lock.json
npm install
```

## 参考リンク

- [ESLint Documentation](https://eslint.org/)
- [Prettier Documentation](https://prettier.io/)
- [React ESLint Plugin](https://github.com/jsx-eslint/eslint-plugin-react)
- [Create React App - ESLint](https://create-react-app.dev/docs/setting-up-your-editor/#extending-or-replacing-the-default-eslint-config)
