# ESLint & Prettier Setup for molcrawl-web

This project uses **ESLint** and **Prettier** to maintain code quality.

## Tool Configuration

### ESLint

- **Purpose**: Code quality checks, bug detection, and enforcing best practices
- **Config file**: `.eslintrc.json`
- **Base config**: `react-app` (Create React App standard)

### Prettier

- **Purpose**: Consistent code formatting
- **Config file**: `.prettierrc.json`

## Local Usage

### Installation

```bash
cd molcrawl-web
npm install
```

### Lint Checks

```bash
# Check with ESLint
npm run lint

# Fix auto-fixable issues
npm run lint:fix

# Check formatting with Prettier
npm run format:check

# Format with Prettier
npm run format
```

## GitHub Actions

Integrated into the CI/CD pipeline via `.github/workflows/eslint.yml`:

- Automatically runs on JavaScript file changes
- Full check (warnings displayed)
- Strict check (warnings treated as errors)

## ESLint Rules

Key rules:

### Errors (must fix)

- `eqeqeq`: Always use `===`/`!==`
- `curly`: Always use curly braces
- `no-var`: Forbid `var`; use `let`/`const`
- `prefer-const`: Use `const` for variables that are never reassigned
- `no-duplicate-imports`: Forbid duplicate imports

### Warnings

- `no-unused-vars`: Unused variables (variables starting with `_` are excluded)
- `no-console`: Forbid `console.log` (`console.warn`/`console.error` are allowed)
- `react/no-array-index-key`: Do not use array indices as `key` props

### React-specific

- `react/jsx-no-target-blank`: Add `rel="noopener noreferrer"` to `target="_blank"`
- `react/no-danger`: Warn on use of `dangerouslySetInnerHTML`
- `react/prop-types`: PropTypes validation not required (TypeScript preferred)

## Prettier Rules

- **Semicolons**: Yes
- **Quotes**: Double quotes
- **Line length**: 100 characters
- **Indentation**: 2 spaces
- **Trailing commas**: ES5 style

## VS Code Integration

Add the following to `.vscode/settings.json` to auto-format on save:

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

Required extensions:

- ESLint (`dbaeumer.vscode-eslint`)
- Prettier (`esbenp.prettier-vscode`)

## Disabling Errors

### Ignore a single line

```javascript
// eslint-disable-next-line no-console
console.log("debug");

const unused = "test"; // eslint-disable-line no-unused-vars
```

### Ignore an entire file

```javascript
/* eslint-disable no-console */
console.log("console.log allowed in this file");
```

### Ignore a specific rule

```javascript
/* eslint-disable react/no-array-index-key */
const items = arr.map((item, index) => <div key={index}>{item}</div>);
/* eslint-enable react/no-array-index-key */
```

## Troubleshooting

### Clear cache

```bash
cd molcrawl-web
rm -rf node_modules/.cache
npm run lint
```

### Reinstall dependencies

```bash
cd molcrawl-web
rm -rf node_modules package-lock.json
npm install
```

## Reference Links

- [ESLint Documentation](https://eslint.org/)
- [Prettier Documentation](https://prettier.io/)
- [React ESLint Plugin](https://github.com/jsx-eslint/eslint-plugin-react)
- [Create React App - ESLint](https://create-react-app.dev/docs/setting-up-your-editor/#extending-or-replacing-the-default-eslint-config)
