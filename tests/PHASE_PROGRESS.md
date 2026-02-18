# Phase Progress Tracking

このドキュメントでは、プロジェクトの各フェーズの進捗を追跡します。

## Phase 1: CBI Conference - Functional Verification

### Task 1-1: BERT (DNA Sequence) Verification

- [ ] モデル初期化テスト
- [ ] トークナイゼーションテスト
- [ ] フォワードパステスト
- [ ] 基本的な予測テスト
- [ ] ドキュメント作成

**Status**: 未着手  
**CI Check**: `gh workflow run phase-validation.yml -f phase=phase1-bert-verification`

### Task 1-2: BERT (Protein Sequence) Verification

- [ ] モデル初期化テスト
- [ ] トークナイゼーションテスト
- [ ] フォワードパステスト
- [ ] 基本的な予測テスト
- [ ] ドキュメント作成

**Status**: 未着手

### Task 1-3: BERT (scRNA-Seq) Verification

- [ ] モデル初期化テスト
- [ ] トークナイゼーションテスト
- [ ] フォワードパステスト
- [ ] 基本的な予測テスト
- [ ] ドキュメント作成

**Status**: 未着手

### Task 1-4: BERT (Compound) Verification

- [ ] モデル初期化テスト
- [ ] SMILES トークナイゼーションテスト
- [ ] フォワードパステスト
- [ ] 基本的な予測テスト
- [ ] ドキュメント作成

**Status**: 未着手

### Task 1-5: BERT (Compound-Language) Verification

- [ ] モデル初期化テスト
- [ ] マルチモーダルトークナイゼーションテスト
- [ ] フォワードパステスト
- [ ] 基本的な予測テスト
- [ ] ドキュメント作成

**Status**: 未着手

### Task 2-1: GPT2 (DNA Sequence) Verification Check

- [ ] モデル初期化テスト
- [ ] 生成テスト
- [ ] 配列検証テスト
- [ ] パフォーマンス評価
- [ ] ドキュメント作成

**Status**: 未着手  
**CI Check**: `gh workflow run phase-validation.yml -f phase=phase1-gpt2-verification`

### Task 2-2: GPT2 (Protein Sequence) Verification Check

- [ ] モデル初期化テスト
- [ ] 生成テスト
- [ ] 配列検証テスト
- [ ] パフォーマンス評価
- [ ] ドキュメント作成

**Status**: 未着手

### Task 2-3: GPT2 (scRNA-seq) Verification Check

- [ ] モデル初期化テスト
- [ ] 生成テスト
- [ ] 配列検証テスト
- [ ] パフォーマンス評価
- [ ] ドキュメント作成

**Status**: 未着手

### Task 2-4: GPT2 (Compound) Verification Check

- [ ] モデル初期化テスト
- [ ] SMILES生成テスト
- [ ] 分子妥当性検証
- [ ] パフォーマンス評価
- [ ] ドキュメント作成

**Status**: 未着手

### Task 2-5: GPT2 (Compound-Lang) Verification Check

- [ ] モデル初期化テスト
- [ ] マルチモーダル生成テスト
- [ ] 結果検証
- [ ] パフォーマンス評価
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3: Documentation and Manualization of Verification Methods

- [ ] 検証手法の体系化
- [ ] マニュアル作成
- [ ] ベストプラクティス文書化
- [ ] トラブルシューティングガイド
- [ ] CI/CD統合ガイド

**Status**: 未着手

---

## Phase 2: Pre-alpha - Dataset Preparation and Method Preparation

### Task 1: Benchmark Data Configuration

- [ ] ベンチマークデータセット選定
- [ ] データパス設定
- [ ] データ検証スクリプト作成
- [ ] CI統合

**Status**: 未着手  
**CI Check**: `gh workflow run phase-validation.yml -f phase=phase2-dataset-prep`

### Task 2-1: Training Log Management Design

- [ ] ログフォーマット設計
- [ ] ログ保存戦略
- [ ] メトリクス定義
- [ ] 可視化要件定義

**Status**: 未着手

### Task 2-2: Training Log Management Implementation

- [ ] ログシステム実装
- [ ] データベース統合
- [ ] API実装
- [ ] テスト作成

**Status**: 未着手

### Task 2-3: Training Log Management Verification

- [ ] ログ記録テスト
- [ ] データ整合性テスト
- [ ] パフォーマンステスト
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3-1: Creation of Dataset for Foundation Model (DNA Sequence)

- [ ] データソース特定
- [ ] データ収集
- [ ] 前処理パイプライン実装
- [ ] データ検証
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3-2: Creation of Dataset for Foundation Model (Protein Sequence)

- [ ] データソース特定
- [ ] データ収集
- [ ] 前処理パイプライン実装
- [ ] データ検証
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3-3: Creation of Dataset for Foundation Model (scRNA-seq)

- [ ] データソース特定
- [ ] データ収集
- [ ] 前処理パイプライン実装
- [ ] データ検証
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3-4: Creation of Dataset for Foundation Model (Compound)

- [ ] データソース特定
- [ ] データ収集
- [ ] 前処理パイプライン実装
- [ ] データ検証
- [ ] ドキュメント作成

**Status**: 未着手

### Task 3-5: Creation of Dataset for Foundation Model (Compound-Lang)

- [ ] データソース特定
- [ ] データ収集
- [ ] 前処理パイプライン実装
- [ ] データ検証
- [ ] ドキュメント作成

**Status**: 未着手

### Task 4-1: Implementation of Scripts for Foundation Model (DNA Sequences)

- [ ] トレーニングスクリプト実装
- [ ] 評価スクリプト実装
- [ ] 設定ファイル作成
- [ ] テスト作成

**Status**: 未着手  
**CI Check**: `gh workflow run phase-validation.yml -f phase=phase2-script-verification`

### Task 4-2: Implementation of Scripts for Foundation Model (Protein Sequences)

- [ ] トレーニングスクリプト実装
- [ ] 評価スクリプト実装
- [ ] 設定ファイル作成
- [ ] テスト作成

**Status**: 未着手

### Task 4-3: Implementation of Scripts for Foundation Model (scRNA-seq)

- [ ] トレーニングスクリプト実装
- [ ] 評価スクリプト実装
- [ ] 設定ファイル作成
- [ ] テスト作成

**Status**: 未着手

### Task 4-4: Implementation of Scripts for Foundation Model (Compound)

- [ ] トレーニングスクリプト実装
- [ ] 評価スクリプト実装
- [ ] 設定ファイル作成
- [ ] テスト作成

**Status**: 未着手

### Task 4-5: Implementation of Scripts for Foundation Model (Compound-Lang)

- [ ] トレーニングスクリプト実装
- [ ] 評価スクリプト実装
- [ ] 設定ファイル作成
- [ ] テスト作成

**Status**: 未着手

### Task 5-1 to 5-5: Verification of Scripts for Base Model Creation

各ドメインのスクリプト検証

- [ ] DNA Sequences
- [ ] Protein Sequences
- [ ] scRNA-seq
- [ ] Compound
- [ ] Compound-Lang

**Status**: 未着手

### Task 6-1: UI Design

- [ ] 要件定義
- [ ] ワイヤーフレーム作成
- [ ] デザインモックアップ
- [ ] レビュー

**Status**: 未着手

### Task 6-2: UI Implementation

- [ ] フロントエンド実装
- [ ] バックエンドAPI統合
- [ ] テスト作成

**Status**: 未着手

### Task 6-3: UI Validation

- [ ] 機能テスト
- [ ] ユーザビリティテスト
- [ ] パフォーマンステスト

**Status**: 未着手

---

## Phase 3: Alpha - Hyperparameter Tuning

### Task 1-1: Alpha Version Base Model Creation & Evaluation (DNA Sequence)

- [ ] ハイパーパラメータ探索
- [ ] モデルトレーニング
- [ ] 評価実施
- [ ] 結果分析
- [ ] ドキュメント作成

**Status**: 未着手  
**CI Check**: `gh workflow run phase-validation.yml -f phase=phase3-alpha-evaluation`

### Task 1-2 to 1-5: Alpha Version Base Model Creation & Evaluation

各ドメインのアルファ版モデル作成と評価

- [ ] Protein Sequence
- [ ] scRNA-seq
- [ ] Compound
- [ ] Compound-Lang

**Status**: 未着手

### Task 2-1: Alpha Release Work and Documentation

- [ ] リリースノート作成
- [ ] インストールガイド作成
- [ ] APIドキュメント作成
- [ ] チュートリアル作成
- [ ] 既知の問題リスト作成

**Status**: 未着手

### Task 2-2: Upload to Hugging Face

- [ ] モデルカード作成
- [ ] メタデータ準備
- [ ] モデルアップロード
- [ ] テストと検証
- [ ] 公開

**Status**: 未着手  
**CI Check**: Release workflowで自動化

### Task 2-3: GitHub Repository Opening

- [ ] READMEファイナライズ
- [ ] ライセンス確認
- [ ] CONTRIBUTING.md作成
- [ ] Issue/PRテンプレート作成
- [ ] リポジトリ公開

**Status**: 未着手

---

## Phase 4: Paper Writing

### Task 1: Identification of Key Points Required for Paper Writing

- [ ] 研究の新規性整理
- [ ] 実験計画策定
- [ ] 比較手法選定
- [ ] 評価指標定義

**Status**: 未着手

### Task 2: Additional Implementation for Paper Writing

- [ ] 追加実験実装
- [ ] ベースライン実装
- [ ] 評価スクリプト拡張

**Status**: 未着手

### Task 3: Verification and Evaluation of Additional Implementation

- [ ] 実験実行
- [ ] 結果検証
- [ ] 統計的検定
- [ ] 図表作成

**Status**: 未着手

---

## 進捗サマリー

| Phase    | 完了タスク | 総タスク数 | 進捗率 |
| -------- | ---------- | ---------- | ------ |
| Phase 1  | 0          | 11         | 0%     |
| Phase 2  | 0          | 23         | 0%     |
| Phase 3  | 0          | 8          | 0%     |
| Phase 4  | 0          | 3          | 0%     |
| **合計** | **0**      | **45**     | **0%** |

最終更新: 2026年1月5日
