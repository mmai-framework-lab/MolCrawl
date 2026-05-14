# 評価用データセット ダウンロードワークフロー (JA)

`workflows/data/` 以下に、Phase 0〜5 の評価タスクが必要とするデータセットを `LEARNING_SOURCE_DIR/eval/<task>/` に配置するためのスクリプト群を用意しています。すべて `tasks/evaluation` の標準レイアウトに沿って `manifest.json`（取得日時 / SHA-256 / ライセンス / ソース URL）を出力します。

## 1. 共通仕様

- 出力先: `${LEARNING_SOURCE_DIR}/eval/<task>/`
- マニフェスト: 同ディレクトリの `manifest.json`
- 既に同一 SHA-256 のファイルがある場合は再ダウンロードしない（idempotent）。
- 認証が必要なデータセット（COSMIC / OMIM / DeepLoc など）は、必須環境変数が無いときに `[eval-data] SKIP ...` として 0 終了し、設定方法を案内します。
- ヘルパー関数は `workflows/data/_eval_data_common.sh` に集約しています。

呼び出し前に `LEARNING_SOURCE_DIR` を設定してください:

```bash
source molcrawl/config/env.sh   # もしくは export LEARNING_SOURCE_DIR=...
```

## 2. 一括ダウンロード

```bash
bash workflows/data/eval-data-all.sh
```

- 各タスクスクリプトを順番に実行し、最後に OK / SKIPPED / FAILED のサマリを出力します。
- 一部だけ実行したい場合: `EVAL_DATA_TASKS="moleculenet moses" bash workflows/data/eval-data-all.sh`
- 何が走るかだけ確認したい場合: `EVAL_DATA_DRY_RUN=1 bash workflows/data/eval-data-all.sh`

## 3. タスク別スクリプト一覧

| スクリプト | 出力先サブディレクトリ | 必須環境変数 | 備考 |
|---|---|---|---|
| `eval-data-clinvar.sh` | `clinvar/` | なし | NCBI FTP から `variant_summary.txt.gz` ほか |
| `eval-data-cosmic.sh` | `cosmic/` | `COSMIC_EMAIL`, `COSMIC_PASSWORD`, `COSMIC_VERSION` | Sanger COSMIC（要登録） |
| `eval-data-omim.sh` | `omim/` | `OMIM_API_KEY` | OMIM API キー（要承認） |
| `eval-data-proteingym.sh` | `proteingym/` | なし | substitutions zip（必要に応じて `PROTEINGYM_INCLUDE_INDELS=1`） |
| `eval-data-gnomad.sh` | `gnomad_af_correlation/` | なし | 1 染色体ずつ（既定: `chr22`） |
| `eval-data-moleculenet.sh` | `moleculenet/<subtask>/` | なし | DeepChem ミラーから 12 種 |
| `eval-data-moses.sh` | `moses/` | なし | MOSES github の train/test/test_scaffolds |
| `eval-data-chembl-heldout.sh` | `chembl_scaffold_heldout/` | `CHEMBL_SCAFFOLD_HELDOUT_SOURCE` | 既存 CSV を登録 |
| `eval-data-tape.sh` | `tape/<task>/` | なし | TAPE 公式 S3 から 5 タスク分の `*.tar.gz` |
| `eval-data-deeploc.sh` | `deeploc/` | （任意）`DEEPLOC_SOURCE` | ライセンス同意必要のため手動取得を登録 |
| `eval-data-protein-foldability.sh` | `protein_foldability/` | （任意）`REFERENCE_URL` | 既定は RCSB `pdb_seqres.txt.gz` |
| `eval-data-gue.sh` | `gue/` | なし | DNABERT-2 の `GUE.zip` |
| `eval-data-rna-benchmark.sh` | `rna_benchmark/` | `RNA_BENCHMARK_SOURCE` | 既存 JSONL を登録 |
| `eval-data-tabula-sapiens.sh` | `tabula_sapiens/` | （任意）`TABULA_DATASET_URL` | CellxGene の H5AD |
| `eval-data-replogle-perturb-seq.sh` | `replogle_perturb_seq/` | （任意）`REPLOGLE_DATASET_URL` | CellxGene の H5AD |
| `eval-data-molecule-nat-lang.sh` | `molecule_nat_lang/` | `MOLECULE_NAT_LANG_SOURCE` | 既存 CSV を登録 |
| `eval-data-chebi20.sh` | `chebi20/` | なし | MolT5 リポの train/validation/test |
| `eval-data-chemllmbench.sh` | `chemllmbench/` | なし | ChemLLMBench 9 タスクの JSONL |

## 4. 実行例

### 4.1 ClinVar（公開データ、無認証）

```bash
source molcrawl/config/env.sh
bash workflows/data/eval-data-clinvar.sh
ls "$LEARNING_SOURCE_DIR/eval/clinvar"
# variant_summary.txt.gz  submission_summary.txt.gz  manifest.json
```

### 4.2 MoleculeNet 全 13 タスク

```bash
bash workflows/data/eval-data-moleculenet.sh
# *.csv.gz は手動で gunzip し、各タスクディレクトリに raw.csv として置く
gunzip -k "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/tox21.csv.gz"
mv "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/tox21.csv" \
   "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/raw.csv"
```

### 4.3 COSMIC（要登録）

```bash
export COSMIC_EMAIL="you@example.com"
export COSMIC_PASSWORD="..."
export COSMIC_VERSION="v100"
bash workflows/data/eval-data-cosmic.sh
```

環境変数が無い場合は `[eval-data] SKIP cosmic: ...` と表示して何もしません。

### 4.4 gnomAD（必要な染色体だけ）

```bash
GNOMAD_CHROM=chr22 bash workflows/data/eval-data-gnomad.sh
GNOMAD_CHROM=chrX  bash workflows/data/eval-data-gnomad.sh
```

## 5. マニフェストの形式

```json
{
  "task": "moleculenet",
  "name": "MoleculeNet (DeepChem mirror)",
  "home": "https://moleculenet.org/",
  "license": "BSD-3-Clause + per-dataset licenses",
  "version": "20260422",
  "fetched_at": "2026-04-22T06:22:13Z",
  "files": [
    {
      "path": "bbbp/raw.csv",
      "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
      "sha256": "..."
    }
  ]
}
```

`molcrawl.tasks.evaluation` 側のローダーはこのマニフェストの存在をチェックしているわけではありませんが、Phase 6 のスナップショット集約や監査用途で参照できます。

## 6. 認証付きデータセットの注意

| データセット | 入手元 | 注意点 |
|---|---|---|
| COSMIC | https://cancer.sanger.ac.uk/cosmic | 商用 / 学術で別ライセンス。E-Mail 登録必須 |
| OMIM | https://www.omim.org/api | API キー申請が必要。利用規約の同意必要 |
| DeepLoc 2.0 | https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ | DTU SBC ライセンス。CSV を手動配置 |
| Tabula Sapiens | https://cellxgene.cziscience.com/ | H5AD のサイズに注意（数 GB 〜） |

## 7. 失敗時のリカバリ

- HTTP エラー: スクリプトは `set -euo pipefail` で停止します。`curl --retry 3` で 3 回リトライしても駄目なら、URL や認証情報を確認してください。
- SHA-256 不一致: マニフェストの該当エントリと現状ファイルを比較します。再ダウンロードする場合はファイルを削除してから再実行してください。
- 並列実行: 同じタスクのスクリプトを並列起動するとマニフェストが競合するので避けてください。タスク同士の並列は安全です。

## 8. 関連

- 評価フレームワーク使い方: [`tasks_evaluation_framework.ja.md`](tasks_evaluation_framework.ja.md)
- 評価実装計画: [`../_tmp/20260422-evaluator-implementation-plan.md`](../_tmp/20260422-evaluator-implementation-plan.md)
