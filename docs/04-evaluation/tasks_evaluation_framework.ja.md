# タスク軸評価フレームワーク (JA)

`molcrawl/tasks/evaluation/` 配下に導入された、5 つの foundation model 群を共通パイプラインで評価するためのフレームワークの使い方です。設計と全体計画は以下を参照してください。

- [`docs/_tmp/20260422-evaluator-implementation-plan.md`](../_tmp/20260422-evaluator-implementation-plan.md): 6 段階の実装計画

## 1. 全体像

評価コードはすべて **タスク軸**（アーキ軸ではない）で整理されています。アーキ差は `ModelAdapter` が吸収します。

```
molcrawl/tasks/evaluation/
  _base/                    # BaseEvaluator / ModelAdapter / MetricRegistry / ReportWriter
  _adapters/                # arch 固有アダプタ (gpt2 など)
  _snapshot/                # 全タスク横断レポート (Phase 6)
  <task>/
    __init__.py
    data_preparation.py
    splits.py
    metrics.py
    evaluator.py
    visualization.py
    configs/<arch>_<size>.yaml
    README.md
```

各評価タスクには対応する `workflows/eval-<task>.sh` が付属します。

## 2. 共通基盤

### 2.1 ModelHandle

評価対象のモデルは、CLI 引数でそのまま組み立てられる `ModelHandle` で表現します。

```python
from molcrawl.tasks.evaluation._base import ModelHandle

handle = ModelHandle(
    arch="gpt2",                         # "gpt2" | "bert" | "chemberta2" | "esm2" | "dnabert2" | "rnaformer"
    modality="genome_sequence",          # foundation model の系統
    model_path="runs_train_gpt2_genome_small/ckpt.pt",
    tokenizer_path="${LEARNING_SOURCE_DIR}/genome_sequence/spm_tokenizer.model",
    size="small",
    extras={"device": "cuda"},
)
```

どのモダリティ・アーキで `--tokenizer-path` に何を渡すかは
[tokenizer_paths.ja.md](tokenizer_paths.ja.md) に一覧があります。

### 2.2 ModelAdapter が提供する capability

- `classification`
- `regression`
- `embedding`
- `likelihood`
- `generation`

各評価タスクは必要な capability だけを要求し、サポートしないアダプタに対しては明示的にエラーを返します。現状で登録済みのアダプタは `gpt2`（likelihood / generation）です。他のアーキは `molcrawl/tasks/evaluation/_adapters/` に追加し、`register_adapter(arch, factory)` で登録してください。

### 2.3 出力フォーマット

すべてのタスクは同じ 2 種類のファイルを `--output-dir` に書き出します。

- `metrics.json`: 機械可読。`{task, modality, arch, category, metrics, details}` を含む。
- `REPORT.md`: 人間可読の 3 軸ヘッダー付きレポート。

この `metrics.json` を Phase 6 の snapshot 集約器が読み込みます。

## 3. タスク一覧とエントリポイント

| カテゴリ | タスク | CLI モジュール | workflow |
|---|---|---|---|
| variant_effect | clinvar | `molcrawl.tasks.evaluation.clinvar` | `workflows/eval-_smoke.sh` |
| variant_effect | cosmic | `molcrawl.tasks.evaluation.cosmic` | `-` |
| variant_effect | omim | `molcrawl.tasks.evaluation.omim` | `-` |
| variant_effect | proteingym | `molcrawl.tasks.evaluation.proteingym` | `workflows/eval-proteingym.sh` |
| variant_effect | gnomad_af_correlation | `molcrawl.tasks.evaluation.gnomad_af_correlation` | `workflows/eval-gnomad.sh` |
| property_prediction | moleculenet | `molcrawl.tasks.evaluation.moleculenet` | `workflows/eval-moleculenet.sh` |
| property_prediction | chembl_scaffold_heldout | `molcrawl.tasks.evaluation.chembl_scaffold_heldout` | `workflows/eval-chembl-heldout.sh` |
| property_prediction | tape | `molcrawl.tasks.evaluation.tape` | `workflows/eval-tape.sh` |
| property_prediction | deeploc | `molcrawl.tasks.evaluation.deeploc` | `workflows/eval-deeploc.sh` |
| sequence_annotation | gue | `molcrawl.tasks.evaluation.gue` | `workflows/eval-gue.sh` |
| generation_quality | moses | `molcrawl.tasks.evaluation.moses` | `workflows/eval-moses.sh` |
| foldability | protein_foldability | `molcrawl.tasks.evaluation.protein_foldability` | `workflows/eval-protein-foldability.sh` |
| cell_type_annotation | rna_benchmark | `molcrawl.tasks.evaluation.rna_benchmark` | `workflows/eval-rna-benchmark.sh` |
| cell_type_annotation | tabula_sapiens | `molcrawl.tasks.evaluation.tabula_sapiens` | `workflows/eval-tabula-sapiens.sh` |
| perturbation_response | replogle_perturb_seq | `molcrawl.tasks.evaluation.replogle_perturb_seq` | `workflows/eval-replogle-perturb-seq.sh` |
| text_alignment | molecule_nat_lang | `molcrawl.tasks.evaluation.molecule_nat_lang` | `workflows/eval-molecule-nat-lang.sh` |
| text_alignment | chebi20 | `molcrawl.tasks.evaluation.chebi20` | `workflows/eval-chebi20.sh` |
| text_alignment | chemllmbench | `molcrawl.tasks.evaluation.chemllmbench` | `workflows/eval-chemllmbench.sh` |

## 4. smoke 実行

Phase 0 の動作確認用ワークフローです。学習済み GPT-2 genome モデルと ClinVar CSV があれば end-to-end が通るかを小サンプルで確認できます。

```bash
export MODEL_PATH=runs_train_gpt2_genome_small/ckpt.pt
export TOKENIZER_PATH=assets/tokenizers/genome.model
export CLINVAR_DATA=learning_source/eval/clinvar/clinvar_small.csv
export MAX_EXAMPLES=16

bash workflows/eval-_smoke.sh
```

`experiment_data/eval/clinvar_smoke/metrics.json` と `REPORT.md` が出力されれば OK です。

## 5. タスクごとの典型的な使い方

### 5.1 ClinVar (variant_effect, genome)

```bash
python -m molcrawl.tasks.evaluation.clinvar \
  --model-path runs_train_gpt2_genome_small/ckpt.pt \
  --tokenizer-path assets/tokenizers/genome.model \
  --clinvar-data learning_source/eval/clinvar/clinvar.csv \
  --arch gpt2 --modality genome_sequence \
  --output-dir experiment_data/eval/clinvar
```

### 5.2 MoleculeNet (property_prediction, compounds)

ディレクトリ構成は `LEARNING_SOURCE_DIR/eval/moleculenet/<subtask>/raw.csv` + `manifest.json` を想定しています。

```bash
export MODEL_PATH=runs_train_chemberta2_small
export MOLECULENET_DIR=learning_source/eval/moleculenet
export SUBTASKS="bbbp esol"
bash workflows/eval-moleculenet.sh
```

個別に呼び出す場合:

```bash
python -m molcrawl.tasks.evaluation.moleculenet \
  --model-path "$MODEL_PATH" \
  --arch chemberta2 --modality compounds \
  --subtask bbbp \
  --task-dir "$MOLECULENET_DIR/bbbp" \
  --output-dir experiment_data/eval/moleculenet/bbbp
```

### 5.3 MOSES (generation_quality, compounds)

```bash
python -m molcrawl.tasks.evaluation.moses \
  --model-path runs_train_gpt2_compounds_small/ckpt.pt \
  --tokenizer-path assets/molecules/spm.model \
  --arch gpt2 --modality compounds \
  --reference-dir learning_source/eval/moses \
  --num-samples 30000 \
  --output-dir experiment_data/eval/moses
```

`moses` Python パッケージがインストールされていれば FCD / SNN / Fragment / Scaffold などの拡張メトリクスも `moses.*` 名前空間で追加されます。未インストール環境では core メトリクス（validity / uniqueness / novelty / internal_diversity）のみが出力されます。

### 5.4 TAPE (property_prediction / sequence_annotation, protein)

```bash
python -m molcrawl.tasks.evaluation.tape \
  --model-path runs_train_esm2_small/ckpt.pt \
  --arch esm2 --modality protein_sequence \
  --task fluorescence \
  --task-dir learning_source/eval/tape/fluorescence \
  --output-dir experiment_data/eval/tape/fluorescence
```

`contact_prediction` は外部構造ラベリングが必要なため、現状は NaN プレースホルダを返します。

### 5.5 GUE 28 タスク一括

```bash
export MODEL_PATH=runs_train_dnabert2_small
export GUE_DIR=learning_source/eval/gue
bash workflows/eval-gue.sh
```

`TASKS` 環境変数で部分実行できます:

```bash
TASKS="prom_300_all H3K4me3 tf_0" bash workflows/eval-gue.sh
```

## 6. 設定 YAML

各タスクの `configs/` に `<arch>_<size>.yaml` のサンプルがあります。中身は CLI 引数と同じキーなので、そのまま pydantic 等に読み込んで `config=` に渡す想定です。

```yaml
# molcrawl/tasks/evaluation/moleculenet/configs/moleculenet_bbbp_chemberta2.yaml
task: moleculenet
subtask: bbbp
modality: compounds
arch: chemberta2
split: scaffold
val_frac: 0.1
test_frac: 0.1
seed: 0
```

## 7. 評価レポートの週次集約 (Phase 6)

すべてのタスクを同じ `--output-dir` 配下へ書き出しておけば、以下で横断 snapshot を生成できます。

```bash
bash workflows/eval-report-weekly.sh
# INPUT_DIR (default: experiment_data/eval) を走査し
# OUTPUT_DIR (default: docs/evaluation) に
# snapshot_<YYYYMMDD>.json と snapshot_<YYYYMMDD>.md を生成
```

前回 snapshot との差分を付けたい場合:

```bash
PREVIOUS=docs/evaluation/snapshot_20260415.json \
  bash workflows/eval-report-weekly.sh
```

スクリプトは次を行います。

1. `INPUT_DIR` 以下の全 `metrics.json` を収集。
2. `(modality, arch, task)` をキーに最新結果のみを保持。
3. 3 軸マトリクス + 前回比 delta top 20 を markdown にレンダリング。

## 8. 新しいタスクを追加する

1. `molcrawl/tasks/evaluation/<task_name>/` を作成し、標準 6 ファイル (`data_preparation.py` / `splits.py` / `metrics.py` / `evaluator.py` / `visualization.py` / `__init__.py` / `__main__.py`) を置く。
2. `evaluator.py` で `BaseEvaluator` を継承し、`task_name`, `category()`, `load_dataset()`, `run_predictions()`, `compute_metrics()` を実装。
3. `configs/<arch>_<size>.yaml` を追加。
4. `workflows/eval-<task>.sh` を追加し、環境変数 → CLI 引数へ橋渡しする。
5. 単体テストを `tests/unit/test_tasks_evaluation_<phase>.py` に追加（合成 `ModelAdapter` を `register_adapter` で登録して pipeline を回せます）。

## 9. 新しいアダプタを追加する

1. `molcrawl/tasks/evaluation/_adapters/<arch>_adapter.py` を作成し、`ModelAdapter` を継承。
2. サポートする capability に対応するメソッドをオーバーライド（例: `embed`, `score_likelihood`, `generate`）。
3. ファイル末尾で `register_adapter("<arch>", MyAdapter)` を呼ぶ。
4. `molcrawl/tasks/evaluation/_adapters/__init__.py` に import を追記。

## 10. 既知の制約

- `gpt2` 以外のアダプタ（bert, chemberta2, esm2, dnabert2, rnaformer）は未登録です。`ModelAdapter` を継承したアダプタを追加した時点で対応するタスクが実データで動くようになります。
- `contact_prediction` (TAPE) / `pfam_hit_rate` (protein_foldability) / ChEBI-20 の BLEU / ROUGE は追加の重い依存（PDB, HMMER, `nltk`, `rouge-score`）が要るため、現状は placeholder もしくは NaN フォールバックです。
- `__main__.py` を直接呼ぶ経路のほかに、設定 YAML をローダー経由で `BaseEvaluator` に渡すランナーはまだ存在しません。必要に応じて小さな `runner.py` を各リポジトリで追加してください。
