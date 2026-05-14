# `--tokenizer-path` に指定するファイル (JA)

`molcrawl/tasks/evaluation/**` の CLI は全て `--tokenizer-path` 引数を共有しています。渡すべき中身は **モダリティ × 学習時に使ったアーキテクチャ** で決まります。このドキュメントは、既存の訓練パイプラインが出力する具体的なパスと、外部 HuggingFace 事前学習モデルを使う場合の指定方法をまとめます。

## 1. 結論だけ欲しい場合

> **現状の実装範囲（2026-04-23 時点）**: `gpt2` アダプタは `genome_sequence` / `compounds` / `protein_sequence` で動作確認済み。`bert` / `esm2` / `chemberta2` / `dnabert2` / `rnaformer` は単一の `HfMlmAdapter` 実装で扱われ、うち `bert` (`genome_sequence` / `compounds`) 、`dnabert2` (`genome_sequence`) 、`chemberta2` (`compounds`) は end-to-end smoke 確認済み。`esm2` / `rnaformer` は実装上は通るが対応する評価器データ未整備で未検証。`molecule_nat_lang` の GPT-2 は tokenizer 経路は通るが既存 checkpoint の重み起因で生成時に失敗する別問題が残っています。詳細は [§6 既知の TODO](#6-既知の-todo)。

| modality (foundation) | arch (CLI `--arch`) | `--tokenizer-path` に渡すもの |
|---|---|---|
| `genome_sequence` | `gpt2`, `bert` | `${LEARNING_SOURCE_DIR}/genome_sequence/spm_tokenizer.model` |
| `genome_sequence` | `dnabert2` | HF repo ID または `--model-path` と同じディレクトリ（例: `zhihan1996/DNABERT-2-117M`） |
| `protein_sequence` | `gpt2`, `bert` | **指定不要**（内蔵 `EsmSequenceTokenizer`）。`--tokenizer-path` を省略 |
| `protein_sequence` | `esm2` | HF repo ID または `--model-path` と同じディレクトリ（例: `facebook/esm2_t6_8M_UR50D`） |
| `compounds` | `gpt2`, `bert` | `assets/molecules/vocab.txt` |
| `compounds` | `chemberta2` | HF repo ID または `--model-path` と同じディレクトリ（例: `seyonec/ChemBERTa-zinc-base-v1`） |
| `rna` | `gpt2`, `bert` | **指定不要**（内蔵 `TranscriptomeTokenizer`）。`--tokenizer-path` を省略 |
| `rna` | `rnaformer` | HF repo ID または `--model-path` と同じディレクトリ |
| `molecule_nat_lang` | `gpt2`, `bert` | **指定不要**（内蔵 `MoleculeNatLangTokenizer` = HF GPT-2 tokenizer） |

ここで言う「内蔵トークナイザ」は、チェックポイントの読み込み時に PyPI 経由でロードされるもので、外部ファイルは必要ありません。その場合は CLI 呼び出しで `--tokenizer-path` オプション自体を外してください（`None` が渡ります）。

## 2. なぜこの対応になるのか

`molcrawl` の訓練コードが各モダリティで採用しているトークナイザは次の通りです。

- **genome_sequence (GPT-2 / BERT)**: SentencePiece（`spm.SentencePieceProcessor`）。パスは `molcrawl.config.paths.get_refseq_tokenizer_path()` が返す `"<LEARNING_SOURCE_DIR>/genome_sequence/spm_tokenizer.model"` に固定されています (`molcrawl/config/paths.py:22`)。
- **protein_sequence (GPT-2 / BERT)**: `molcrawl.protein_sequence.dataset.tokenizer.EsmSequenceTokenizer` (HuggingFace `EsmTokenizer` ラッパ)。ファイルパスは持ちません。
- **compounds (GPT-2 / BERT)**: `molcrawl.compounds.utils.tokenizer.CompoundsTokenizer` で、ボキャブラリファイル `assets/molecules/vocab.txt` を読みます (`molcrawl/gpt2/configs/compounds/train_gpt2_small_config.py:14`, `molcrawl/bert/configs/compounds.py:9`)。
- **rna (GPT-2 / BERT)**: `molcrawl.rna.dataset.geneformer.tokenizer.TranscriptomeTokenizer`。内部で Geneformer の WordLevel 辞書を構築するのでパス指定不要です。
- **molecule_nat_lang (GPT-2 / BERT)**: `molcrawl.molecule_nat_lang.utils.tokenizer.MoleculeNatLangTokenizer`（HF の GPT-2 tokenizer ラッパ）。外部ファイル不要。

外部 HuggingFace 系モデル（ChemBERTa-2 / ESM-2 / DNABERT-2 / RNAformer）は、それぞれの PyPI / HF 配布の Auto Tokenizer を使います。`--model-path` が HF のスナップショットディレクトリであれば、同じディレクトリを `--tokenizer-path` に渡せば十分です。

## 3. 各評価タスクでの具体例

### 3.1 ClinVar / COSMIC / OMIM / gnomAD (genome_sequence)

訓練した nanoGPT / BERT を使う場合:

```bash
export LSD="$LEARNING_SOURCE_DIR"
python -m molcrawl.tasks.evaluation.clinvar \
  --model-path "$LSD/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt" \
  --tokenizer-path "$LSD/genome_sequence/spm_tokenizer.model" \
  --clinvar-data "$LSD/eval/clinvar/clinvar.csv" \
  --arch gpt2 --modality genome_sequence \
  --output-dir experiment_data/eval/clinvar
```

DNABERT-2 を使う場合:

```bash
python -m molcrawl.tasks.evaluation.gue \
  --model-path zhihan1996/DNABERT-2-117M \
  --tokenizer-path zhihan1996/DNABERT-2-117M \
  --arch dnabert2 --modality genome_sequence \
  --task prom_300_all \
  --task-dir "$LSD/eval/gue/prom_300_all" \
  --output-dir experiment_data/eval/gue/prom_300_all
```

### 3.2 ProteinGym / TAPE / DeepLoc / foldability (protein_sequence)

訓練した GPT-2 protein / BERT protein は **`--tokenizer-path` を渡さない** のが正解です。

```bash
python -m molcrawl.tasks.evaluation.proteingym \
  --model-path "$LSD/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt" \
  --arch gpt2 --modality protein_sequence \
  --proteingym-data "$LSD/eval/proteingym/substitutions.csv" \
  --output-dir experiment_data/eval/proteingym
```

ESM-2 を使う場合:

```bash
python -m molcrawl.tasks.evaluation.tape \
  --model-path facebook/esm2_t6_8M_UR50D \
  --tokenizer-path facebook/esm2_t6_8M_UR50D \
  --arch esm2 --modality protein_sequence \
  --task fluorescence \
  --task-dir "$LSD/eval/tape/fluorescence" \
  --output-dir experiment_data/eval/tape/fluorescence
```

### 3.3 MoleculeNet / MOSES / ChEMBL scaffold (compounds)

内製 GPT-2 compound:

```bash
python -m molcrawl.tasks.evaluation.moses \
  --model-path "$LSD/compounds/gpt2-output/compounds-small/ckpt.pt" \
  --tokenizer-path assets/molecules/vocab.txt \
  --arch gpt2 --modality compounds \
  --reference-dir "$LSD/eval/moses" \
  --output-dir experiment_data/eval/moses
```

ChemBERTa-2 の probe 用途:

```bash
python -m molcrawl.tasks.evaluation.moleculenet \
  --model-path seyonec/ChemBERTa-zinc-base-v1 \
  --tokenizer-path seyonec/ChemBERTa-zinc-base-v1 \
  --arch chemberta2 --modality compounds \
  --subtask bbbp \
  --task-dir "$LSD/eval/moleculenet/bbbp" \
  --output-dir experiment_data/eval/moleculenet/bbbp
```

### 3.4 rna_benchmark / Tabula Sapiens / Replogle (rna)

内製 BERT / GPT-2 rna:

```bash
python -m molcrawl.tasks.evaluation.rna_benchmark \
  --model-path "$LSD/rna/bert-output/rna-small" \
  --arch bert --modality rna \
  --rna-jsonl "$LSD/eval/rna_benchmark/source.jsonl" \
  --output-dir experiment_data/eval/rna_benchmark
```

### 3.5 molecule_nat_lang / ChEBI-20 / ChemLLMBench

内製 GPT-2 / BERT molecule_nat_lang は `--tokenizer-path` 不要:

```bash
python -m molcrawl.tasks.evaluation.chebi20 \
  --model-path "$LSD/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt" \
  --arch gpt2 --modality molecule_nat_lang \
  --dataset-dir "$LSD/eval/chebi20" \
  --output-dir experiment_data/eval/chebi20
```

## 4. よくある質問

### Q1. `--tokenizer-path` を渡さないとエラーになりますか？
`gpt2` アダプタは modality で分岐します:

- `genome_sequence`: 必須（SentencePiece モデルを指す）。未指定は `ValueError`。
- `compounds`: 任意。未指定時は `assets/molecules/vocab.txt` を使用。
- `molecule_nat_lang` / `protein_sequence`: 無視（内蔵 `MoleculeNatLangTokenizer` / `EsmSequenceTokenizer` を使用。指定すると info ログを出して無視）。

`bert` / `esm2` / `chemberta2` / `dnabert2` / `rnaformer` は共通の `HfMlmAdapter` 実装で動作します。tokenizer 解決の優先順位:

1. `--tokenizer-path` を `AutoTokenizer.from_pretrained` で読み込み
2. checkpoint ディレクトリ同梱の tokenizer ファイル (`tokenizer.json` 等)
3. `(arch, modality)` ごとのフォールバック
   - `bert` / `chemberta2` + `compounds` → `CompoundsTokenizer`
   - `bert` + `molecule_nat_lang` → `MoleculeNatLangTokenizer`
   - `bert` / `esm2` + `protein_sequence` → `BertProteinSequenceTokenizer`
   - `dnabert2` + `genome_sequence` / `rnaformer` + `rna` → `get_custom_tokenizer_path(modality, arch)` 直下の HF tokenizer ディレクトリ

モデル本体は `AutoModelForMaskedLM.from_pretrained(model_path)` で読み、`config.json` の `architectures` に応じて `BertForMaskedLM` / `EsmForMaskedLM` / `RobertaForMaskedLM` 等が自動選択されます。HfMlmAdapter は MLM 専用なので `generate()` は `NotImplementedError`。生成タスクには `gpt2` を使ってください。

### Q2. 学習時にディレクトリを移動してしまいました
`spm_tokenizer.model` と `assets/molecules/vocab.txt` は訓練時と同じファイルを指してください。SentencePiece はモデルサイズが一致していないとロード時に無言で異なる vocab を使ってしまい、スコアが壊れます。`molcrawl.experiment_tracker` に残っている訓練時の `config` / `environment` を参照するのが確実です。

### Q3. HF リポの snapshot を別ディレクトリに持っていたい
`transformers` の `snapshot_download()` でローカルキャッシュに落としたディレクトリを `--tokenizer-path` / `--model-path` 双方に渡せば動きます（例: `/root/.cache/huggingface/hub/models--facebook--esm2_t6_8M_UR50D/snapshots/<hash>`）。

## 5. 関連

- 全体の使い方: [`tasks_evaluation_framework.ja.md`](tasks_evaluation_framework.ja.md)
- データ取得: [`eval_dataset_downloaders.ja.md`](eval_dataset_downloaders.ja.md)
- 既存の BERT / GPT-2 テスター（内製モデルのトークナイザ配置の参考になります）: [`README_bert_tester.ja.md`](README_bert_tester.ja.md), [`README_gpt2_tester.ja.md`](README_gpt2_tester.ja.md)

## 6. 既知の TODO

adapter レジストリは現在 `gpt2` / `bert` / `esm2` / `chemberta2` / `dnabert2` / `rnaformer` の 6 種類 (`molcrawl/tasks/evaluation/_adapters/__init__.py`)。

- `GPT2Adapter` は modality 分岐で tokenizer を切り替え、`genome_sequence` / `compounds` / `protein_sequence` で smoke 動作確認済み。
- `HfMlmAdapter` は MLM 系 5 arch 共通の実装。`AutoModelForMaskedLM.from_pretrained` で本体を読み、PLL(pseudo-log-likelihood)で `score_likelihood()` を実装。smoke 動作確認済みは `bert + genome_sequence` / `bert + compounds` / `dnabert2 + genome_sequence` / `chemberta2 + compounds`。残る `esm2` / `rnaformer` は tokenizer load とモデル load は通るが、対応する評価器データが未整備で end-to-end smoke は未実行。`generate()` は MLM なのでサポート外。

残っている項目:

| modality | arch | 現在の症状 | 必要対応 |
|---|---|---|---|
| `molecule_nat_lang` | `gpt2` | tokenizer は通るが `model.generate()` 内で `torch.multinomial` が nan/inf で assert。既存 checkpoint (small は vocab 50002 への旧ドリフト、medium は vocab 50257 一致だが forward が nan 出力) 側の問題で、アダプタ層の修正で救える範囲を超えている | 既存 checkpoint の weight 健全性確認、または再訓練 |
| `rna` | `gpt2` | interface が異質（`TranscriptomeTokenizer` は loom 単位で `encode(str)` を持たない） | RNA 評価系は `encode(text)` 前提のアダプタとは別経路が必要 |
| `rna` | `bert` / `rnaformer` | HfMlmAdapter 上は動くが、既存 RNA 評価器が `encode(text)` を期待しないデータ(pre-tokenised JSONL)を受け取る設計のため、評価器側との繋ぎが未整備 | RNA 評価器(rna_benchmark / Tabula Sapiens / Replogle)の入力形式を確認し、必要ならアダプタまたは評価器側で tokens 経路を分岐 |
| `protein_sequence` | `esm2` / `bert` | アダプタ・モデルは load 成功するが、評価器データ (ProteinGym は 404、TAPE / DeepLoc データ未ダウンロード) が欠落 | `workflows/data/eval-data-proteingym.sh` の URL 修正 (Zenodo/HF へ移行済み)、TAPE のダウンロード、検証 |
| `molecule_nat_lang` | `bert` | アダプタは通るが対応する評価器データ (pairs_csv) が未整備 | `RNA_BENCHMARK_SOURCE` 的な project-internal データを stage する手順整備、または別評価器のデータ準備 |
| `compounds` | `bert` | アダプタは通るが MoleculeNet の bace サブタスクは 403。他サブタスク (bbbp / tox21 / esol など) は smoke 済み | `workflows/data/eval-data-moleculenet.sh` の bace URL 修正または差し替え |

関連ファイル:

- [`molcrawl/tasks/evaluation/_adapters/gpt2_adapter.py`](../../molcrawl/tasks/evaluation/_adapters/gpt2_adapter.py)
- [`molcrawl/tasks/evaluation/_adapters/__init__.py`](../../molcrawl/tasks/evaluation/_adapters/__init__.py)
- [`molcrawl/tasks/evaluation/_base/model_adapter.py`](../../molcrawl/tasks/evaluation/_base/model_adapter.py) (`register_adapter` / `build_adapter`)

副次的に判明した外部データ取得 URL の死活:

- `workflows/data/eval-data-proteingym.sh` が叩く `https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip` は **404**。ProteinGym は配布場所が Zenodo / HuggingFace に移行済みで URL 再調整が必要。
- `workflows/data/eval-data-moleculenet.sh` が叩く `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace_classification.csv` は **403**。他の MoleculeNet サブタスク (bbbp / tox21 / hiv / esol / ...) は取得可能。
