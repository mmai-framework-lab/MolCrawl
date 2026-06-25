# What to pass to `--tokenizer-path`

Every CLI under `molcrawl/tasks/evaluation/**` accepts the same
`--tokenizer-path` argument.  The right value depends on **which
architecture the checkpoint was trained with** and the modality.  This
document lists the concrete paths produced by the existing training
pipelines, together with the conventions for externally hosted
HuggingFace models.

## 1. Quick reference

> **Current implementation status (as of 2026-04-23)**: the `gpt2`
> adapter is end-to-end verified for `genome_sequence`, `compounds`,
> and `protein_sequence`. The `bert` / `esm2` / `chemberta2` /
> `dnabert2` /archs share a single `HfMlmAdapter`
> implementation; `bert` (`genome_sequence` / `compounds`), `dnabert2`
> (`genome_sequence`), and `chemberta2` (`compounds`) are smoke-verified
> end-to-end. `esm2` /load correctly but lack ready
> evaluator data. `molecule_nat_lang` GPT-2 routes correctly but the
> existing checkpoints hit a separate model-side issue at generation
> time. See [§6 Known TODO](#6-known-todo).

| modality (foundation) | arch (`--arch`) | what to pass to `--tokenizer-path` |
|---|---|---|
| `genome_sequence` | `gpt2`, `bert` | `${LEARNING_SOURCE_DIR}/genome_sequence/spm_tokenizer.model` |
| `genome_sequence` | `dnabert2` | HF repo id / same dir as `--model-path` (e.g. `zhihan1996/DNABERT-2-117M`) |
| `protein_sequence` | `gpt2`, `bert` | **omit the flag** - the adapter uses the built-in `EsmSequenceTokenizer` |
| `protein_sequence` | `esm2` | HF repo id / same dir as `--model-path` (e.g. `facebook/esm2_t6_8M_UR50D`) |
| `compounds` | `gpt2`, `bert` | `assets/molecules/vocab.txt` |
| `compounds` | `chemberta2` | HF repo id / same dir as `--model-path` (e.g. `seyonec/ChemBERTa-zinc-base-v1`) |
| `rna` | `gpt2`, `bert` | **omit the flag** - built-in `TranscriptomeTokenizer` |
| `rna` || HF repo id / same dir as `--model-path` |
| `molecule_nat_lang` | `gpt2`, `bert` | **omit the flag** - built-in `MoleculeNatLangTokenizer` (HF GPT-2) |

"Built-in tokenizer" means the class is instantiated from Python (PyPI
packages) at load time, with no external files required.  In those
cases simply drop the `--tokenizer-path` flag (it defaults to `None`).

## 2. Why these values

The in-house training code wires each modality to the following
tokenizers:

- **genome_sequence (GPT-2 / BERT)**: SentencePiece.  The path is
  pinned by `molcrawl.config.paths.get_refseq_tokenizer_path()` to
  `"<LEARNING_SOURCE_DIR>/genome_sequence/spm_tokenizer.model"`
  (`molcrawl/config/paths.py:22`).
- **protein_sequence (GPT-2 / BERT)**:
  `molcrawl.protein_sequence.dataset.tokenizer.EsmSequenceTokenizer`
  (wrapper around the HF `EsmTokenizer`).  No external file.
- **compounds (GPT-2 / BERT)**:
  `molcrawl.compounds.utils.tokenizer.CompoundsTokenizer`, initialised
  from `assets/molecules/vocab.txt`
  (`molcrawl/gpt2/configs/compounds/train_gpt2_small_config.py:14`,
  `molcrawl/bert/configs/compounds.py:9`).
- **rna (GPT-2 / BERT)**:
  `molcrawl.rna.dataset.geneformer.tokenizer.TranscriptomeTokenizer`.
  Builds the Geneformer WordLevel vocabulary internally.
- **molecule_nat_lang (GPT-2 / BERT)**:
  `molcrawl.molecule_nat_lang.utils.tokenizer.MoleculeNatLangTokenizer`
  (wraps the HF GPT-2 tokenizer).  No external file.

External HuggingFace models (ChemBERTa-2 / ESM-2 / DNABERT-2)
use their own `AutoTokenizer`; pass the same directory (or
repo id) to both `--model-path` and `--tokenizer-path`.

## 3. Per-task examples

### 3.1 ClinVar / COSMIC / OMIM / gnomAD (genome_sequence)

Trained nanoGPT / BERT:

```bash
export LSD="$LEARNING_SOURCE_DIR"
python -m molcrawl.tasks.evaluation.clinvar \
  --model-path "$LSD/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt" \
  --tokenizer-path "$LSD/genome_sequence/spm_tokenizer.model" \
  --clinvar-data "$LSD/eval/clinvar/clinvar.csv" \
  --arch gpt2 --modality genome_sequence \
  --output-dir experiment_data/eval/clinvar
```

DNABERT-2:

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

In-house GPT-2 / BERT protein models **do not** need
`--tokenizer-path`:

```bash
python -m molcrawl.tasks.evaluation.proteingym \
  --model-path "$LSD/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt" \
  --arch gpt2 --modality protein_sequence \
  --proteingym-data "$LSD/eval/proteingym/substitutions.csv" \
  --output-dir experiment_data/eval/proteingym
```

ESM-2:

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

In-house GPT-2 compound:

```bash
python -m molcrawl.tasks.evaluation.moses \
  --model-path "$LSD/compounds/gpt2-output/compounds-small/ckpt.pt" \
  --tokenizer-path assets/molecules/vocab.txt \
  --arch gpt2 --modality compounds \
  --reference-dir "$LSD/eval/moses" \
  --output-dir experiment_data/eval/moses
```

ChemBERTa-2 probe:

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

In-house BERT / GPT-2 rna:

```bash
python -m molcrawl.tasks.evaluation.rna_benchmark \
  --model-path "$LSD/rna/bert-output/rna-small" \
  --arch bert --modality rna \
  --rna-jsonl "$LSD/eval/rna_benchmark/source.jsonl" \
  --output-dir experiment_data/eval/rna_benchmark
```

### 3.5 molecule_nat_lang / ChEBI-20 / ChemLLMBench

In-house GPT-2 / BERT molecule_nat_lang does not need
`--tokenizer-path`:

```bash
python -m molcrawl.tasks.evaluation.chebi20 \
  --model-path "$LSD/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt" \
  --arch gpt2 --modality molecule_nat_lang \
  --dataset-dir "$LSD/eval/chebi20" \
  --output-dir experiment_data/eval/chebi20
```

## 4. FAQ

### Is `--tokenizer-path` required?
The `gpt2` adapter branches on modality:

- `genome_sequence`: required (points at the SentencePiece model).
  Missing path raises `ValueError`.
- `compounds`: optional; defaults to `assets/molecules/vocab.txt`.
- `molecule_nat_lang` / `protein_sequence`: ignored; the adapter uses
  the built-in `MoleculeNatLangTokenizer` / `EsmSequenceTokenizer`
  (if you pass a path it will be logged and dropped).

The `bert` / `esm2` / `chemberta2` / `dnabert2` /archs
share a single `HfMlmAdapter` implementation. Tokenizer resolution
order:

1. `--tokenizer-path` via `AutoTokenizer.from_pretrained`
2. The checkpoint directory itself (co-saved `tokenizer.json` etc.)
3. An `(arch, modality)` fallback:
   - `bert` / `chemberta2` + `compounds` → `CompoundsTokenizer`
   - `bert` + `molecule_nat_lang` → `MoleculeNatLangTokenizer`
   - `bert` / `esm2` + `protein_sequence` → `BertProteinSequenceTokenizer`
   - `dnabert2` + `genome_sequence` and+ `rna` →
     `AutoTokenizer.from_pretrained(get_custom_tokenizer_path(modality, arch))`

The model is loaded via `AutoModelForMaskedLM.from_pretrained`, which
dispatches to the concrete class (`BertForMaskedLM`, `EsmForMaskedLM`,
`RobertaForMaskedLM`, ...) from the checkpoint's `config.json`.
`HfMlmAdapter` is MLM-only and raises `NotImplementedError` for
`generate()`; use `gpt2` for generation tasks.

### I moved directories after training
Always point at the *same* SentencePiece model / vocab file that was
used during training.  SentencePiece silently produces different token
ids with a different model, and the metrics will be meaningless.  The
original path is preserved under `config` / `environment` in
`molcrawl.experiment_tracker` entries for the training run.

### I want to keep HF snapshots outside the default cache
Use `transformers.snapshot_download()` (or `huggingface-cli download`)
to materialise the snapshot into a directory, then pass that directory
to both `--tokenizer-path` and `--model-path` (e.g.
`/root/.cache/huggingface/hub/models--facebook--esm2_t6_8M_UR50D/snapshots/<hash>`).

## 5. Related docs

- Framework usage: [`tasks_evaluation_framework.md`](tasks_evaluation_framework.md)
- Dataset downloaders: [`eval_dataset_downloaders.md`](eval_dataset_downloaders.md)
- Existing BERT / GPT-2 testers (useful when checking where the
  in-house tokenizers live in your installation):
  [`README_bert_tester.md`](README_bert_tester.md),
  [`README_gpt2_tester.md`](README_gpt2_tester.md)

## 6. Known TODO

The adapter registry now contains `gpt2`, `bert`, `esm2`, `chemberta2`,
`dnabert2`, and(`molcrawl/tasks/evaluation/_adapters/__init__.py`).

- `GPT2Adapter.load()` routes tokenizer loading by modality and is
  smoke-verified for `genome_sequence`, `compounds`, and
  `protein_sequence`.
- `HfMlmAdapter.load()` is the shared implementation for the five MLM
  architectures. It loads via `AutoModelForMaskedLM.from_pretrained`,
  walks the three-tier tokenizer fallback, and scores sequences as
  pseudo-log-likelihood. Smoke-verified end-to-end for
  `bert + genome_sequence`, `bert + compounds`, `dnabert2 +
  genome_sequence`, and `chemberta2 + compounds`. `esm2`
  load correctly in isolation but have no ready evaluator data to
  verify end-to-end. `generate()` is not supported.

Remaining gaps:

| modality | arch | current symptom | required work |
|---|---|---|---|
| `molecule_nat_lang` | `gpt2` | tokenizer routes correctly but `model.generate()` trips `torch.multinomial` on nan/inf logits with the existing checkpoints (`-small` has a legacy vocab drift to 50002; `-medium` matches vocab 50257 but the forward still produces nan) — not fixable at the adapter layer | audit the checkpoint weights / retrain |
| `rna` | `gpt2` | interface mismatch: `TranscriptomeTokenizer` operates on loom files and has no `encode(str)` method | design a separate RNA adapter path (pre-tokenised JSONL) |
| `rna` | `bert` /| HfMlmAdapter loads fine, but the existing RNA evaluators expect pre-tokenised cell JSONLs rather than strings to `encode(text)` | decide whether to bridge in the evaluator or add a token-id input path to the adapter |
| `protein_sequence` | `esm2` / `bert` | adapter loads, but evaluator data is missing (ProteinGym 404, TAPE not downloaded) | fix `workflows/data/eval-data-proteingym.sh` URL (Zenodo/HF) and stage TAPE |
| `molecule_nat_lang` | `bert` | adapter loads, but no matching evaluator data is staged (pairs_csv etc.) | stage project-internal pairs_csv or pick a different evaluator |
| `compounds` | `bert` | bace subtask downloader returns 403; other subtasks (bbbp / tox21 / esol / ...) work | fix `workflows/data/eval-data-moleculenet.sh` bace URL or substitute |

Relevant files:

- [`molcrawl/tasks/evaluation/_adapters/gpt2_adapter.py`](../../molcrawl/tasks/evaluation/_adapters/gpt2_adapter.py)
- [`molcrawl/tasks/evaluation/_adapters/__init__.py`](../../molcrawl/tasks/evaluation/_adapters/__init__.py)
- [`molcrawl/tasks/evaluation/_base/model_adapter.py`](../../molcrawl/tasks/evaluation/_base/model_adapter.py) (`register_adapter` / `build_adapter`)

Side findings on dataset downloaders:

- `workflows/data/eval-data-proteingym.sh` hits `https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip` which returns **404**. ProteinGym has moved distribution to Zenodo / HuggingFace; the URL needs updating.
- `workflows/data/eval-data-moleculenet.sh` fails on `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace_classification.csv` with **403** (other MoleculeNet subtasks such as bbbp / tox21 / hiv / esol still download fine).
