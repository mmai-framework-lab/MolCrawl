# Evaluation dataset download workflows

`workflows/data/` ships a set of scripts that fetch every dataset
needed by the Phase 0-5 evaluators into
`LEARNING_SOURCE_DIR/eval/<task>/`.  All scripts emit a manifest
(`manifest.json`) capturing fetch timestamp, SHA-256, license, and
source URL.

## 1. Conventions

- Destination: `${LEARNING_SOURCE_DIR}/eval/<task>/`
- Manifest: `manifest.json` in the same directory.
- Idempotent: files whose SHA-256 already matches the manifest are
  skipped.
- Credentialed datasets (COSMIC / OMIM / DeepLoc / project-internal
  CSVs) exit 0 with `[eval-data] SKIP ...` when the required env vars
  are missing instead of failing silently.
- Shared helpers live in `workflows/data/_eval_data_common.sh`.

Set `LEARNING_SOURCE_DIR` before running any script:

```bash
source molcrawl/config/env.sh   # or: export LEARNING_SOURCE_DIR=...
```

## 2. Download everything in one go

```bash
bash workflows/data/eval-data-all.sh
```

- Runs every task script in turn and prints an OK / SKIPPED / FAILED
  summary at the end.
- Subset run: `EVAL_DATA_TASKS="moleculenet moses" bash workflows/data/eval-data-all.sh`
- Dry run: `EVAL_DATA_DRY_RUN=1 bash workflows/data/eval-data-all.sh`

## 3. Per-task scripts

| Script | Destination | Required env | Notes |
|---|---|---|---|
| `eval-data-clinvar.sh` | `clinvar/` | none | NCBI FTP `variant_summary.txt.gz` etc. |
| `eval-data-cosmic.sh` | `cosmic/` | `COSMIC_EMAIL`, `COSMIC_PASSWORD`, `COSMIC_VERSION` | Sanger COSMIC (login required) |
| `eval-data-omim.sh` | `omim/` | `OMIM_API_KEY` | OMIM API key (terms of use) |
| `eval-data-proteingym.sh` | `proteingym/` | none | substitution zip; set `PROTEINGYM_INCLUDE_INDELS=1` for indels |
| `eval-data-gnomad.sh` | `gnomad_af_correlation/` | none | one chromosome at a time (default `chr22`) |
| `eval-data-moleculenet.sh` | `moleculenet/<subtask>/` | none | DeepChem mirror, 12 datasets |
| `eval-data-moses.sh` | `moses/` | none | MOSES github reference split |
| `eval-data-chembl-heldout.sh` | `chembl_scaffold_heldout/` | `CHEMBL_SCAFFOLD_HELDOUT_SOURCE` | registers an existing CSV |
| `eval-data-tape.sh` | `tape/<task>/` | none | TAPE S3 archives for the 5 tasks |
| `eval-data-deeploc.sh` | `deeploc/` | optional `DEEPLOC_SOURCE` | manual download (license gated) |
| `eval-data-protein-foldability.sh` | `protein_foldability/` | optional `REFERENCE_URL` | defaults to RCSB `pdb_seqres.txt.gz` |
| `eval-data-gue.sh` | `gue/` | none | `GUE.zip` from the DNABERT-2 dataset card |
| `eval-data-rna-benchmark.sh` | `rna_benchmark/` | `RNA_BENCHMARK_SOURCE` | registers an existing JSONL |
| `eval-data-tabula-sapiens.sh` | `tabula_sapiens/` | optional `TABULA_DATASET_URL` | CellxGene H5AD |
| `eval-data-replogle-perturb-seq.sh` | `replogle_perturb_seq/` | optional `REPLOGLE_DATASET_URL` | CellxGene H5AD |
| `eval-data-molecule-nat-lang.sh` | `molecule_nat_lang/` | `MOLECULE_NAT_LANG_SOURCE` | registers an existing CSV |
| `eval-data-chebi20.sh` | `chebi20/` | none | MolT5 ChEBI-20 splits |
| `eval-data-chemllmbench.sh` | `chemllmbench/` | none | 9 ChemLLMBench JSONLs |

## 4. Examples

### 4.1 ClinVar (open access)

```bash
source molcrawl/config/env.sh
bash workflows/data/eval-data-clinvar.sh
ls "$LEARNING_SOURCE_DIR/eval/clinvar"
# variant_summary.txt.gz  submission_summary.txt.gz  manifest.json
```

### 4.2 MoleculeNet (13 tasks)

```bash
bash workflows/data/eval-data-moleculenet.sh
# Decompress the *.csv.gz files manually and rename to raw.csv:
gunzip -k "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/tox21.csv.gz"
mv "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/tox21.csv" \
   "$LEARNING_SOURCE_DIR/eval/moleculenet/tox21/raw.csv"
```

### 4.3 COSMIC (gated)

```bash
export COSMIC_EMAIL="you@example.com"
export COSMIC_PASSWORD="..."
export COSMIC_VERSION="v100"
bash workflows/data/eval-data-cosmic.sh
```

Without the env vars the script exits 0 with
`[eval-data] SKIP cosmic: ...` and does nothing.

### 4.4 gnomAD (per chromosome)

```bash
GNOMAD_CHROM=chr22 bash workflows/data/eval-data-gnomad.sh
GNOMAD_CHROM=chrX  bash workflows/data/eval-data-gnomad.sh
```

## 5. Manifest format

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

The evaluator code does not currently require this manifest, but it is
useful for the Phase 6 snapshot pipeline and for licensing audits.

## 6. Gated datasets

| Dataset | Source | Notes |
|---|---|---|
| COSMIC | https://cancer.sanger.ac.uk/cosmic | Distinct academic / commercial licenses; email registration required. |
| OMIM | https://www.omim.org/api | API key request and terms-of-use acknowledgement required. |
| DeepLoc 2.0 | https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ | DTU SBC license; CSV must be placed manually. |
| Tabula Sapiens / Replogle | https://cellxgene.cziscience.com/ | Multi-GB H5AD files; allocate disk accordingly. |

## 7. Recovering from failures

- HTTP errors: scripts use `set -euo pipefail` and `curl --retry 3`.
  After three retries the script fails; verify the URL and any
  credentials.
- SHA-256 mismatch: compare the offending file with the manifest entry
  and delete the file before re-running to force a fresh download.
- Concurrency: avoid running the same task script in parallel (manifest
  contention).  Different tasks can run concurrently safely.

## 8. Related docs

- Framework usage: [`tasks_evaluation_framework.md`](tasks_evaluation_framework.md)
- Implementation plan: [`../_tmp/20260422-evaluator-implementation-plan.md`](../_tmp/20260422-evaluator-implementation-plan.md)
