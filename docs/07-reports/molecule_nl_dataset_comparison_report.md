# Molecule NL Dataset Structure Comparison Report

## Comparison Scope

- Old dataset: `learning_source`
- New dataset: `learning_source/molecule_nat_lang/arrow_splits`
- Run date: 2025-11-25

## Key Differences

### 1. Sample Count

| Split    | Old Data      | New Data      | Difference           |
| -------- | ------------- | ------------- | -------------------- |
| train    | 3,288,855     | 3,267,176     | -21,679 (-0.66%)     |
| test     | 33,061        | 30,344        | -2,717 (-8.21%)      |
| valid    | 20,498        | 17,781        | -2,717 (-13.25%)     |
| **Total**| **3,342,414** | **3,315,301** | **-27,113 (-0.81%)** |

Reason:

- New implementation added SMILES validation
- Invalid chemical samples were removed via `validate_smiles_in_sample()`

### 2. Column Schema Changes

Old dataset had 18 columns, new dataset has 10 columns.

#### Removed columns

- `input`
- `output`
- `input_core_tag_left`
- `input_core_tag_right`
- `output_core_tag_left`
- `output_core_tag_right`
- `raw_input`
- `raw_output`
- `sample_id`
- `split`
- `target`
- `task`

#### Added columns

- `__index_level_0__`
- `input_too_long`
- `task_type` (renamed from `task` concept)
- `valid_sample`

### 3. Shared Columns and Type Compatibility

The following shared columns are type-compatible across old/new datasets:

- `attention_mask`
- `input_ids`
- `input_text`
- `labels`
- `output_ids`
- `real_input_text`

Compatibility for shared columns: **100%**.

### 4. Data Content Change

- Old `input_text` often contained task instructions + SMILES
- New `input_text` is typically plain SMILES only
- Task information is managed by `task_type`

Token-length trend:

| Split | Old Avg Length | New Avg Length | Change |
| ----- | -------------- | -------------- | ------ |
| train | 106 tokens     | 46 tokens      | -57%   |
| test  | 48 tokens      | 35 tokens      | -27%   |
| valid | 103 tokens     | 85 tokens      | -17%   |

Likely cause:

- Prompt template simplification
- Task instruction moved out of `input_text`

## HF Format Transition

### Old implementation (before ~2025-08)

```python
dataset = load_dataset("osunlp/SMolInstruct")
```

### New implementation (2025-11)

```python
def load_jsonl_dataset(dataset_path):
    # Load from raw/{train,dev,test}/*.jsonl
    # Build Dataset with explicit Features schema
```

Main changes:

1. Migration to JSONL-based ingestion
2. Explicit `Features` schema definition
3. Better use of task metadata fields

## Compatibility Assessment

### Breaking Changes

1. `task`-style access now needs `task_type`
2. Removed columns break code that directly depends on them
3. Prompt generation logic must be revisited if it assumed instruction-rich `input_text`

### Minor Changes

1. Sample count reduced by ~0.81%
2. New quality-related columns available (`input_too_long`, `valid_sample`)

### Compatible Area

Core model-training columns remain available and type-consistent.

## Recommended Actions

### Required short-term updates

```python
# old
task_name = dataset['task']

# new
task_name = dataset['task_type']
```

```python
# old
sample_id = dataset['sample_id']

# new
sample_idx = dataset['__index_level_0__']
```

If prompt text is needed, reconstruct it from `task_type` + SMILES instead of relying on `input_text` containing instructions.
