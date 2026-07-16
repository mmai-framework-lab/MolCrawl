"""Phase 0-5 patch: apply production-spec values to each (modality × arch × size) config.

Updates per production spec (2026-07-08):
- learning_rate: GPT-3 ladder placeholder (Phase 2 sweep will confirm actual value)
- max_iters (GPT-2) / max_steps (BERT): per-modality, per production-training-finalize table
- warmup: ~2% of max_iters/max_steps
- min_lr: learning_rate / 10
- weight_decay: GPT-2 keep 0.1, BERT set 0.01
- compounds seq_len: block_size / max_length → 128
- betas / grad_clip / max_grad_norm: rely on train.py/main.py defaults or add if missing
- rna GPT-2 vocab: already 25,426 via TranscriptomeTokenizer, verified separately

Compounds max_iters is DEFERRED to Phase 1-3 (needs ZINC20-added train row count).

Uses regex-based line replacement so that non-numeric surrounding code (comments,
docstrings, imports) is untouched. Idempotent: re-running produces no changes.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
CFG_ROOT = REPO / "molcrawl/tasks/pretrain/configs"

# --- Learning-rate ladder (placeholder — Phase 2 sweep confirms actual value) ---
LR_GPT2 = {"small": 6e-4, "medium": 3e-4, "large": 2.5e-4, "xl": 2e-4}
LR_BERT = {"small": 1e-4, "medium": 1e-4, "large": 1.5e-4, "xl": 1e-4}

# --- max_iters/max_steps per production-training-finalize table ---
MAX_ITERS = {
    "rna":               {"gpt2": 65821, "bert": 49366},
    "genome_sequence":   {"gpt2": 103548, "bert": 77661},
    "protein_sequence":  {"gpt2": 1754, "bert": 1316},
    "molecule_nat_lang": {"gpt2": 321, "bert": 321},
    # compounds: deferred to Phase 1-3
}

COMPOUNDS_SEQ_LEN = 128
WD_BERT = 0.01
WD_GPT2 = 0.1  # spec keep

CONFIGS = [
    # (modality, arch, size, file_path, size_subset_suffix_or_None)
    ("rna",               "gpt2", "small",  "rna/gpt2_small.py",  None),
    ("rna",               "gpt2", "medium", "rna/gpt2_medium.py", None),
    ("rna",               "gpt2", "large",  "rna/gpt2_large.py",  None),
    ("rna",               "gpt2", "xl",     "rna/gpt2_xl.py",     None),
    ("rna",               "bert", "small",  "rna/bert_small.py",  None),
    ("rna",               "bert", "medium", "rna/bert_medium.py", None),
    ("rna",               "bert", "large",  "rna/bert_large.py",  None),
    ("rna",               "bert", "xl",     "rna/bert_xl.py",     None),
    ("genome_sequence",   "bert", "small",  "genome_sequence/bert_small_subset.py",  "subset"),
    ("genome_sequence",   "bert", "xl",     "genome_sequence/bert_xl_subset.py",     "subset"),
    ("genome_sequence",   "gpt2", "small",  "genome_sequence/gpt2_small_subset.py",  "subset"),
    ("protein_sequence",  "gpt2", "small",  "protein_sequence/gpt2_small.py",  None),
    ("protein_sequence",  "gpt2", "medium", "protein_sequence/gpt2_medium.py", None),
    ("protein_sequence",  "gpt2", "large",  "protein_sequence/gpt2_large.py",  None),
    ("protein_sequence",  "gpt2", "xl",     "protein_sequence/gpt2_xl.py",     None),
    ("protein_sequence",  "bert", "small",  "protein_sequence/bert_small.py",  None),
    ("protein_sequence",  "bert", "medium", "protein_sequence/bert_medium.py", None),
    ("protein_sequence",  "bert", "large",  "protein_sequence/bert_large.py",  None),
    ("protein_sequence",  "bert", "xl",     "protein_sequence/bert_xl.py",     None),
    ("molecule_nat_lang", "gpt2", "small",  "molecule_nat_lang/gpt2_small.py",  None),
    ("molecule_nat_lang", "gpt2", "medium", "molecule_nat_lang/gpt2_medium.py", None),
    ("molecule_nat_lang", "gpt2", "large",  "molecule_nat_lang/gpt2_large.py",  None),
    ("molecule_nat_lang", "gpt2", "xl",     "molecule_nat_lang/gpt2_xl.py",     None),
    ("molecule_nat_lang", "bert", "small",  "molecule_nat_lang/bert_small.py",  None),
    ("molecule_nat_lang", "bert", "medium", "molecule_nat_lang/bert_medium.py", None),
    ("molecule_nat_lang", "bert", "large",  "molecule_nat_lang/bert_large.py",  None),
    ("molecule_nat_lang", "bert", "xl",     "molecule_nat_lang/bert_xl.py",     None),
    ("compounds",         "gpt2", "small",  "compounds/gpt2_small.py",  None),
    ("compounds",         "gpt2", "medium", "compounds/gpt2_medium.py", None),
    ("compounds",         "gpt2", "large",  "compounds/gpt2_large.py",  None),
    ("compounds",         "gpt2", "xl",     "compounds/gpt2_xl.py",     None),
    ("compounds",         "bert", "small",  "compounds/bert_small.py",  None),
    ("compounds",         "bert", "medium", "compounds/bert_medium.py", None),
    ("compounds",         "bert", "large",  "compounds/bert_large.py",  None),
    ("compounds",         "bert", "xl",     "compounds/bert_xl.py",     None),
]


def _fmt_lr(x: float) -> str:
    # Prefer scientific for tiny values
    if x < 1e-3:
        return f"{x:.4g}"
    return str(x)


def _replace_var(src: str, var: str, new_value_expr: str) -> tuple[str, bool]:
    """Replace top-level `var = <expr>` or `var: type = <expr>` in src.

    Returns (new_src, changed). Preserves the type annotation if present so
    that PEP 484-annotated configs (e.g. rna/bert_*.py) keep their static types.
    """
    # Group 1: leading whitespace. Group 2: optional ": <type>" (empty if none).
    # Match up to end of value expression; preserve trailing comment if present.
    pat = re.compile(
        rf"^([ \t]*){re.escape(var)}(\s*:\s*[A-Za-z_][A-Za-z_0-9.\[\], ]*)?\s*=\s*[^\n#]+(#[^\n]*)?",
        re.MULTILINE,
    )
    matches = list(pat.finditer(src))
    if not matches:
        return src, False
    m = matches[0]
    indent = m.group(1)
    annot = m.group(2) or ""
    trailing = m.group(3) or ""
    if trailing:
        trailing = "  " + trailing.strip()  # normalise to `  # comment`
    new_line = f"{indent}{var}{annot} = {new_value_expr}{trailing}"
    new_src = src[:m.start()] + new_line + src[m.end():]
    return new_src, True


def patch_config(modality: str, arch: str, size: str, rel_path: str, subset_suffix) -> dict:
    """Apply the phase 0-5 updates to one config file. Returns dict of change results."""
    path = CFG_ROOT / rel_path
    if not path.exists():
        return {"path": str(path), "error": "not_found"}

    src = path.read_text()
    original = src
    changes = []

    # 1. learning_rate
    lr_new = LR_GPT2[size] if arch == "gpt2" else LR_BERT[size]
    src, ok = _replace_var(src, "learning_rate", _fmt_lr(lr_new))
    if ok:
        changes.append(f"learning_rate = {_fmt_lr(lr_new)}")

    # 2. min_lr = learning_rate / 10
    min_lr = lr_new / 10
    src, ok = _replace_var(src, "min_lr", _fmt_lr(min_lr))
    if ok:
        changes.append(f"min_lr = {_fmt_lr(min_lr)}")

    # 3. max_iters / max_steps (skip compounds, deferred to Phase 1-3)
    if modality != "compounds":
        target = MAX_ITERS[modality][arch]
        if arch == "gpt2":
            src, ok = _replace_var(src, "max_iters", str(target))
            if ok:
                changes.append(f"max_iters = {target}")
            # lr_decay_iters = max_iters is a common nanoGPT idiom; keep aligned
            src, ok = _replace_var(src, "lr_decay_iters", str(target))
            if ok:
                changes.append(f"lr_decay_iters = {target}")
        else:
            src, ok = _replace_var(src, "max_steps", str(target))
            if ok:
                changes.append(f"max_steps = {target}")

    # 4. warmup ≈ 2% of max_iters
    if modality != "compounds":
        target = MAX_ITERS[modality][arch]
        warmup = max(1, int(target * 0.02))
        if arch == "gpt2":
            src, ok = _replace_var(src, "warmup_iters", str(warmup))
            if ok:
                changes.append(f"warmup_iters = {warmup}")
        else:
            src, ok = _replace_var(src, "warmup_steps", str(warmup))
            if ok:
                changes.append(f"warmup_steps = {warmup}")

    # 5. weight_decay
    wd = WD_BERT if arch == "bert" else WD_GPT2
    src, ok = _replace_var(src, "weight_decay", str(wd))
    if ok:
        changes.append(f"weight_decay = {wd}")

    # 6. compounds seq_len = 128
    if modality == "compounds":
        if arch == "gpt2":
            src, ok = _replace_var(src, "block_size", str(COMPOUNDS_SEQ_LEN))
            if ok:
                changes.append(f"block_size = {COMPOUNDS_SEQ_LEN}")
        else:
            src, ok = _replace_var(src, "max_length", str(COMPOUNDS_SEQ_LEN))
            if ok:
                changes.append(f"max_length = {COMPOUNDS_SEQ_LEN}")

    # 7. compounds GPT-2: set pad_token_id_for_loss = 0 to enable pad masking in
    #    train.py (Phase 0-1). Insert near existing dataset_dir / config globals.
    if modality == "compounds" and arch == "gpt2":
        if "pad_token_id_for_loss" not in src:
            # Insert after block_size line
            src = re.sub(
                r"(^[ \t]*block_size\s*=.*$)",
                r"\1\n\n# Enable pad-position CLM loss masking (Phase 0-1). compounds uses pad_id=0.\npad_token_id_for_loss = 0",
                src, count=1, flags=re.MULTILINE,
            )
            changes.append("+ pad_token_id_for_loss = 0")

    if src != original:
        path.write_text(src)
    return {"path": rel_path, "changes": changes, "changed": src != original}


def main() -> int:
    print("=== Phase 0-5 config patch ===\n")
    n_touched = 0
    n_no_change = 0
    for modality, arch, size, rel, subset in CONFIGS:
        r = patch_config(modality, arch, size, rel, subset)
        if r.get("error"):
            print(f"[MISS] {rel}: {r['error']}")
            continue
        if r["changed"]:
            n_touched += 1
            print(f"[EDIT] {rel}: {len(r['changes'])} field(s)")
            for c in r["changes"]:
                print(f"    - {c}")
        else:
            n_no_change += 1
            print(f"[OK]   {rel}: no changes")
    print(f"\n=== Summary: {n_touched} files edited, {n_no_change} unchanged ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
