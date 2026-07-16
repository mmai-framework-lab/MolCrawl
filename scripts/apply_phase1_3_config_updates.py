"""Phase 1-3 patch: apply per-variant max_iters / warmup / dataset_dir to compounds configs.

Production spec (2026-07-09) confirmed by upstream after Phase 1-2 completion:
- max_iters = 3 × (train rows) / (8 × 80 × 4) = 3 × rows / 2560
  - BERT (train=10,591,633):  12,412
  - GPT-2 (train=10,593,826): 12,415
- warmup ≈ 2%:
  - BERT / GPT-2: 249 both
- lr_decay_iters (GPT-2): equal to max_iters
- min_lr = learning_rate / 10 (Phase 0-5 で確定済、 変更なし)
- dataset_dir:
  - BERT: COMPOUNDS_DATASET_DIR_BERT
  - GPT-2: COMPOUNDS_DATASET_DIR_GPT2
- Overwrite the Phase 0-5 placeholder values (30,000 / 6,000).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
CFG_DIR = REPO / "molcrawl/tasks/pretrain/configs/compounds"

BERT_MAX_ITERS = 12_412
GPT2_MAX_ITERS = 12_415
WARMUP = 249


def _replace_var(src: str, var: str, new_value_expr: str) -> tuple[str, bool]:
    pat = re.compile(
        rf"^([ \t]*){re.escape(var)}(\s*:\s*[A-Za-z_][A-Za-z_0-9.\[\], ]*)?\s*=\s*[^\n#]+(#[^\n]*)?",
        re.MULTILINE,
    )
    m = pat.search(src)
    if not m:
        return src, False
    indent = m.group(1)
    annot = m.group(2) or ""
    trailing = m.group(3) or ""
    if trailing:
        trailing = "  " + trailing.strip()
    new_line = f"{indent}{var}{annot} = {new_value_expr}{trailing}"
    return src[:m.start()] + new_line + src[m.end():], True


def patch_bert(name: str) -> dict:
    path = CFG_DIR / f"bert_{name}.py"
    src = path.read_text()
    orig = src
    changes = []
    for var, val in [
        ("max_steps", BERT_MAX_ITERS),
        ("warmup_steps", WARMUP),
        ("dataset_dir", "COMPOUNDS_DATASET_DIR_BERT"),
    ]:
        src, ok = _replace_var(src, var, str(val))
        if ok: changes.append(f"{var} = {val}")

    # Update the import to include the new BERT-specific constant.
    if "COMPOUNDS_DATASET_DIR_BERT" not in orig:
        src = re.sub(
            r"(from molcrawl\.core\.paths import[^\n]*?)(COMPOUNDS_DATASET_DIR)([^_])",
            r"\1COMPOUNDS_DATASET_DIR_BERT\3",
            src, count=1,
        )
        if "COMPOUNDS_DATASET_DIR_BERT" in src:
            changes.append("import: COMPOUNDS_DATASET_DIR → COMPOUNDS_DATASET_DIR_BERT")

    if src != orig:
        path.write_text(src)
    return {"path": path.name, "changes": changes, "changed": src != orig}


def patch_gpt2(name: str) -> dict:
    path = CFG_DIR / f"gpt2_{name}.py"
    src = path.read_text()
    orig = src
    changes = []
    for var, val in [
        ("max_iters", GPT2_MAX_ITERS),
        ("lr_decay_iters", GPT2_MAX_ITERS),
        ("warmup_iters", WARMUP),
        ("dataset_dir", "COMPOUNDS_DATASET_DIR_GPT2"),
    ]:
        src, ok = _replace_var(src, var, str(val))
        if ok: changes.append(f"{var} = {val}")

    if "COMPOUNDS_DATASET_DIR_GPT2" not in orig:
        src = re.sub(
            r"(from molcrawl\.core\.paths import[^\n]*?)(COMPOUNDS_DATASET_DIR)([^_])",
            r"\1COMPOUNDS_DATASET_DIR_GPT2\3",
            src, count=1,
        )
        if "COMPOUNDS_DATASET_DIR_GPT2" in src:
            changes.append("import: COMPOUNDS_DATASET_DIR → COMPOUNDS_DATASET_DIR_GPT2")

    if src != orig:
        path.write_text(src)
    return {"path": path.name, "changes": changes, "changed": src != orig}


def main() -> int:
    print("=== Phase 1-3 config patch ===")
    for name in ("small", "medium", "large"):
        r = patch_bert(name)
        print(f"[BERT {name}] {'edited' if r['changed'] else 'no change'}")
        for c in r["changes"]:
            print(f"    - {c}")
    for name in ("small", "medium", "large", "xl"):
        r = patch_gpt2(name)
        print(f"[GPT-2 {name}] {'edited' if r['changed'] else 'no change'}")
        for c in r["changes"]:
            print(f"    - {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
