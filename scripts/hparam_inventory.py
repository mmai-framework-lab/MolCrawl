"""Read the training hyperparameters out of the pretrain configs, as a table.

The configs are Python modules that import paths and tokenizers, so importing one
builds tokenizers and touches the data tree. This reads them with ``ast`` instead:
no import, no side effect, and a config whose value is a call (an output path, a
tokenizer) is simply left out -- those are operational values, not hyperparameters.

A grid arm is three lines on top of a base config (``from ...bert_small import *``),
so the star-imports are followed first and the arm's own assignments win.

    python scripts/hparam_inventory.py [--out <file.tsv>] [--arch bert|gpt2]

Columns are the union of the keys found, in the order given by KEYS.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

CONFIGS = Path("molcrawl/tasks/pretrain/configs")

# The narrow sense of "hyperparameter": what the optimisation does, not where it
# writes. Order is the order of the columns.
KEYS = [
    # shape
    "model_size", "n_layer", "n_head", "n_embd", "block_size", "max_length",
    # batch
    "batch_size", "gradient_accumulation_steps", "expected_global_batch",
    # schedule
    "max_iters", "max_steps", "num_train_epochs", "warmup_iters", "warmup_steps",
    "lr_decay_iters", "learning_rate", "min_lr", "decay_lr",
    # regularisation and the optimiser
    "weight_decay", "beta1", "beta2", "grad_clip", "max_grad_norm", "dropout", "bias",
    # objective
    "mlm_probability", "document_masking", "mask_replace_prob", "random_replace_prob",
    # numerics and reproducibility
    "dtype", "bf16", "fp16", "tf32", "seed",
]

# BertConfig's own defaults, which molcrawl/models/bert/main.py leaves alone for
# "small" and overrides for the larger sizes (main.py:385-417).
BERT_SHAPES = {
    "small": (12, 12, 768, 3072),
    "medium": (24, 16, 1024, 4096),
    "large": (36, 18, 1152, 4608),
    "xl": (48, 25, 1600, 6400),
}


def env_default(node):
    """("VAR", value) for float(os.environ.get("VAR", "3e-5")), else None.

    Two BERT configs take their learning rate from the environment with a literal
    fallback. Reading them as "no value" would leave a hole in the inventory where
    a rate that was actually used belongs.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("float", "int") and len(node.args) == 1):
        return None
    inner = node.args[0]
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "get" and len(inner.args) == 2):
        return None
    try:
        name, fallback = ast.literal_eval(inner.args[0]), ast.literal_eval(inner.args[1])
        return str(name), (float(fallback) if node.func.id == "float" else int(fallback))
    except (ValueError, TypeError, SyntaxError):
        return None


def literal(node):
    """The value of an assignment, or None when it is not a plain literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = literal(node.operand)
            return -inner if isinstance(inner, (int, float)) else None
        return None


def read(path: Path, seen: set[Path] | None = None) -> dict:
    """Resolve one config to {key: value}, following its star-imports first."""
    seen = seen if seen is not None else set()
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            # "from molcrawl.tasks.pretrain.configs.rna.bert_small import *"
            parts = (node.module or "").split(".")
            if "configs" in parts:
                base = CONFIGS.joinpath(*parts[parts.index("configs") + 1:]).with_suffix(".py")
                values.update(read(base, seen))

    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if not targets or getattr(node, "value", None) is None:
            continue
        value = literal(node.value)
        from_env = None
        if value is None:
            got = env_default(node.value)
            if got is None:
                continue
            from_env, value = got
        for name in targets:
            if name in KEYS:
                values[name] = value
                if from_env:
                    envs = values.get("_env", {})
                    values["_env"] = {**envs, name: from_env}
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="-", help="TSV destination, or - for stdout")
    ap.add_argument("--arch", choices=["bert", "gpt2"], help="only this architecture")
    args = ap.parse_args()

    rows = []
    for cfg in sorted(CONFIGS.glob("*/*.py")):
        if cfg.name == "__init__.py":
            continue
        arch = "bert" if cfg.name.startswith("bert") else "gpt2" if cfg.name.startswith("gpt2") else None
        if arch is None or (args.arch and arch != args.arch):
            continue
        values = read(cfg)
        if not values:
            continue
        if arch == "bert":
            shape = BERT_SHAPES.get(str(values.get("model_size", "")))
            if shape:
                # The size name is the only shape a BERT config carries; the
                # dimensions live in main.py, so fill them in here or the column
                # is empty for every BERT row.
                for key, got in zip(("n_layer", "n_head", "n_embd"), shape[:3]):
                    values.setdefault(key, got)
        envs = values.pop("_env", {})
        rows.append({"modality": cfg.parent.name, "arch": arch, "config": cfg.stem, **values,
                     "env_overridable": ", ".join(f"{k}={v}" for k, v in sorted(envs.items()))})

    columns = (["modality", "arch", "config"] + [k for k in KEYS if any(k in r for r in rows)]
               + (["env_overridable"] if any(r.get("env_overridable") for r in rows) else []))
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    with out as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
    print(f"{len(rows)} configs, {len(columns)} columns", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
