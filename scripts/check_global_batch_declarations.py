"""Check that every pretrain config declares its global batch, and that it adds up.

The effective batch is a product of per-device factors, so a config can be wrong
in two ways: it can stay silent, in which case a wrong GPU allocation produces a
run that is not comparable with the rest of its ladder, or it can declare a number
that its own factors do not multiply to. genome GPT-2 trained at 640 while every
document said 2,560, which is the incident this guards against.

Configs are read with ``ast`` rather than imported: importing one builds
tokenizers and touches the data tree. Star-imports are followed, so an arm that
inherits its factors is judged on what it actually runs with.

    python scripts/check_global_batch_declarations.py [--world-size 4] [--quiet]

Exit status is 1 when a declaration contradicts its factors. A missing
declaration is reported but does not fail the run: adopting one is a decision
about the run, not a lint.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

CONFIGS = Path("molcrawl/tasks/pretrain/configs")
KEYS = ("batch_size", "gradient_accumulation_steps", "expected_global_batch")


def value_of(node, env):
    """A literal, an arithmetic expression over literals, or a name already set.

    Configs write "5 * 16" and "_GLOBAL_BATCH" as often as they write a number.
    Reading only bare literals silently skips exactly the configs most worth
    checking, which is how the genome ones escaped the first pass.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        left, right = value_of(node.left, env), value_of(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def read(path: Path, seen: set[Path] | None = None) -> dict:
    """{key: value} for one config, following its star-imports first."""
    seen = seen if seen is not None else set()
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict = {}
    env: dict = {}  # every module-level number, so a name reference resolves
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            parts = (node.module or "").split(".")
            if "configs" in parts:
                values.update(read(CONFIGS.joinpath(*parts[parts.index("configs") + 1:])
                                   .with_suffix(".py"), seen))
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        got = value_of(node.value, env) if targets else None
        for name in targets:
            if got is not None:
                env[name] = got
            if name in KEYS:
                values[name] = got
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=4,
                    help="GPUs a BERT run is placed on; its effective batch scales with this")
    ap.add_argument("--quiet", action="store_true", help="only print the problems")
    args = ap.parse_args()

    bad, silent, checked = [], [], 0
    for cfg in sorted(CONFIGS.glob("*/*.py")):
        if cfg.name == "__init__.py":
            continue
        arch = "bert" if cfg.name.startswith("bert") else "gpt2" if cfg.name.startswith("gpt2") else None
        if arch is None:
            continue
        v = read(cfg)
        b, g, want = v.get("batch_size"), v.get("gradient_accumulation_steps"), v.get("expected_global_batch")
        if b is None or g is None:
            continue
        # nanoGPT divides the accumulation by the world size before training, so a
        # GPT-2 product is GPU-count independent; the HF side is not.
        effective = b * g if arch == "gpt2" else b * g * args.world_size
        name = f"{cfg.parent.name}/{cfg.stem}"
        if want is None:
            silent.append((name, effective))
            continue
        checked += 1
        if want != effective:
            bad.append((name, want, effective))
        elif not args.quiet:
            print(f"  ok      {name:52s} {want}")

    for name, want, eff in bad:
        print(f"  MISMATCH {name:52s} declares {want}, factors give {eff}")
    if silent and not args.quiet:
        print(f"\n  宣言なし ({len(silent)} 件) -- 実効値は factors から:")
        for name, eff in silent:
            print(f"    {name:52s} {eff}")
    print(f"\nchecked {checked} declarations, {len(bad)} mismatched, {len(silent)} undeclared"
          f" (BERT は world_size={args.world_size} を仮定)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
