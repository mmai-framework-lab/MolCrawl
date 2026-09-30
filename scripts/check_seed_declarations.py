"""List what seed every pretrain config resolves to, and flag what needs a look.

The directive of 2026-09-30 is that a new config declares ``seed = 42`` and
``data_seed = 42``, that existing runs keep what they ran with, that an arm added
to a grid takes that grid's seed, and that variance runs use 1 and 17. Those rules
need the current values in front of you, which is what this prints.

Configs are read with ``ast`` rather than imported: importing one builds
tokenizers and touches the data tree. Star-imports are followed, so an arm that
inherits its seed is reported with the number it actually runs with.

    python scripts/check_seed_declarations.py [--modality rna] [--quiet]

Exit status is 1 when a config declares neither seed, because then the framework
default decides and nobody chose it. A value other than 42 is reported, not
failed: existing grids are supposed to keep their own.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path

CONFIGS = Path("molcrawl/tasks/pretrain/configs")
KEYS = ("seed", "data_seed")
TARGET = 42


def value_of(node, env):
    """A literal, arithmetic over literals, or a name already bound in the file."""
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
    return None


def read(path: Path, seen: set[Path] | None = None) -> dict:
    seen = seen if seen is not None else set()
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict = {}
    env: dict = {}
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
        got = value_of(node.value, env) if targets and node.value is not None else None
        for name in targets:
            if got is not None:
                env[name] = got
            if name in KEYS:
                values[name] = got
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", help="only this directory under configs/")
    ap.add_argument("--quiet", action="store_true", help="only the summary and the problems")
    args = ap.parse_args()

    missing, off_target = [], []
    by_family = defaultdict(Counter)
    total = 0
    for cfg in sorted(CONFIGS.glob("*/*.py")):
        if cfg.name == "__init__.py":
            continue
        if args.modality and cfg.parent.name != args.modality:
            continue
        arch = "bert" if cfg.name.startswith("bert") else "gpt2" if cfg.name.startswith("gpt2") else None
        if arch is None:
            continue
        total += 1
        v = read(cfg)
        name = f"{cfg.parent.name}/{cfg.stem}"
        seed, data_seed = v.get("seed"), v.get("data_seed")
        by_family[(cfg.parent.name, arch)][seed] += 1
        if seed is None and data_seed is None:
            missing.append(name)
        elif seed != TARGET:
            off_target.append((name, seed, data_seed))
        elif not args.quiet:
            print(f"  42      {name:52s} data_seed={data_seed}")

    if missing:
        print(f"\n  宣言なし ({len(missing)} 件) -- framework 既定が決めてしまう:")
        for name in missing:
            print(f"    {name}")
    if off_target and not args.quiet:
        print(f"\n  42 以外 ({len(off_target)} 件) -- 既存の格子はこのままでよい:")
        for name, seed, data_seed in off_target:
            print(f"    {name:52s} seed={seed} data_seed={data_seed}")

    print("\n  モダリティ・arch ごとの seed の分布:")
    for key in sorted(by_family):
        spread = ", ".join(f"{s}×{n}" for s, n in sorted(by_family[key].items(),
                                                         key=lambda kv: (kv[0] is None, kv[0])))
        print(f"    {key[0]:20s} {key[1]:5s} {spread}")

    print(f"\nchecked {total} configs, {len(missing)} undeclared, {len(off_target)} not 42")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
