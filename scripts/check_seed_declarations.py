"""List what seed every pretrain config resolves to, and fail if one is off 42.

The directive of 2026-09-30 is that a config declares ``seed = 42`` and no
``data_seed``, and that variance runs use 1 and 17. On 2026-11-10 the whole tree
was brought to 42, so from then on the only configs allowed off 42 are the
variance arms, which carry ``_seed1`` / ``_seed17`` / ``_s1`` / ``_s17`` in their
name. Existing runs keep what they ran with: a seed is read back from the run's
``run_manifest.json``, not from the config tree.

Configs are read with ``ast`` rather than imported: importing one builds
tokenizers and touches the data tree. Star-imports are followed, so an arm that
inherits its seed is reported with the number it actually runs with.

    python scripts/check_seed_declarations.py [--modality rna] [--quiet]

Exit status is 1 when a config declares neither seed (the framework default would
decide and nobody chose it), and 1 when a config that is not a variance arm
resolves to anything but 42.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

CONFIGS = Path("molcrawl/tasks/pretrain/configs")
KEYS = ("seed", "data_seed")
TARGET = 42

#: The variance arms, the only configs allowed off 42. Their seed is the point of
#: the run, so it is in the file name as well.
VARIANCE = re.compile(r"_(s|seed)(1|17)$")


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

    missing, off_target, variance = [], [], []
    by_family = defaultdict(Counter)
    total = 0
    for cfg in sorted(CONFIGS.glob("*/*.py")):
        if cfg.name == "__init__.py":
            continue
        if args.modality and cfg.parent.name != args.modality:
            continue
        # Every config, not only the two paths that train: llama / roberta / esm2 /
        # chemberta2 / dnabert2 configs declare a seed too, and 42 is the tree's rule.
        arch = ("bert" if cfg.name.startswith("bert")
                else "gpt2" if cfg.name.startswith("gpt2") else "other")
        total += 1
        v = read(cfg)
        name = f"{cfg.parent.name}/{cfg.stem}"
        seed, data_seed = v.get("seed"), v.get("data_seed")
        by_family[(cfg.parent.name, arch)][seed] += 1
        if seed is None and data_seed is None:
            missing.append(name)
        elif seed != TARGET and VARIANCE.search(cfg.stem):
            variance.append((name, seed, data_seed))
        elif seed != TARGET:
            off_target.append((name, seed, data_seed))
        elif not args.quiet:
            print(f"  42      {name:52s} data_seed={data_seed}")

    if missing:
        print(f"\n  宣言なし ({len(missing)} 件) -- framework 既定が決めてしまう:")
        for name in missing:
            print(f"    {name}")
    if variance and not args.quiet:
        print(f"\n  変動測定の arm ({len(variance)} 件) -- 1 と 17 のままでよい:")
        for name, seed, data_seed in variance:
            print(f"    {name:52s} seed={seed} data_seed={data_seed}")
    if off_target:
        print(f"\n  42 でない ({len(off_target)} 件) -- 2026-11-10 以降は許されない:")
        for name, seed, data_seed in off_target:
            print(f"    {name:52s} seed={seed} data_seed={data_seed}")

    print("\n  モダリティ・arch ごとの seed の分布:")
    for key in sorted(by_family):
        spread = ", ".join(f"{s}×{n}" for s, n in sorted(by_family[key].items(),
                                                         key=lambda kv: (kv[0] is None, kv[0])))
        print(f"    {key[0]:20s} {key[1]:5s} {spread}")

    print(f"\nchecked {total} configs, {len(missing)} undeclared,"
          f" {len(variance)} variance arms, {len(off_target)} wrongly not 42")
    return 1 if (missing or off_target) else 0


if __name__ == "__main__":
    raise SystemExit(main())
