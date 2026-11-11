"""Every pretrain config declares seed 42, except the variance arms.

The rule is the 2026-09-30 directive; on 2026-11-10 it was applied to the whole
tree, which is what makes a test possible: before that, grids kept their own seed
and the only way to tell right from wrong was to know the grid. Now the tree has
one value and 32 named exceptions, so a config that drifts off 42 -- a copy of an
old file, a hand-edited number -- fails here instead of being found in a
``run_manifest.json`` after the run.

The checker script is the same code path, so this test and
``sbatch workflows/check-seed-declarations.sbatch`` cannot disagree.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from check_seed_declarations import CONFIGS, TARGET, VARIANCE, read  # noqa: E402


def configs():
    return [p for p in sorted(CONFIGS.glob("*/*.py")) if p.name != "__init__.py"]


def test_the_config_tree_is_not_empty():
    assert len(configs()) > 300


def test_every_config_resolves_to_42_or_is_a_named_variance_arm():
    off = {}
    for cfg in configs():
        seed = read(cfg).get("seed")
        if seed == TARGET:
            continue
        if VARIANCE.search(cfg.stem) and seed in (1, 17):
            continue
        off[f"{cfg.parent.name}/{cfg.stem}"] = seed
    assert off == {}, f"configs off 42: {off}"


def test_the_variance_arms_are_still_1_and_17():
    arms = {f"{c.parent.name}/{c.stem}": read(c).get("seed")
            for c in configs() if VARIANCE.search(c.stem)}
    assert len(arms) == 32, arms
    assert sorted(set(arms.values())) == [1, 17], arms
    for name, seed in arms.items():
        assert seed == (1 if name.endswith(("_s1", "_seed1")) else 17), (name, seed)


def test_no_config_declares_a_data_seed_other_than_seed():
    """main.py refuses such a config at startup; nothing in the tree should carry one."""
    bad = {}
    for cfg in configs():
        v = read(cfg)
        if v.get("data_seed") is not None and v["data_seed"] != v.get("seed"):
            bad[f"{cfg.parent.name}/{cfg.stem}"] = (v.get("seed"), v["data_seed"])
    assert bad == {}, bad
