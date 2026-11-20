"""The molecule_nat_lang batch-size grid: the rules it is derived from, and its configs.

Two things are pinned here.

The derivation. ``scripts/molnl_batch_sweep_arms.py`` turns the 2026-11-20 directive's
rules into 68 arms and compares the result against the directive's own printed tables;
this asserts that comparison comes out clean, and adds the invariants the directive
states in words rather than in a table -- that one GPU's block count can build the
grid's smallest batch, that the three numbers multiply out to it, and that the
intervals fall on whole steps at every batch.

The configs. They are generated, so what is worth testing is not their text but that
each one still carries its own row of the inventory. The reading is by ast, without
importing the repository, so this runs anywhere; the launcher-level check that executes
each config lives in scripts/molnl_batch_sweep_configs.py --check and runs on a compute
node, where the corpus and the tokenizer are reachable.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "molcrawl" / "tasks" / "pretrain" / "configs" / "molecule_nat_lang"


def _load(name):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms_module():
    return _load("molnl_batch_sweep_arms")


@pytest.fixture(scope="module")
def rows(arms_module):
    return arms_module.arms("current")


def test_sixty_eight_arms(arms_module, rows):
    assert len(rows) == sum(arms_module.EXPECTED_ARMS.values()) == 68
    for arch, want in arms_module.EXPECTED_ARMS.items():
        assert sum(1 for r in rows if r["arch"] == arch) == want


def test_derivation_agrees_with_the_directive(arms_module, rows, capsys):
    """§1-§4's printed tables against the rules they follow from."""
    problems = arms_module.check(rows, "current")
    capsys.readouterr()
    assert problems == 0


def test_the_three_numbers_multiply_out_to_the_batch(rows):
    for row in rows:
        micro = row["micro_batch"]
        accum = row["grad_accum"]
        world = row["world_size"]
        assert micro * accum * world == row["batch"] == row["expected_global_batch"], row


def test_one_gpu_can_build_the_smallest_batch(arms_module, rows):
    """§4: one GPU's block count times 4 GPUs must divide 256."""
    for row in rows:
        assert row["micro_batch"] in arms_module.MICRO_CANDIDATES, row
        assert 256 % (row["micro_batch"] * 4) == 0, row


def test_one_gpu_block_count_is_held_across_the_batches(rows):
    """§4: it is chosen per size and does not move between the five batches."""
    per_size = {}
    for row in rows:
        per_size.setdefault((row["arch"], row["size"]), set()).add(row["micro_batch"])
    for key, values in per_size.items():
        assert len(values) == 1, (key, values)


def test_the_budget_is_the_same_for_every_arm_of_a_framework(arms_module, rows):
    for arch, blocks in arms_module.BUDGET_BLOCKS.items():
        sel = [r for r in rows if r["arch"] == arch]
        assert {r["blocks"] for r in sel} == {blocks}
        assert {r["tokens"] for r in sel} == {blocks * arms_module.SEQUENCE_LENGTH}
        # §2: the step count is the budget divided by the batch, exactly.
        for row in sel:
            assert row["steps"] * row["batch"] == blocks, row


def test_intervals_land_on_whole_steps_and_on_the_same_blocks(arms_module, rows):
    for row in rows:
        every = arms_module.EVAL_EVERY_BLOCKS[row["arch"]]
        assert row["eval_interval"] * row["batch"] == every, row
        assert row["save_interval"] % row["eval_interval"] == 0, row
        assert row["steps"] % row["eval_interval"] == 0, row
        assert row["eval_points"] * every == row["blocks"], row


def test_warmup_is_the_frameworks_own_fraction(arms_module, rows):
    for row in rows:
        fraction = arms_module.WARMUP_FRACTION[row["arch"]]
        assert row["warmup_steps"] == int(fraction * row["steps"]), row
        # A warmup as long as the run would never reach the peak.
        assert row["warmup_steps"] < row["steps"], row


def test_gpt2_floor_and_decay_length(arms_module, rows):
    for row in rows:
        if row["arch"] != "gpt2":
            assert row["min_lr"] is None and row["lr_decay_iters"] is None, row
            continue
        assert row["lr_decay_iters"] == row["steps"], row
        ratio = float(row["lr"]) / float(row["min_lr"])
        assert abs(ratio - arms_module.GPT2_MIN_LR_DIVISOR) < 0.02, row


def test_every_arm_has_its_own_config_and_directory(rows):
    assert len({r["config"] for r in rows}) == len(rows)
    assert len({r["out_dir"] for r in rows}) == len(rows)
    for row in rows:
        stem = Path(row["config"]).stem
        assert stem == f"{row['arch']}_{row['size']}_b{row['batch']}_lr{row['lr_slug']}"
        # bert-grid.sbatch refuses a model_path whose basename is not named for a rate,
        # which is what stops two arms of a size sharing one directory.
        assert "-lr" in row["out_dir"]


def test_bert_evaluation_is_held_across_the_grid(arms_module, rows):
    bert = [r for r in rows if r["arch"] == "bert"]
    assert {r["eval_batch"] for r in bert} == {arms_module.BERT_EVAL_BATCH}
    assert {r["eval_blocks"] for r in bert} == {arms_module.BERT_EVAL_BLOCKS}
    assert {r["world_size"] for r in bert} == {4}
    assert {r["judge_metric"] for r in bert} == {"eval_loss_mask"}
    assert {r["judge_metric"] for r in rows if r["arch"] == "gpt2"} == {"val_loss"}


# ---------------------------------------------------------------------------
# The 68 generated configs
# ---------------------------------------------------------------------------

def _declared(path):
    """Top-level literal assignments, by ast, without importing anything."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            out[target.id] = None  # a call or a name: read from the text instead
    return out


def _config_paths(rows):
    return [(row, CONFIG_DIR / Path(row["config"]).name) for row in rows]


def test_every_arm_has_its_config_on_disk(rows):
    missing = [row["config"] for row, path in _config_paths(rows) if not path.exists()]
    assert not missing, f"{len(missing)} of {len(rows)} configs are not written: {missing[:5]}"


def test_configs_carry_their_inventory_row(rows):
    keys = {
        "bert": [("steps", "max_steps"), ("warmup_steps", "warmup_steps"),
                 ("eval_interval", "log_interval"), ("save_interval", "save_steps"),
                 ("micro_batch", "batch_size"),
                 ("grad_accum", "gradient_accumulation_steps"),
                 ("expected_global_batch", "expected_global_batch"),
                 ("eval_batch", "per_device_eval_batch_size")],
        "gpt2": [("steps", "max_iters"), ("warmup_steps", "warmup_iters"),
                 ("eval_interval", "eval_interval"),
                 ("save_interval", "save_checkpoint_steps"),
                 ("lr_decay_iters", "lr_decay_iters"),
                 ("micro_batch", "batch_size"),
                 ("grad_accum", "gradient_accumulation_steps"),
                 ("expected_global_batch", "expected_global_batch")],
    }
    for row, path in _config_paths(rows):
        if not path.exists():
            pytest.skip("configs are not written yet")
        values = _declared(path)
        for column, key in keys[row["arch"]]:
            assert values.get(key) == int(row[column]), (path.name, key)
        assert values.get("learning_rate") == pytest.approx(float(row["lr"]), rel=1e-12), path.name
        assert values.get("seed") == 42, path.name
        assert values.get("early_stopping") is False, path.name
        if row["arch"] == "gpt2":
            assert values.get("min_lr") == pytest.approx(float(row["min_lr"]), rel=1e-12), path.name
            assert values.get("init_from") == "scratch", path.name
            assert values.get("deterministic_val_eval") is True, path.name
        else:
            assert values.get("fixed_eval_mask") is True, path.name
            assert values.get("eval_mask_seed") == 42, path.name
            for key in ("judge_on", "metric_for_best_model", "save_on_improve_metric",
                        "checkpoint_metric"):
                assert values.get(key) == "eval_loss_mask", (path.name, key)
            assert values.get("save_on_improve") is True, path.name


def test_configs_do_not_declare_an_eval_subset(rows):
    """molnl scores its whole 1,631-block validation split; see §6 in the generator."""
    for row, path in _config_paths(rows):
        if not path.exists() or row["arch"] != "bert":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not line.startswith(("eval_subset_random", "eval_subset_seed")), path.name


def test_configs_name_their_own_output_directory(rows):
    for row, path in _config_paths(rows):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        suffix = f'-b{row["batch"]}-lr{row["lr_slug"]}"'
        assert suffix in text, (path.name, suffix)
        assert "get_bert_output_path(" in text or "get_gpt2_output_path(" in text


def test_configs_inherit_their_size_base(rows):
    """Not a style point: the base carries preprocess_function and the vocab guard."""
    for row, path in _config_paths(rows):
        if not path.exists():
            continue
        base = f"{'bert' if row['arch'] == 'bert' else 'gpt2'}_{row['size']}"
        want = (f"from molcrawl.tasks.pretrain.configs.molecule_nat_lang.{base} "
                f"import *")
        assert want in path.read_text(encoding="utf-8"), (path.name, want)
