"""The compounds batch-size grid: the arms it derives and the 56 configs it writes.

The grid's whole point is that four batches cover the same data and are compared at
the same column, so what is tested here is the arithmetic that makes that true --
the budget, the interval anchoring, the warmup fraction, the accumulation -- and then
that each written config actually carries the numbers the inventory says it does.

The directive's own printed tables are the expectations inside
compounds_batch_sweep_arms.py; one case here asserts that the derivation disagrees
with none of them, so a change to either side shows up as a failure rather than as a
quiet difference between the report and the configs.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


def load(name: str):
    path = repo_root() / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms_mod():
    return load("compounds_batch_sweep_arms")


@pytest.fixture(scope="module")
def cfg_mod():
    return load("compounds_batch_sweep_configs")


@pytest.fixture(scope="module")
def rows(arms_mod):
    return arms_mod.arms("order")


# --- the grid's shape (§1) -------------------------------------------------------

def test_fifty_six_arms(rows):
    assert len(rows) == 56


def test_forty_bert_and_sixteen_gpt2(rows, arms_mod):
    for arch, want in arms_mod.EXPECTED_ARMS.items():
        assert sum(1 for r in rows if r["arch"] == arch) == want


def test_every_size_and_batch_is_present(rows, arms_mod):
    seen = {(r["arch"], r["size"], r["batch"]) for r in rows}
    assert len(seen) == 2 * len(arms_mod.SIZES) * len(arms_mod.BATCHES)


def test_config_names_are_unique(rows):
    names = [r["config"] for r in rows]
    assert len(set(names)) == len(names)


def test_output_directories_are_unique(rows):
    names = [r["out_dir"] for r in rows]
    assert len(set(names)) == len(names)


def test_output_directory_carries_lr_so_bert_grid_accepts_it(rows):
    # workflows/bert-grid.sbatch refuses a model_path whose basename has no -lr.
    assert all("-lr" in r["out_dir"] for r in rows if r["arch"] == "bert")


# --- the directive's own tables --------------------------------------------------

def test_derivation_matches_every_printed_table(rows, arms_mod, capsys):
    problems = arms_mod.check(rows, "order")
    capsys.readouterr()
    assert problems == 0


# --- the budget (§2) -------------------------------------------------------------

def test_steps_times_batch_is_the_budget(rows, arms_mod):
    for row in rows:
        assert row["steps"] * row["batch"] == arms_mod.BUDGET_ROWS[row["arch"]]


def test_tokens_are_columns_times_the_sequence_length(rows, arms_mod):
    for row in rows:
        assert row["tokens"] == row["rows"] * arms_mod.SEQUENCE_LENGTH


def test_evaluation_points_land_at_the_same_column(rows):
    for arch in ("bert", "gpt2"):
        columns = {r["eval_interval"] * r["batch"]
                   for r in rows if r["arch"] == arch}
        assert len(columns) == 1


def test_save_points_land_at_the_same_column(rows):
    columns = {r["save_interval"] * r["batch"] for r in rows if r["arch"] == "bert"}
    assert len(columns) == 1


def test_bert_save_interval_is_a_multiple_of_the_eval_interval(rows):
    # main.py warns otherwise: a checkpoint that misses an eval cannot be ranked.
    for row in rows:
        if row["arch"] == "bert":
            assert row["save_interval"] % row["eval_interval"] == 0


def test_every_arm_has_the_same_number_of_eval_points(rows):
    for arch in ("bert", "gpt2"):
        counts = {r["steps"] // r["eval_interval"] for r in rows if r["arch"] == arch}
        assert len(counts) == 1


# --- the rates (§3) --------------------------------------------------------------

def test_rate_scales_as_the_square_root_of_the_batch(arms_mod):
    for size in arms_mod.SIZES:
        base = arms_mod.BASE_LR_2560["bert"][size]
        for batch in arms_mod.BATCHES:
            got = arms_mod.scaled_rate("bert", size, batch, 1.0)
            want = arms_mod.three_sig(base * (batch / 2560) ** 0.5)
            assert got == want


def test_bert_runs_two_rates_below_2048_and_three_at_or_above(rows):
    for size in ("small", "medium", "large", "xl"):
        for batch, want in ((512, 2), (1024, 2), (2048, 3), (4096, 3)):
            sel = [r for r in rows if r["arch"] == "bert" and r["size"] == size
                   and r["batch"] == batch]
            assert len(sel) == want


def test_gpt2_runs_one_rate_per_batch(rows):
    for size in ("small", "medium", "large", "xl"):
        for batch in (512, 1024, 2048, 4096):
            sel = [r for r in rows if r["arch"] == "gpt2" and r["size"] == size
                   and r["batch"] == batch]
            assert len(sel) == 1


def test_lr_slug_is_the_existing_naming(arms_mod):
    assert arms_mod.lr_slug(4.47e-4) == "4p47e4"
    assert arms_mod.lr_slug(1.25e-3) == "1p25e3"
    assert arms_mod.lr_slug(1e-3) == "1e3"


# --- the schedule (§4) -----------------------------------------------------------

def test_warmup_is_the_frameworks_fraction_truncated(rows, arms_mod):
    for row in rows:
        fraction = arms_mod.WARMUP_FRACTION[row["arch"]]
        assert row["warmup_steps"] == int(fraction * row["steps"])


def test_warmup_is_a_whole_number_of_steps(rows):
    assert all(isinstance(r["warmup_steps"], int) for r in rows)


def test_warmup_is_shorter_than_the_run(rows):
    assert all(r["warmup_steps"] < r["steps"] for r in rows)


def test_gpt2_decay_runs_the_length_of_its_own_arm(rows):
    for row in rows:
        if row["arch"] == "gpt2":
            assert row["lr_decay_iters"] == row["steps"]


def test_gpt2_floor_is_a_tenth_of_its_own_peak(rows, arms_mod):
    for row in rows:
        if row["arch"] == "gpt2":
            want = arms_mod.three_sig(float(row["lr"]) / 10)
            assert float(row["min_lr"]) == pytest.approx(want, rel=1e-9)


def test_bert_carries_no_gpt2_only_keys(rows):
    for row in rows:
        if row["arch"] == "bert":
            assert row["min_lr"] is None and row["lr_decay_iters"] is None


# --- the global batch (§5) and the accumulation (§8) -----------------------------

def test_declared_global_batch_is_the_arms_batch(rows):
    assert all(r["expected_global_batch"] == r["batch"] for r in rows)


def test_micro_times_accumulation_times_world_is_the_batch(rows):
    for row in rows:
        assert row["micro_batch"] * row["grad_accum"] * row["world_size"] \
            == row["batch"]


def test_bert_runs_on_four_gpus_and_gpt2_has_no_world_factor(rows):
    for row in rows:
        assert row["world_size"] == (4 if row["arch"] == "bert" else 1)


def test_one_gpus_count_is_held_across_the_four_batches(rows):
    for arch in ("bert", "gpt2"):
        for size in ("small", "medium", "large", "xl"):
            micro = {r["micro_batch"] for r in rows
                     if r["arch"] == arch and r["size"] == size}
            assert len(micro) == 1


def test_both_micro_batch_sets_divide_every_batch(arms_mod):
    for which in arms_mod.MICRO_BATCH:
        for arch in ("bert", "gpt2"):
            for size in arms_mod.SIZES:
                for batch in arms_mod.BATCHES:
                    # Raises SystemExit when the shape does not divide the batch.
                    arms_mod.accumulation(arch, size, batch, which)


def test_a_shape_that_does_not_divide_the_batch_is_refused(arms_mod):
    # medium's current 160 x 4 GPUs = 640, which divides none of the four batches.
    arms_mod.MICRO_BATCH["_probe"] = {"bert": {"medium": 160}, "gpt2": {}}
    try:
        with pytest.raises(SystemExit):
            arms_mod.accumulation("bert", "medium", 512, "_probe")
    finally:
        del arms_mod.MICRO_BATCH["_probe"]


# --- the evaluation (§6) and the judged metric (§7) ------------------------------

def test_all_bert_arms_score_at_one_evaluation_micro_batch(rows, arms_mod):
    values = {r["eval_batch"] for r in rows if r["arch"] == "bert"}
    assert values == {arms_mod.BERT_EVAL_BATCH}


def test_the_evaluation_micro_batch_does_not_follow_the_training_batch(rows):
    for row in rows:
        if row["arch"] == "bert":
            assert row["eval_batch"] == 32  # whatever the training batch is


def test_judged_metric_is_eval_loss_for_bert_and_val_loss_for_gpt2(rows):
    for row in rows:
        assert row["judge_metric"] == ("eval_loss" if row["arch"] == "bert"
                                       else "val_loss")


# --- the configs ----------------------------------------------------------------

@pytest.fixture(scope="module")
def written(rows, cfg_mod, tmp_path_factory):
    out = tmp_path_factory.mktemp("configs")
    as_strings = [{k: ("" if v is None else str(v)) for k, v in r.items()}
                  for r in rows]
    cfg_mod.write_all(as_strings, str(out))
    return out, as_strings


def test_one_config_per_arm_is_written(written, rows):
    out, _ = written
    assert len(list(out.glob("*.py"))) == len(rows)


def test_every_written_config_parses(written):
    out, _ = written
    for path in out.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))


def test_written_configs_agree_with_the_inventory(written, cfg_mod, capsys):
    out, as_strings = written
    problems = cfg_mod.check_all(as_strings, str(out))
    capsys.readouterr()
    assert problems == 0


def test_a_tampered_config_is_caught(written, cfg_mod, capsys):
    out, as_strings = written
    row = next(r for r in as_strings if r["arch"] == "bert")
    path = out / Path(row["config"]).name
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(original.replace(f"max_steps = {row['steps']}",
                                         "max_steps = 1"), encoding="utf-8")
        problems = cfg_mod.check_all(as_strings, str(out))
        capsys.readouterr()
        assert problems >= 1
    finally:
        path.write_text(original, encoding="utf-8")


def test_no_config_declares_the_evaluation_subset_seed(written):
    # The rows come from one constant in main.py; a config that named it is how two
    # arms would stop sharing the subset. tests/unit/test_bert_eval_subset.py refuses
    # any config in the tree that does, so the generator must not write one.
    out, _ = written
    for path in out.glob("bert_*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(line.startswith("eval_subset_seed")
                       for line in text.splitlines()), path.name


def test_configs_carry_no_absolute_path_or_account(written):
    out, _ = written
    for path in out.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "/data1/" not in text
        assert "/home/" not in text


def test_configs_are_lf_only(written):
    out, _ = written
    for path in out.glob("*.py"):
        assert b"\r\n" not in path.read_bytes()
