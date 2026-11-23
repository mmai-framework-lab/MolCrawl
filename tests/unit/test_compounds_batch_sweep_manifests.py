"""Reading an arm's manifest back and comparing it with the plan.

What is worth testing here is not the arithmetic -- that lives with the inventory --
but the comparison itself: that a value the manifest does not carry is distinguishable
from a value it carries as null, that a disagreement is counted rather than rounded
away, and that a float written as 0.000221 matches a plan that says 2.21e-04.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load(name: str):
    path = repo_root() / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load("compounds_batch_sweep_manifests")


@pytest.fixture(scope="module")
def arms_mod():
    return load("compounds_batch_sweep_arms")


# --- dig ------------------------------------------------------------------------

def test_a_nested_path_is_read(mod):
    assert mod.dig({"a": {"b": {"c": 7}}}, "a/b/c") == 7


def test_a_missing_path_is_not_none(mod):
    # A manifest that recorded null and a manifest that recorded nothing are
    # different facts, and only one of them is a disagreement.
    assert mod.dig({"a": {}}, "a/b") is KeyError
    assert mod.dig({"a": {"b": None}}, "a/b") is None


def test_a_path_through_a_leaf_stops(mod):
    assert mod.dig({"a": 1}, "a/b") is KeyError


# --- same -----------------------------------------------------------------------

def test_a_float_matches_its_scientific_spelling(mod):
    assert mod.same(float("2.21e-04"), 0.000221)
    assert mod.same(float("9.96e-06"), 9.96e-06)


def test_a_float_that_differs_does_not_match(mod):
    assert not mod.same(float("2.21e-04"), 0.000222)


def test_an_int_against_a_string_matches(mod):
    assert mod.same(450000, 450000)
    assert mod.same(256, "256")


def test_booleans_compare_as_booleans(mod):
    assert mod.same(True, True)
    assert not mod.same(True, False)


def test_a_missing_value_never_matches(mod):
    assert not mod.same(32, KeyError)


def test_no_expectation_always_passes(mod):
    # checked=no rows are read out, not judged.
    assert mod.same(None, "anything")


# --- cell -----------------------------------------------------------------------

def test_a_missing_value_is_an_empty_cell(mod):
    # R10 of the column convention: no NaN, no dash.
    assert mod.cell(KeyError) == ""
    assert mod.cell(None) == ""


def test_a_bool_is_spelled_out(mod):
    assert mod.cell(True) == "True"


# --- the run directory ----------------------------------------------------------

def test_gpt2_xl_resolves_to_ex_large(mod):
    arm = {"arch": "gpt2", "size": "xl", "batch": "256", "lr_slug": "5p06e4"}
    assert mod.run_dir(arm, "/r").endswith("gpt2-output/compounds-ex-large-b256-lr5p06e4")


def test_bert_resolves_to_its_own_tree(mod):
    arm = {"arch": "bert", "size": "small", "batch": "256", "lr_slug": "2p21e4"}
    assert mod.run_dir(arm, "/r").endswith("bert-output/compounds-small-b256-lr2p21e4")


# --- end to end over a manifest built from the plan -----------------------------

def bert_manifest(arm):
    """What a run that followed the plan exactly would have recorded."""
    return {
        "placement": {"world_size": 4},
        "batch": {"per_device_train_batch_size": int(arm["micro_batch"]),
                  "gradient_accumulation_steps": int(arm["grad_accum"]),
                  "expected_global_batch": int(arm["batch"]),
                  "effective_global_batch": int(arm["batch"])},
        "schedule": {"max_steps": int(arm["steps"]),
                     "warmup_steps": int(arm["warmup_steps"]),
                     "learning_rate": float(arm["lr"])},
        "objective": {"mlm_probability": 0.2},
        "eval": {"rows": 10000, "subset_random": True, "subset_seed": 42,
                 "fixed_mask": {"enabled": True, "seed": 42}},
        "selection": {"judge_on": arm["judge_metric"],
                      "metric_for_best_model": arm["judge_metric"],
                      "save_steps": int(arm["save_interval"]),
                      "eval_steps": int(arm["eval_interval"]),
                      "retention": {"metric": arm["judge_metric"]}},
        "introduced": {"checkpoint_metric": arm["judge_metric"],
                       "save_on_improve_metric": arm["judge_metric"]},
        "seed": {"seed": 42},
    }


@pytest.fixture
def one_arm(arms_mod, tmp_path):
    arm = next(r for r in arms_mod.arms("micro-order")
               if r["arch"] == "bert" and r["batch"] == 256)
    arm = {k: ("" if v is None else str(v)) for k, v in arm.items()}
    where = tmp_path / "compounds" / "bert-output" / f"compounds-{arm['size']}-b256-lr{arm['lr_slug']}"
    where.mkdir(parents=True)
    (where / "run_manifest.json").write_text(json.dumps(bert_manifest(arm)),
                                             encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / Path(arm["config"]).name).write_text(
        f"per_device_eval_batch_size = {arm['eval_batch']}\n", encoding="utf-8")
    return arm, tmp_path, configs


def test_a_manifest_that_followed_the_plan_disagrees_nowhere(mod, one_arm, capsys):
    arm, root, configs = one_arm
    rows, missing, problems = mod.rows_for([arm], str(root), str(configs))
    capsys.readouterr()
    assert not missing
    assert problems == 0
    assert all(r["agrees"] in ("yes", "") for r in rows)


def test_every_key_the_directive_lists_is_read(mod, one_arm):
    arm, root, configs = one_arm
    rows, _, _ = mod.rows_for([arm], str(root), str(configs))
    keys = {r["key"] for r in rows}
    for key in ("seed", "fixed_eval_mask", "eval_mask_seed", "judge_on",
                "metric_for_best_model", "save_on_improve_metric",
                "checkpoint_metric", "retention_metric", "gpus", "eval_rows",
                "eval_subset_random", "eval_subset_seed", "mask_rate", "batch_size",
                "gradient_accumulation_steps", "expected_global_batch", "max_steps",
                "warmup_steps", "eval_steps", "save_steps", "learning_rate",
                "per_device_eval_batch_size"):
        assert key in keys, key


def test_one_wrong_value_is_counted_and_named(mod, one_arm, capsys):
    arm, root, configs = one_arm
    path = (root / "compounds" / "bert-output"
            / f"compounds-{arm['size']}-b256-lr{arm['lr_slug']}" / "run_manifest.json")
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["selection"]["judge_on"] = "eval_loss"
    path.write_text(json.dumps(blob), encoding="utf-8")
    rows, _, problems = mod.rows_for([arm], str(root), str(configs))
    capsys.readouterr()
    assert problems == 1
    bad = [r for r in rows if r["agrees"] == "NO"]
    assert len(bad) == 1 and bad[0]["key"] == "judge_on"
    assert bad[0]["observed"] == "eval_loss"


def test_an_arm_without_a_manifest_is_reported_not_passed(mod, one_arm, capsys):
    arm, root, configs = one_arm
    other = dict(arm, lr_slug="0p00e0")
    rows, missing, problems = mod.rows_for([arm, other], str(root), str(configs))
    capsys.readouterr()
    assert len(missing) == 1
    assert problems == 0
    assert all(r["arm"].endswith(arm["lr_slug"]) for r in rows)


def test_the_eval_batch_row_names_the_config_as_its_source(mod, one_arm):
    arm, root, configs = one_arm
    rows, _, _ = mod.rows_for([arm], str(root), str(configs))
    row = next(r for r in rows if r["key"] == "per_device_eval_batch_size")
    assert row["source"].startswith("config:")
    assert row["observed"] == str(arm["eval_batch"])


def test_a_missing_config_leaves_the_eval_batch_disagreeing(mod, one_arm, capsys):
    arm, root, _ = one_arm
    rows, _, problems = mod.rows_for([arm], str(root), str(root / "nowhere"))
    capsys.readouterr()
    row = next(r for r in rows if r["key"] == "per_device_eval_batch_size")
    assert row["observed"] == "" and row["agrees"] == "NO"
    assert problems == 1
