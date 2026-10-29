"""The two places the checkpoint-metric audit got a wrong answer before.

1. `eval_loss_mask` does not share a log_history row with `eval_loss`: the breakdown
   mixin adds it through `self.log()` after `super().evaluate()` returns. A scan that
   only reads the eval rows finds no mask metric at all.
2. A run name and a run path are prefixes of their siblings'
   (`compounds-medium-lr2p8e4` of `compounds-medium-lr2p8e4-seed42`). A plain
   substring search credited 21 genome runs and one compounds run with downstream
   evaluations that had read a different checkpoint.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from audit_checkpoint_metric_match import (  # noqa: E402
    eval_set_description,
    metric_minimum,
    model_size,
    seed_description,
)
from audit_downstream_uses_of_runs import NAME_TAIL  # noqa: E402

import re  # noqa: E402


# The shape a real trainer_state.json has: three kinds of row per eval point.
HISTORY = [
    {"step": 100, "loss": 1.0},
    {"step": 100, "eval_loss": 0.50},
    {"step": 100, "eval_loss_mask": 0.90, "eval_loss_copy": 0.1},
    {"step": 200, "eval_loss": 0.40},
    {"step": 200, "eval_loss_mask": 0.70},
    {"step": 300, "eval_loss": 0.30},
    {"step": 300, "eval_loss_mask": 0.80},
]


def test_mask_minimum_comes_from_its_own_rows():
    got = metric_minimum(HISTORY, "eval_loss_mask")
    assert got["step"] == 200
    assert got["value"] == 0.70
    assert got["points"] == 3


def test_the_two_metrics_can_pick_different_steps():
    # eval_loss keeps falling to step 300 while the mask metric bottoms at 200.
    # This is the whole reason the audit exists, so it is pinned.
    assert metric_minimum(HISTORY, "eval_loss")["step"] == 300
    assert metric_minimum(HISTORY, "eval_loss_mask")["step"] == 200


def test_a_metric_absent_from_the_history_is_not_a_zero():
    assert metric_minimum(HISTORY, "eval_loss_random") is None


def test_ties_are_reported_rather_than_broken_silently():
    history = [
        {"step": 10, "eval_loss_mask": 0.5},
        {"step": 20, "eval_loss_mask": 0.5},
        {"step": 30, "eval_loss_mask": 0.6},
    ]
    got = metric_minimum(history, "eval_loss_mask")
    assert got["step"] == 10
    assert got["tied_steps"] == [10, 20]


def test_no_tie_leaves_the_list_empty():
    assert metric_minimum(HISTORY, "eval_loss_mask")["tied_steps"] == []


def _hits(needle, text):
    return bool(re.search(re.escape(needle) + NAME_TAIL, text))


def test_a_run_name_does_not_match_its_longer_sibling():
    assert not _hits("compounds-medium-lr2p8e4", "compounds-medium-lr2p8e4-seed42")
    assert not _hits("bert-small-global_random_seed1",
                     "bert-small-global_random_seed1-w1026")


def test_a_run_path_still_matches_its_own_checkpoint():
    assert _hits("/runs/compounds-medium-lr2p8e4",
                 "/runs/compounds-medium-lr2p8e4/checkpoint-15000")


def test_a_trailing_digit_does_not_extend_a_name():
    # seed1 must not be found inside seed10.
    assert not _hits("bert-small-global_random_seed1",
                     "bert-small-global_random_seed10/metrics.json")


def test_size_falls_back_to_the_run_name():
    assert model_size("compounds-large-lr1p5e4", {}) == ("large", "run name")
    assert model_size("compounds-ex-large-lr4e4", {}) == ("ex-large", "run name")
    assert model_size("bert-small-mammal_centered-w1026", None) == ("small", "run name")


def test_a_recorded_size_wins_over_the_name():
    manifest = {"introduced": {"model_size": "xl"}}
    assert model_size("whatever-large-lr1", manifest) == ("xl", "manifest")


def test_eval_set_description_separates_head_from_random():
    head = {"arch": "bert", "eval": {"rows": 10000, "subset_random": False}}
    assert eval_set_description(head) == "固定 10,000 行（先頭取り）"
    drawn = {"arch": "bert",
             "eval": {"rows": 10000, "subset_random": True, "subset_seed": 42}}
    assert "無作為抽出、seed 42" in eval_set_description(drawn)


def test_gpt2_redraw_is_not_reported_as_fixed():
    redraw = {"arch": "gpt2",
              "eval": {"rows": 3200, "deterministic_val_eval": False}}
    assert eval_set_description(redraw).startswith("引き直し")
    fixed = {"arch": "gpt2",
             "eval": {"rows": 3200, "deterministic_val_eval": True}}
    assert fixed and eval_set_description(fixed).startswith("固定")


def test_a_missing_record_is_not_guessed():
    assert eval_set_description({"arch": "gpt2", "eval": {}}) == "不明（manifest 無し）"
    assert eval_set_description({"arch": "bert", "eval": {}}) == "不明（manifest 無し）"


def test_a_seed_the_config_asked_for_but_did_not_get_is_shown():
    assert seed_description({"seed": 42, "seed_configured": 1}) == \
        "42（config は 1 を指定。効いていない）"
    assert seed_description({"seed": 9, "seed_configured": 9}) == "9"
    assert seed_description({"seed": 42, "seed_configured": None}) == "42"
