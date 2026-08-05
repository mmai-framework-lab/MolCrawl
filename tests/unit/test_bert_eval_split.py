"""Which split BERT evaluates on — and therefore selects checkpoints on.

Regression guard for the 2026-08-05 fix: the loader used to prefer ``test``
whenever it was present, so ``load_best_model_at_end`` picked the checkpoint
with the lowest *test* loss and the test split was no longer held out.
"""

from molcrawl.models.bert.main import DEFAULT_EVAL_SPLIT, resolve_eval_split


def test_defaults_to_valid_even_when_test_exists():
    # The regression: every modality's training_ready carries all three splits.
    assert resolve_eval_split(["train", "valid", "test"]) == "valid"


def test_default_constant_is_valid():
    assert DEFAULT_EVAL_SPLIT == "valid"


def test_config_can_pin_another_split():
    assert resolve_eval_split(["train", "valid", "test"], "test") == "test"


def test_falls_back_to_valid_when_requested_split_is_absent():
    assert resolve_eval_split(["train", "valid"], "holdout") == "valid"


def test_falls_back_to_test_when_there_is_no_valid():
    assert resolve_eval_split(["train", "test"]) == "test"


def test_returns_none_when_no_eval_split_exists():
    assert resolve_eval_split(["train"]) is None


def test_accepts_any_iterable_of_names():
    # Call sites pass DatasetDict.keys() and plain lists.
    assert resolve_eval_split({"train": 1, "valid": 2}.keys()) == "valid"
