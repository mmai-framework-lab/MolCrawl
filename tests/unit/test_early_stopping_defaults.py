"""Both pretraining paths default to early stopping on (2026-11-16).

Read from the source rather than by importing the trainers: both run their whole
training script at import time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


def test_gpt2_defaults_to_early_stopping_on_with_patience_10():
    tree = ast.parse((repo_root() / "molcrawl/models/gpt2/train.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("early_stopping", "early_stopping_patience"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    assert found == {"early_stopping": True, "early_stopping_patience": 10}


def test_bert_defaults_to_early_stopping_on_with_patience_10():
    src = (repo_root() / "molcrawl/models/bert/main.py").read_text()
    assert re.search(r'early_stopping = globals\(\)\.get\("early_stopping", True\)', src)
    assert re.search(r'early_stopping_patience = globals\(\)\.get\("early_stopping_patience", 10\)', src)


def test_bert_uses_the_callback_that_sees_eval_loss_mask():
    src = (repo_root() / "molcrawl/models/bert/main.py").read_text()
    assert "LoggedMetricEarlyStopping(early_stopping_patience=" in src
    assert "callbacks.append(EarlyStoppingCallback(" not in src


def test_turning_early_stopping_on_for_a_run_checkpointed_without_it_can_still_save():
    """The crash this guards against happens after the step was trained, not at startup.

    ``Trainer._save_checkpoint`` writes every ``ExportableState`` callback into
    ``state.stateful_callbacks[cb.__class__.__name__]`` with no default, and a resume
    restores that dict from the checkpoint. A checkpoint written with early stopping off
    has no entry for the callback, so the first save after the resume died with
    ``KeyError: 'LoggedMetricEarlyStopping'`` -- molnl BERT jobs 355708 and 355711 on
    2026-11-16, both at step 24,100, both on an improvement they could not write.
    """
    from transformers import TrainerControl, TrainerState

    from molcrawl.models.bert._early_stopping import LoggedMetricEarlyStopping

    # What a resume restores from a checkpoint written without this callback.
    state = TrainerState(stateful_callbacks={"TrainerControl": TrainerControl().state()})
    stopper = LoggedMetricEarlyStopping(early_stopping_patience=90)
    assert "LoggedMetricEarlyStopping" not in state.stateful_callbacks

    stopper.on_train_begin(None, state, TrainerControl())
    entry = state.stateful_callbacks["LoggedMetricEarlyStopping"]
    assert entry["args"]["early_stopping_patience"] == 90
    assert entry["attributes"]["early_stopping_patience_counter"] == 0
    # HF indexes the dict the same way on the save path; this is that lookup.
    assert not isinstance(state.stateful_callbacks["LoggedMetricEarlyStopping"], list)


def test_a_restored_counter_is_not_reset_by_that_seeding():
    from transformers import TrainerControl, TrainerState

    from molcrawl.models.bert._early_stopping import LoggedMetricEarlyStopping

    restored = {"args": {"early_stopping_patience": 90, "early_stopping_threshold": 0.0},
                "attributes": {"early_stopping_patience_counter": 7}}
    state = TrainerState(stateful_callbacks={"LoggedMetricEarlyStopping": restored})
    LoggedMetricEarlyStopping(early_stopping_patience=90).on_train_begin(None, state, TrainerControl())
    assert state.stateful_callbacks["LoggedMetricEarlyStopping"]["attributes"][
        "early_stopping_patience_counter"] == 7
