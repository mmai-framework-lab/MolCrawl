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
