"""The four execution settings are declared in BERT main.py and accepted from the command line.

Before they were declared, main.py read them through ``globals().get(..., False)``:
nothing moved by default, but ``--bf16=True`` was rejected by the configurator as an
unknown key, and a config that did not set them left the manifest with no record.
Declaring them must not change what an existing run gets, so the defaults are held
to the values that fallback produced.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "molcrawl" / "models" / "bert" / "main.py"
CONFIGURATOR = ROOT / "molcrawl" / "models" / "bert" / "configurator.py"

# The value each one had through the old globals().get fallback.
FALLBACK = {
    "bf16": False,
    "tf32": False,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": False,
}


def _declared_defaults():
    found = {}
    for node in ast.walk(ast.parse(MAIN.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in FALLBACK:
                found[target.id] = ast.literal_eval(node.value)
    return found


def test_declared_with_the_value_the_fallback_gave():
    assert _declared_defaults() == FALLBACK


def test_training_arguments_no_longer_reads_them_through_globals_get():
    source = MAIN.read_text(encoding="utf-8")
    for name in FALLBACK:
        assert f'globals().get("{name}"' not in source, name


@pytest.mark.parametrize(
    "arg, name, expected",
    [
        ("--bf16=True", "bf16", True),
        ("--tf32=True", "tf32", True),
        ("--dataloader_num_workers=4", "dataloader_num_workers", 4),
        ("--dataloader_pin_memory=True", "dataloader_pin_memory", True),
    ],
)
def test_configurator_accepts_them_from_the_command_line(monkeypatch, arg, name, expected):
    scope = dict(FALLBACK, __name__="__main__")
    monkeypatch.setattr(sys, "argv", ["main.py", arg])
    exec(CONFIGURATOR.read_text(encoding="utf-8"), scope)
    assert scope[name] == expected


def test_configurator_still_rejects_a_wrong_type(monkeypatch):
    scope = dict(FALLBACK, __name__="__main__")
    monkeypatch.setattr(sys, "argv", ["main.py", "--dataloader_num_workers=True"])
    with pytest.raises(AssertionError):
        exec(CONFIGURATOR.read_text(encoding="utf-8"), scope)
