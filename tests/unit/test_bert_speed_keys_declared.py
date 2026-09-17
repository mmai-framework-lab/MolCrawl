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
    "dataloader_persistent_workers": False,
    "torch_compile": False,
    # The old fallback was None; "" is declared because None is not snapshotted,
    # and main.py maps "" back to None before TrainingArguments sees it.
    "torch_compile_backend": "",
    "stop_at_step": 0,
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


def _code_lines():
    return [ln for ln in MAIN.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]


def test_training_arguments_no_longer_reads_them_through_globals_get():
    code = "\n".join(_code_lines())
    for name in FALLBACK:
        assert f'globals().get("{name}"' not in code, name


def test_flash_attention_stays_undeclared():
    # transformers 4.45.1's BERT cannot build with flash_attention_2, so declaring the
    # key would make a setting that only fails look like a working one.
    assert "flash_attention" not in _declared_defaults()
    assert not any(ln.strip().startswith("flash_attention =") for ln in _code_lines())


@pytest.mark.parametrize(
    "arg, name, expected",
    [
        ("--bf16=True", "bf16", True),
        ("--tf32=True", "tf32", True),
        ("--dataloader_num_workers=4", "dataloader_num_workers", 4),
        ("--dataloader_pin_memory=True", "dataloader_pin_memory", True),
        ("--dataloader_persistent_workers=True", "dataloader_persistent_workers", True),
        ("--torch_compile=True", "torch_compile", True),
        ("--torch_compile_backend=inductor", "torch_compile_backend", "inductor"),
        ("--stop_at_step=8000", "stop_at_step", 8000),
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
