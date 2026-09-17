"""configurator.py records the config-file stage and refuses a misleading argv order.

Both trainers' configurators are the same file; the test runs each the way the
trainers do, by ``exec`` into a module-like namespace.
"""

import os
import sys

import pytest

from molcrawl.models._provenance import ArgvOrderError

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGURATORS = [
    os.path.join(REPO, "molcrawl", "models", "gpt2", "configurator.py"),
    os.path.join(REPO, "molcrawl", "models", "bert", "configurator.py"),
]


def _run(configurator, argv, monkeypatch, trainer_file):
    ns = {"__name__": "__main__", "__file__": trainer_file, "learning_rate": 6e-4, "seed": 1337}
    monkeypatch.setattr(sys, "argv", ["trainer.py"] + argv)
    exec(open(configurator).read(), ns)
    return ns


@pytest.mark.parametrize("configurator", CONFIGURATORS, ids=["gpt2", "bert"])
def test_after_config_file_holds_config_values_not_cli_overrides(configurator, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("learning_rate = 1e-4\nextra_name = 3\n")
    ns = _run(configurator, [str(cfg), "--seed=7"], monkeypatch, str(tmp_path / "trainer.py"))
    snap = ns["_config_after_file"]
    assert snap["learning_rate"] == 1e-4
    assert snap["extra_name"] == 3
    assert snap["seed"] == 1337
    assert ns["seed"] == 7


@pytest.mark.parametrize("configurator", CONFIGURATORS, ids=["gpt2", "bert"])
def test_without_overrides_the_snapshot_is_taken_at_the_end(configurator, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("learning_rate = 2e-4\n")
    ns = _run(configurator, [str(cfg)], monkeypatch, str(tmp_path / "trainer.py"))
    assert ns["_config_after_file"]["learning_rate"] == 2e-4


@pytest.mark.parametrize("configurator", CONFIGURATORS, ids=["gpt2", "bert"])
def test_without_a_config_file_the_snapshot_is_the_defaults(configurator, tmp_path, monkeypatch):
    ns = _run(configurator, ["--seed=7"], monkeypatch, str(tmp_path / "trainer.py"))
    assert ns["_config_after_file"]["seed"] == 1337
    assert ns["seed"] == 7


@pytest.mark.parametrize("configurator", CONFIGURATORS, ids=["gpt2", "bert"])
def test_config_file_after_override_is_refused(configurator, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("learning_rate = 1e-4\n")
    with pytest.raises(ArgvOrderError):
        _run(configurator, ["--seed=7", str(cfg)], monkeypatch, str(tmp_path / "trainer.py"))


@pytest.mark.parametrize("configurator", CONFIGURATORS, ids=["gpt2", "bert"])
def test_unknown_key_and_type_mismatch_are_still_refused(configurator, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown config key"):
        _run(configurator, ["--nope=1"], monkeypatch, str(tmp_path / "trainer.py"))
    with pytest.raises(AssertionError, match="Type mismatch"):
        _run(configurator, ["--seed=0.5"], monkeypatch, str(tmp_path / "trainer.py"))


def test_both_configurators_are_still_the_same_file():
    a, b = (open(path).read() for path in CONFIGURATORS)
    assert a == b
