"""A TRACKED name the trainer never declares is coverage that is not there.

``config_keys`` is snapshotted from the trainer's globals before its configurator
runs, and ``value_sources`` skips any key the resulting config dict does not
carry. So a name that is never assigned in the trainer produces no entry, no
warning and no failure -- the manifest simply has one fewer field than the list
implies, and reading TRACKED is not enough to notice.

BERT shipped three of them (#159): ``per_device_train_batch_size`` and
``eval_steps``, which exist only on TrainingArguments, and ``checkpoint_metric``,
which configs introduce rather than override.

The trainers cannot be imported to check this -- their globals live inside
``if __name__ == "__main__"`` and importing runs nothing -- so the names are read
out of the source.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

TRAINERS = [
    pytest.param(
        "molcrawl/models/gpt2/train.py",
        "molcrawl/models/gpt2/_run_manifest.py",
        id="nanoGPT",
    ),
    pytest.param(
        "molcrawl/models/bert/main.py",
        "molcrawl/models/bert/_run_manifest.py",
        id="BERT",
    ),
]


def _assigned_names(path):
    """Names bound at module level, including inside ``if`` blocks.

    Both trainers put their defaults inside ``if __name__ == "__main__"``, which
    still binds them as globals, so the walk descends into If bodies. It does not
    descend into functions or classes: those bindings are local and never reach
    config_keys.
    """
    names = set()

    def walk(body):
        for node in body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.If):
                walk(node.body)
                walk(node.orelse)
                continue
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

    walk(ast.parse((REPO / path).read_text()).body)
    return names


def _literal(path, name):
    for node in ast.parse((REPO / path).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} defines no {name}")


@pytest.mark.parametrize("trainer, manifest", TRAINERS)
def test_every_tracked_name_is_declared_by_its_trainer(trainer, manifest):
    tracked = _literal(manifest, "TRACKED")
    declared = _assigned_names(trainer)

    missing = sorted(name for name in tracked if name not in declared)
    assert not missing, (
        f"{manifest} tracks {missing}, which {trainer} never assigns before its "
        "configurator runs. value_sources will skip them silently, so the manifest "
        "will claim less coverage than the list suggests. Either declare a default "
        "in the trainer or drop the name."
    )


@pytest.mark.parametrize("trainer, manifest", TRAINERS)
def test_tracked_is_not_empty(trainer, manifest):
    assert _literal(manifest, "TRACKED")


def test_the_check_would_catch_a_name_that_is_only_on_training_arguments():
    """The three BERT shipped were of this kind; the walk has to see the absence."""
    declared = _assigned_names("molcrawl/models/bert/main.py")

    assert "learning_rate" in declared
    assert "per_device_train_batch_size" not in declared
    assert "eval_steps" not in declared
    assert "checkpoint_metric" not in declared


def test_names_bound_only_inside_a_function_do_not_count(tmp_path):
    """A local binding never reaches globals(), so it must not read as declared."""
    source = tmp_path / "trainer.py"
    source.write_text(
        "top = 1\n"
        "if __name__ == '__main__':\n"
        "    inside_if = 2\n"
        "def f():\n"
        "    local_only = 3\n"
    )

    names = _assigned_names(source)  # absolute; REPO / absolute is the absolute path

    assert {"top", "inside_if"} <= names
    assert "local_only" not in names
