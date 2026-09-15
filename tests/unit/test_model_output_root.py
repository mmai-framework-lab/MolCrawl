"""Where generated model output lands, with and without MODEL_OUTPUT_ROOT.

The default has to stay exactly what it was: runs are in flight against paths
derived from LEARNING_SOURCE_DIR, and a checkpoint directory that moves is a run
that restarts from scratch. So the interesting assertion is the negative one --
unset, every path is what it always was.
"""

import os

import pytest

from molcrawl.core import paths


@pytest.fixture
def source_dir(monkeypatch):
    monkeypatch.setattr(paths, "LEARNING_SOURCE_DIR", "/corpus/learning_source_compounds")
    return "/corpus/learning_source_compounds"


def test_unset_leaves_every_path_under_the_corpus(monkeypatch, source_dir):
    monkeypatch.setattr(paths, "MODEL_OUTPUT_ROOT", "")
    assert paths.get_bert_output_path("compounds", "small") == os.path.join(
        source_dir, "compounds", "bert-output", "compounds-small"
    )
    assert paths.get_gpt2_output_path("compounds", "medium") == os.path.join(
        source_dir, "compounds", "gpt2-output", "compounds-medium"
    )


def test_set_moves_model_output_off_the_corpus(monkeypatch, source_dir):
    monkeypatch.setattr(paths, "MODEL_OUTPUT_ROOT", "/runs/compounds")
    out = paths.get_bert_output_path("compounds", "small")
    assert out == os.path.join("/runs/compounds", "compounds", "bert-output", "compounds-small")
    assert source_dir not in out


def test_dataset_paths_are_not_moved(monkeypatch, source_dir):
    """Inputs stay where the inputs are; only generated output follows the override."""
    monkeypatch.setattr(paths, "MODEL_OUTPUT_ROOT", "/runs/compounds")
    assert paths.get_custom_tokenizer_path("compounds", "bert").startswith(source_dir)


def test_the_gpt2_xl_spelling_is_unchanged_by_the_override(monkeypatch, source_dir):
    """xl still writes to the 'ex-large' directory -- the override is a base swap only."""
    monkeypatch.setattr(paths, "MODEL_OUTPUT_ROOT", "/runs/compounds")
    assert paths.get_gpt2_output_path("compounds", "xl").endswith("compounds-ex-large")
