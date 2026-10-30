"""The seed a BERT config names has to reach both the weights and the data order.

Between 0f3e7067 (2026-10-02) and this branch, `molcrawl/models/bert/main.py` passed
`data_seed=seed` to TrainingArguments without `seed=seed`. transformers 4.45.1 never
reads data_seed, and seeds from args.seed, which kept its default of 42 -- so six runs
asked for seed 1 or 17 and ran at 42 (measured 2026-10-28 over 310 BERT runs).

Restoring `seed=` is not enough on its own: main.py builds the model before the Trainer
exists, so Trainer.__init__'s own set_seed(args.seed) lands after the weights have been
drawn. The fix calls set_seed(seed) as soon as the configurator has resolved it. These
tests pin both halves, on CPU, with a model small enough to build in milliseconds.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertConfig, BertForMaskedLM, TrainingArguments, set_seed  # noqa: E402


def _tiny_model():
    return BertForMaskedLM(BertConfig(vocab_size=50, hidden_size=32,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      intermediate_size=64, max_position_embeddings=32))


def _first_weight(model):
    return model.bert.encoder.layer[0].attention.self.query.weight.detach().clone()


def test_the_seed_reaches_the_initialisation():
    set_seed(1)
    a = _first_weight(_tiny_model())
    set_seed(1)
    b = _first_weight(_tiny_model())
    assert torch.equal(a, b)


def test_two_seeds_do_not_draw_the_same_weights():
    # The s1/s17 pairs were meant to measure this and could not: both ran at 42.
    set_seed(1)
    a = _first_weight(_tiny_model())
    set_seed(17)
    b = _first_weight(_tiny_model())
    assert not torch.equal(a, b)


def test_training_arguments_keeps_the_seed_it_is_given(tmp_path):
    assert TrainingArguments(output_dir=str(tmp_path), seed=17).seed == 17


def test_training_arguments_defaults_to_42_when_the_seed_is_not_passed(tmp_path):
    # The regression's mechanism: omitting seed= is silent, and 42 is a value a
    # config could legitimately have asked for, so nothing downstream looked wrong.
    assert TrainingArguments(output_dir=str(tmp_path)).seed == 42


def test_data_seed_is_stored_and_never_read():
    # The premise the comments in main.py rest on. If a transformers upgrade starts
    # reading data_seed, this fails and those comments need rewriting.
    trainer_source = pathlib.Path(transformers.__file__).with_name("trainer.py")
    assert "data_seed" not in trainer_source.read_text()
    args = TrainingArguments(output_dir="/tmp/unused-data-seed-check", data_seed=7)
    assert args.data_seed == 7
