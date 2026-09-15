"""The shared trunk loader the probes read their features from.

It sits in molcrawl/models because genome, protein, compounds, rna and
molecule_nat_lang all need the same thing. With one implementation per modality,
a difference between modalities could not be told apart from a difference
between implementations.

Only the pure parts are exercised here; loading a trunk needs a checkpoint and a
GPU, and the probe's own run covers that.
"""
from molcrawl.models._representations import (special_token_offset,
                                              strip_compile_prefix)


def test_encoders_shift_by_one_and_decoders_do_not():
    """HF encoders prepend [CLS]; nanoGPT and llama prepend nothing."""
    assert special_token_offset("bert") == 1
    assert special_token_offset("roberta") == 1
    assert special_token_offset("esm2") == 1
    assert special_token_offset("gpt2") == 0
    assert special_token_offset("llama") == 0


def test_an_unknown_architecture_is_treated_as_an_encoder():
    """The HF families outnumber the rest, and a wrong 0 would read the token
    before the one intended; a wrong 1 is caught by the centre-vs-ref check."""
    assert special_token_offset("chemberta2") == 1
    assert special_token_offset("something-new") == 1


def test_the_compile_prefix_is_stripped():
    """torch.compile saves under _orig_mod.; load_state_dict would reject it."""
    got = strip_compile_prefix({"_orig_mod.wte.weight": 1, "ln_f.bias": 2})

    assert got == {"wte.weight": 1, "ln_f.bias": 2}


def test_a_state_dict_without_the_prefix_is_unchanged():
    d = {"wte.weight": 1, "ln_f.bias": 2}

    assert strip_compile_prefix(d) == d
