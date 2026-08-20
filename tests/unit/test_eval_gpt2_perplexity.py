"""Whole-split scoring has to stay on the scale of the run's own val losses.

get_batch blanks ambiguous targets (protein X B Z, genome N and the IUPAC codes) and,
where a config sets it, pad positions, and GPT.forward passes ignore_index. Scoring
without the same exclusions produces a number that looks comparable and is not, so
these pin the resolution of what to exclude and the pieces around it.
"""
import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "eval_gpt2_perplexity.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("eval_gpt2_perplexity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modalities_cover_every_ladder(script):
    assert set(script.MODALITIES) == {
        "compounds", "protein_sequence", "rna", "molecule_nat_lang", "genome_sequence",
    }


def test_dataset_dir_constants_exist(script):
    from molcrawl.core import paths

    for modality, constant in script.DATASET_DIR_CONSTANTS.items():
        assert hasattr(paths, constant), f"{modality} -> {constant}"


def test_rna_vocab_default_is_the_one_the_configs_use(script):
    """RNABinDataset raises on a None vocab, so the documented invocation needs this."""
    assert script.DEFAULT_RNA_VOCAB.name == "token_dictionary.pkl"
    assert script.DEFAULT_RNA_VOCAB.exists(), script.DEFAULT_RNA_VOCAB


@pytest.mark.parametrize("modality", ["compounds", "rna", "molecule_nat_lang"])
def test_modalities_without_ambiguous_tokens_exclude_nothing(script, modality):
    assert script.resolve_ignored_target_ids(modality, None, None) == []


def test_pad_id_is_excluded_when_the_config_sets_one(script):
    """compounds sets pad_token_id_for_loss = 0; training drops those targets."""
    assert script.resolve_ignored_target_ids("compounds", None, 0) == [0]


def test_protein_excludes_the_same_tokens_training_does(script):
    from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer
    from molcrawl.models._collators import (
        PROTEIN_AMBIGUOUS_TOKENS,
        resolve_ambiguous_token_ids,
    )

    expected = resolve_ambiguous_token_ids(EsmSequenceTokenizer(), PROTEIN_AMBIGUOUS_TOKENS, log=False)
    assert script.resolve_ignored_target_ids("protein_sequence", None, None) == list(expected)
    assert expected, "protein should exclude something, or this test proves nothing"


def test_genome_refuses_without_a_tokenizer_model(script, tmp_path):
    """Silently scoring N as a real target would not be comparable to the val losses."""
    missing = tmp_path / "nope.model"
    with pytest.raises(SystemExit, match="genome tokenizer model"):
        script.resolve_ignored_target_ids("genome_sequence", str(missing), None)


def test_checkpoint_template_takes_the_size_verbatim(script):
    """The protein ladder wrote runs/ladder-gpt2-xl, not the normalised ex-large."""
    template = "/x/runs/ladder-gpt2-{size}/ckpt.pt"
    assert template.format(size="xl") == "/x/runs/ladder-gpt2-xl/ckpt.pt"
