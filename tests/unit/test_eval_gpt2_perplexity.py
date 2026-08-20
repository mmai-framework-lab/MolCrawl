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
    """RNABinDataset raises on a None vocab, so the documented invocation needs this.

    Compared against the path the RNA config builds, so moving the dictionary without
    moving this default is caught rather than passing on the filename alone.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    config = (root / "molcrawl/tasks/pretrain/configs/rna/gpt2_small.py").read_text(encoding="utf-8")
    # rna_vocab_file = os.path.join(PROJECT_ROOT, "molcrawl", "data", ...)
    body = config.split("rna_vocab_file = os.path.join(", 1)[1].split(")", 1)[0]
    parts = [p.strip().strip('",') for p in body.split(",") if p.strip() and "PROJECT_ROOT" not in p]
    assert script.DEFAULT_RNA_VOCAB == root.joinpath(*parts), (script.DEFAULT_RNA_VOCAB, parts)
    assert script.DEFAULT_RNA_VOCAB.exists(), script.DEFAULT_RNA_VOCAB


@pytest.mark.parametrize("modality", ["compounds", "rna", "molecule_nat_lang"])
def test_modalities_without_ambiguous_tokens_exclude_nothing(script, modality):
    assert script.resolve_ignored_target_ids(modality, None, None) == []


def test_pad_id_is_excluded_when_the_config_sets_one(script):
    """Only configs/genome_sequence/gpt2_small_subset.py sets it (= 5); the ladders do not.

    Passing it for a run whose config does not set it would exclude targets training
    scored, which is the mismatch this script exists to avoid.
    """
    assert script.resolve_ignored_target_ids("compounds", None, 5) == [5]


def test_only_the_documented_config_sets_a_pad_id():
    """Pins the claim the --pad-token-id-for-loss help makes."""
    root = pathlib.Path(__file__).resolve().parents[2] / "molcrawl/tasks/pretrain/configs"
    setters = {
        path.relative_to(root).as_posix(): line
        for path in root.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("pad_token_id_for_loss")
    }
    assert setters == {"genome_sequence/gpt2_small_subset.py": "pad_token_id_for_loss = 5"}


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
    got = script.checkpoint_path_for("protein_sequence", "xl", "/x/runs/ladder-gpt2-{size}/ckpt.pt")
    assert got == pathlib.Path("/x/runs/ladder-gpt2-xl/ckpt.pt")


def test_without_a_template_the_path_helper_normalises_xl(script):
    from molcrawl.core.paths import get_gpt2_output_path

    got = script.checkpoint_path_for("compounds", "xl", None)
    assert got == pathlib.Path(get_gpt2_output_path("compounds", "xl")) / "ckpt.pt"
    assert got.parent.name.endswith("ex-large"), got


def test_a_checkpoint_from_another_modality_is_refused(script):
    """--checkpoint-template carries no modality, so the config is the only check."""
    ckpt = {"config": {"dataset": "protein_sequence"}}
    with pytest.raises(SystemExit, match="protein_sequence"):
        script.check_modality_matches(ckpt, "compounds", pathlib.Path("/x/ckpt.pt"))
    assert script.check_modality_matches(ckpt, "protein_sequence", pathlib.Path("/x/ckpt.pt")) == "protein_sequence"


def test_a_checkpoint_without_a_recorded_modality_is_allowed(script):
    """Older checkpoints predate the stored config; refusing them would help nobody."""
    assert script.check_modality_matches({}, "compounds", pathlib.Path("/x/ckpt.pt")) is None


def test_loss_is_weighted_by_kept_targets_not_row_size(script):
    """cross_entropy averages over kept targets, so the weight must count those.

    Weighting by y.numel() instead would under-weight a batch whose targets are mostly
    excluded, which is exactly the case ambiguous tokens and padding produce.
    """
    import torch
    from molcrawl.models._collators.ambiguity_aware_collator import IGNORE_INDEX

    AMBIGUOUS = 7

    class StubModel:
        """Returns what GPT.forward returns: cross_entropy with ignore_index=-1."""

        def __call__(self, x, y):
            # One logit per class, constant, so the loss depends only on which targets survive.
            logits = torch.zeros(y.shape[0], y.shape[1], 16)
            logits[..., 1] = 2.0  # class 1 is the confident prediction
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 16), y.reshape(-1), ignore_index=IGNORE_INDEX
            )
            return logits, loss

    # Row 0 is all class 1 (low loss); row 1 is all the ambiguous token.
    rows = torch.tensor([[1] * 8, [AMBIGUOUS] * 8])

    def fetch(start, stop):
        return rows[start:stop]

    kept_loss, kept_tokens, _ = script.score_split(StubModel(), fetch, 2, "cpu", 0, [AMBIGUOUS])
    all_loss, all_tokens, _ = script.score_split(StubModel(), fetch, 2, "cpu", 0, ())

    # Only row 0's targets survive the exclusion: 1 row x 7 shifted positions.
    assert kept_tokens == 7
    assert all_tokens == 14
    # With the ambiguous row dropped the mean is over the confident targets alone.
    assert kept_loss < all_loss
