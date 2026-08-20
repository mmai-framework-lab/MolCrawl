"""Whole-split scoring has to stay on the scale of the run's own val losses.

get_batch blanks ambiguous targets (protein X B Z, genome N and the IUPAC codes) and,
where a config sets it, pad positions, and GPT.forward passes ignore_index. Scoring
without the same exclusions produces a number that looks comparable and is not, so
these pin the resolution of what to exclude and the pieces around it.
"""
import importlib.util
import pathlib
import sys

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


def test_the_refusal_path_does_not_need_sentencepiece(script, tmp_path, monkeypatch):
    """CI installs no sentencepiece, so importing it first turned this into an error.

    environment.yaml pins the library for the cluster, but the unit-test job does not
    install it, and a missing tokenizer model has to report itself either way. Setting
    the module to None makes `import sentencepiece` raise however the environment is
    set up, so this pins the ordering rather than the installed package set.
    """
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    with pytest.raises(SystemExit, match="genome tokenizer model"):
        script.resolve_ignored_target_ids("genome_sequence", str(tmp_path / "nope.model"), None)


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


def test_an_allowed_mismatch_still_records_what_it_scored(script):
    """The mismatch case is where trained_on matters most, so it must not go missing."""
    ckpt = {"config": {"dataset": "protein_sequence"}}
    got = script.check_modality_matches(ckpt, "compounds", pathlib.Path("/x/ckpt.pt"), strict=False)
    assert got == "protein_sequence"


def test_exclusions_follow_the_config_string_not_the_modality_flag(script):
    """`protein_sequence_proteingym` trained without the X B Z mask; scoring must match.

    train.py looks the list up by the config's `dataset` string, and
    MODALITY_TO_AMBIGUOUS has no entry for the proteingym variant, so it resolves to
    nothing. Excluding protein's tokens for it would be the mismatch in reverse.
    """
    assert script.resolve_ignored_target_ids("protein_sequence_proteingym", None, None) == []
    assert script.resolve_ignored_target_ids("protein_sequence", None, None) != []


def test_the_variant_config_strings_are_real():
    """Guards the premise of the test above against the configs being renamed."""
    root = pathlib.Path(__file__).resolve().parents[2] / "molcrawl/tasks/pretrain/configs"
    seen = {
        line.split(" = ", 1)[1].strip().strip('"')
        for path in root.rglob("gpt2_*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("dataset = ")
    }
    assert "protein_sequence_proteingym" in seen
    assert seen - set(script_modalities()), "variants beyond --modality should exist"


def script_modalities():
    return {"compounds", "protein_sequence", "rna", "molecule_nat_lang", "genome_sequence"}


def test_a_checkpoint_without_a_recorded_modality_is_allowed(script):
    """Older checkpoints predate the stored config; refusing them would help nobody."""
    assert script.check_modality_matches({}, "compounds", pathlib.Path("/x/ckpt.pt")) is None


def test_loss_is_weighted_by_kept_targets_across_batches(script):
    """cross_entropy averages over kept targets, so the running weight must count those.

    Within a single batch the two weightings agree (loss*w/w), so this spans two
    batches with different exclusion rates - which is where weighting by y.numel()
    actually produces a different mean.
    """
    import torch
    from molcrawl.models._collators.ambiguity_aware_collator import IGNORE_INDEX

    AMBIGUOUS, SEQ = 7, 8
    per_row = SEQ - 1  # targets after the causal shift

    class StubModel:
        """Returns what GPT.forward returns: cross_entropy with ignore_index=-1."""

        def __call__(self, x, y):
            logits = torch.zeros(y.shape[0], y.shape[1], 16)
            logits[..., 1] = 4.0  # class 1 is cheap, anything else is expensive
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 16), y.reshape(-1), ignore_index=IGNORE_INDEX
            )
            return logits, loss

    # Batch 1 (16 rows): class 1 throughout, nothing excluded -> cheap loss.
    # Batch 2 (16 rows): class 2 for the first half, ambiguous after -> expensive loss
    #                    over 3 of the 7 targets per row.
    batch1 = torch.tensor([[1] * SEQ] * script.BATCH_BLOCKS)
    batch2 = torch.tensor([[2, 2, 2, 2, AMBIGUOUS, AMBIGUOUS, AMBIGUOUS, AMBIGUOUS]] * script.BATCH_BLOCKS)
    rows = torch.cat([batch1, batch2])

    def fetch(start, stop):
        return rows[start:stop]

    got, kept_tokens, sequences = script.score_split(
        StubModel(), fetch, len(rows), "cpu", 0, [AMBIGUOUS]
    )
    assert sequences == 2 * script.BATCH_BLOCKS

    # Per-batch losses, to build both weightings by hand.
    model = StubModel()
    from molcrawl.models._collators import mask_ambiguous_targets_for_clm

    losses, kept = [], []
    for batch in (batch1, batch2):
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        y = mask_ambiguous_targets_for_clm(y, [AMBIGUOUS]).contiguous()
        losses.append(float(model(x, y)[1]))
        kept.append(int((y != IGNORE_INDEX).sum()))

    assert kept == [script.BATCH_BLOCKS * per_row, script.BATCH_BLOCKS * 3]
    assert kept_tokens == sum(kept)

    by_kept = (losses[0] * kept[0] + losses[1] * kept[1]) / sum(kept)
    numel = script.BATCH_BLOCKS * per_row
    by_numel = (losses[0] * numel + losses[1] * numel) / (2 * numel)

    assert got == pytest.approx(by_kept, rel=1e-6)
    # The regression this guards: the two weightings must not coincide here, or the
    # test would pass with either.
    assert abs(by_kept - by_numel) > 0.05, (by_kept, by_numel)
