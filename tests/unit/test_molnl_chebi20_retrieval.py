"""The ChEBI-20 retrieval scorer: ranks count ties against the model, metrics follow ranks."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ret():
    return _load("molnl_chebi20_retrieval")


def test_rank_is_one_when_the_true_candidate_is_strictly_best(ret):
    text = torch.tensor([[1.0, 0.0]])
    mols = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    ranks, ties = ret.ranks_of_true(text, mols, torch.tensor([1]))
    assert ranks.tolist() == [1]
    assert ties.tolist() == [0]


def test_a_tie_is_not_resolved_in_the_models_favour(ret):
    # Two candidates with the same vector -- the one pair of ChEBI-20 CIDs whose SDF
    # gives the same SMILES -- must not both be rank 1.
    text = torch.tensor([[1.0, 0.0]])
    mols = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ranks, ties = ret.ranks_of_true(text, mols, torch.tensor([0]))
    assert ranks.tolist() == [2]
    assert ties.tolist() == [1]


def test_rank_counts_every_higher_candidate(ret):
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mols = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
    ranks, _ = ret.ranks_of_true(text, mols, torch.tensor([3, 1]), chunk=1)
    # query 0: 0.9 and 0.8 are above 0.7 -> rank 3. query 1: 0.9 and 0.3 above 0.2 -> rank 3.
    assert ranks.tolist() == [3, 3]


def test_metrics_follow_from_ranks(ret):
    m = ret.metrics([1, 2, 10, 11])
    assert m["hits@1"] == pytest.approx(0.25)
    assert m["hits@10"] == pytest.approx(0.75)
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 10 + 1 / 11) / 4)
    assert m["mean_rank"] == pytest.approx(6.0)


def test_loss_is_symmetric_and_low_for_aligned_pairs(ret):
    eye = torch.eye(4)
    aligned = ret.clip_loss(eye, eye, torch.tensor(100.0))
    shuffled = ret.clip_loss(eye, eye[[1, 0, 3, 2]], torch.tensor(100.0))
    assert aligned.item() < 1e-6
    assert shuffled.item() > 10
