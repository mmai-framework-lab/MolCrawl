"""The parameter count has to survive a checkpoint that was written as shards.

A model over the single-file limit is saved as `model-0000k-of-0000n.safetensors`
plus an index, and `model.safetensors` is then absent. Reading only that one name
returned no count for every xl run, so the largest size was dropped from any
figure drawn against compute without anything being logged.

These build safetensors headers directly -- eight bytes of length, then JSON --
so no weights exist and no GPU is involved.
"""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bert_grid_summary.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bert_grid_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: Path, tensors):
    """A header-only safetensors file: the shapes are all these tests read."""
    header = {name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in tensors.items()}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_counts_a_single_file_checkpoint(mod, tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"a": (4, 5), "b": (3,)})
    assert mod.param_count(tmp_path) == 23


def test_counts_a_sharded_checkpoint(mod, tmp_path):
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": (4, 5)})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b": (3, 7)})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {"a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"},
    }), encoding="utf-8")
    assert mod.param_count(tmp_path) == 20 + 21


def test_a_tensor_in_two_shards_is_counted_once(mod, tmp_path):
    """Tied weights appear in both shards' headers; the sum must not double them."""
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors",
                      {"shared": (2, 3), "a": (5,)})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors",
                      {"shared": (2, 3), "b": (7,)})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"shared": "model-00001-of-00002.safetensors",
                       "a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"},
    }), encoding="utf-8")
    assert mod.param_count(tmp_path) == 6 + 5 + 7


def test_a_missing_shard_is_not_a_partial_count(mod, tmp_path):
    """Half a model counted as a whole one would move N, and with it the x axis."""
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": (4, 5)})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"},
    }), encoding="utf-8")
    assert mod.param_count(tmp_path) is None


def test_no_weights_at_all_is_none(mod, tmp_path):
    assert mod.param_count(tmp_path) is None


def test_schedule_reads_both_steps(mod, tmp_path):
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 85000, "max_steps": 120960, "log_history": []}),
        encoding="utf-8")
    assert mod.schedule(tmp_path) == (85000, 120960)


def test_non_embedding_subtracts_the_embeddings(mod):
    """N for 6ND is the part that enters the matrix multiplies."""
    conf = {"hidden_size": 10, "vocab_size": 100, "max_position_embeddings": 20}
    assert mod.non_embedding(5_000, conf) == 5_000 - (100 * 10 + 20 * 10 + 2 * 10)
