"""The guard has to refuse the three ways output reached an input tree."""

import os

import pytest

from molcrawl.core.output_guard import assert_output_dir, is_inside, map_cache_path

CORPUS = "/data1/rkp00024/yigarashi/learning_source"


@pytest.fixture
def corpus_env(monkeypatch):
    monkeypatch.setenv("LEARNING_SOURCE_DIR", CORPUS)
    monkeypatch.delenv("GENOME_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("SRC_ROOT", raising=False)


def test_refuses_output_inside_the_corpus(corpus_env):
    with pytest.raises(ValueError, match="inside an input tree"):
        assert_output_dir(f"{CORPUS}/genome_sequence/mammal_centered/parquet_bert")


def test_refuses_the_corpus_root_itself(corpus_env):
    with pytest.raises(ValueError, match="inside an input tree"):
        assert_output_dir(CORPUS)


def test_allows_a_directory_we_own(corpus_env, tmp_path):
    assert assert_output_dir(tmp_path / "runs" / "subset") == (
        tmp_path / "runs" / "subset").resolve()


def test_refuses_an_unwritable_destination(corpus_env, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        with pytest.raises(ValueError, match="not writable"):
            assert_output_dir(locked / "out")
    finally:
        os.chmod(locked, 0o700)


def test_extra_roots_are_honoured_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LEARNING_SOURCE_DIR", raising=False)
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(ValueError, match="inside an input tree"):
        assert_output_dir(src / "derived", extra_roots=(src,))


def test_map_cache_sits_beside_the_output_not_the_input(tmp_path):
    dst = tmp_path / "out" / "training_ready_hf_dataset_bert"
    cache = map_cache_path(dst)
    assert is_inside(cache, tmp_path / "out")
    assert not is_inside(cache, dst)


def test_is_inside_does_not_match_a_sibling_prefix(tmp_path):
    assert not is_inside(tmp_path / "learning_source_1024", tmp_path / "learning_source")
