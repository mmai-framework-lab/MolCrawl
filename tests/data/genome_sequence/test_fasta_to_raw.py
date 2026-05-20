"""Tests for the split-at-N-runs FASTA reader.

Covers the strategy adopted in commit a9e4784: instead of dropping every
FASTA entry containing any N, ``read_fasta_sequences`` splits each entry at
runs of N and yields the surrounding ACGT segments. Segments below
``min_segment_len`` are filtered out.
"""

from pathlib import Path

import pytest

from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import (
    DEFAULT_MIN_SEGMENT_LEN,
    _split_at_n_runs,
    read_fasta_sequences,
)


def _write_fasta(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "test.fna"
    path.write_text(contents)
    return path


class TestSplitAtNRuns:
    def test_no_n_returns_single_uppercase_segment(self):
        seq = "A" * 200
        assert list(_split_at_n_runs(seq, min_len=100)) == [seq]

    def test_lowercase_input_is_uppercased(self):
        seq = "a" * 200
        assert list(_split_at_n_runs(seq, min_len=100)) == ["A" * 200]

    def test_single_n_run_in_middle_yields_two_segments(self):
        seq = "A" * 150 + "N" * 50 + "C" * 150
        assert list(_split_at_n_runs(seq, min_len=100)) == ["A" * 150, "C" * 150]

    def test_multiple_n_runs_yield_multiple_segments(self):
        seq = "A" * 150 + "NNN" + "C" * 150 + "NN" + "G" * 150
        assert list(_split_at_n_runs(seq, min_len=100)) == ["A" * 150, "C" * 150, "G" * 150]

    def test_segments_below_threshold_are_dropped(self):
        # Middle segment is 50 bp, below the 100 bp threshold.
        seq = "A" * 150 + "N" * 10 + "C" * 50 + "N" * 10 + "G" * 150
        assert list(_split_at_n_runs(seq, min_len=100)) == ["A" * 150, "G" * 150]

    def test_min_len_kwarg_is_respected(self):
        seq = "A" * 30 + "N" * 5 + "C" * 30
        assert list(_split_at_n_runs(seq, min_len=20)) == ["A" * 30, "C" * 30]
        assert list(_split_at_n_runs(seq, min_len=50)) == []

    def test_all_n_yields_nothing(self):
        assert list(_split_at_n_runs("N" * 500, min_len=100)) == []

    def test_empty_input_yields_nothing(self):
        assert list(_split_at_n_runs("", min_len=100)) == []

    def test_leading_and_trailing_ns_are_handled(self):
        # Leading/trailing N runs produce empty splits which the length
        # filter drops.
        seq = "N" * 20 + "A" * 150 + "N" * 20
        assert list(_split_at_n_runs(seq, min_len=100)) == ["A" * 150]


class TestReadFastaSequences:
    def test_single_entry_with_no_n_yields_one_segment(self, tmp_path):
        fasta = _write_fasta(tmp_path, ">chr1\n" + "A" * 200 + "\n")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == ["A" * 200 + "\n"]

    def test_single_entry_with_n_run_yields_split_segments(self, tmp_path):
        seq = "A" * 150 + "N" * 50 + "C" * 150
        fasta = _write_fasta(tmp_path, ">chr1\n" + seq + "\n")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == [
            "A" * 150 + "\n",
            "C" * 150 + "\n",
        ]

    def test_multi_entry_fasta_emits_from_each_entry(self, tmp_path):
        body = ">chr1\n" + "A" * 200 + "\n>chr2\n" + "G" * 200 + "\n"
        fasta = _write_fasta(tmp_path, body)
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == [
            "A" * 200 + "\n",
            "G" * 200 + "\n",
        ]

    def test_multi_line_entry_is_joined_across_lines(self, tmp_path):
        # FASTA convention wraps sequences at ~80 characters; the reader must
        # join all continuation lines of one entry before applying the
        # split-at-N-runs logic.
        body = ">chr1\n" + ("A" * 80 + "\n") * 3
        fasta = _write_fasta(tmp_path, body)
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == ["A" * 240 + "\n"]

    def test_lowercase_in_file_is_uppercased(self, tmp_path):
        fasta = _write_fasta(tmp_path, ">chr1\n" + "a" * 200 + "\n")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == ["A" * 200 + "\n"]

    def test_empty_file_yields_nothing(self, tmp_path):
        fasta = _write_fasta(tmp_path, "")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == []

    def test_header_only_file_yields_nothing(self, tmp_path):
        fasta = _write_fasta(tmp_path, ">chr1\n>chr2\n")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == []

    def test_entry_below_threshold_is_dropped(self, tmp_path):
        # 50 bp entry is shorter than the 100 bp threshold.
        fasta = _write_fasta(tmp_path, ">chr1\n" + "A" * 50 + "\n")
        assert list(read_fasta_sequences(fasta, min_segment_len=100)) == []

    def test_default_min_segment_len_constant_is_100(self):
        assert DEFAULT_MIN_SEGMENT_LEN == 100

    def test_min_segment_len_kwarg_overrides_default(self, tmp_path):
        fasta = _write_fasta(tmp_path, ">chr1\n" + "A" * 50 + "\n")
        # Default (100): dropped. Override to 20: kept.
        assert list(read_fasta_sequences(fasta, min_segment_len=20)) == ["A" * 50 + "\n"]
