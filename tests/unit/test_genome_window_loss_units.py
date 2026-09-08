"""Which unit each figure is counted in, and why the two differ.

A raw line is not a contig and not even a whole segment. fasta_to_raw.py splits
a contig at runs of N and wraps any piece longer than RAW_LINE_LEN across several
lines, repeating the contig id. mammal_centered's human assembly is 12,114 lines
carrying 701 contig ids, so counting lines as contigs caps the mean length at
261,120 and dilutes the share yielding no window -- worst exactly where the
assemblies are chromosome-scale.

Window arithmetic stays per line, because raw_to_parquet_single_nuc.py chunks per
line too. That is why the dataset cross-check passes either way and could not
have caught the confusion: it agrees with the pipeline, and the pipeline is also
line-based.
"""
import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "genome_window_loss.py"
_spec = importlib.util.spec_from_file_location("_genome_window_loss", _SRC)
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)

LINE = wl.RAW_LINE_LEN


def _raw(tmp_path, rows, name="a.raw"):
    (tmp_path / name).write_text("".join(f"{cid}\t{'A' * n}\n" for cid, n in rows))
    return str(tmp_path)


def test_a_wrapped_segment_is_one_segment_not_several(tmp_path):
    """Three lines of one wrapped piece must not read as three segments."""
    d = _raw(tmp_path, [("chr1", LINE), ("chr1", LINE), ("chr1", 5000)])

    r = wl.scan(d, 1024)

    assert r["lines"] == 3
    assert r["segments"] == 1
    assert r["bases"] == 2 * LINE + 5000


def test_the_mean_is_not_capped_by_the_wrap(tmp_path):
    """Counting lines would report 175,746; the segment is 527,240 long."""
    d = _raw(tmp_path, [("chr1", LINE), ("chr1", LINE), ("chr1", 5000)])

    r = wl.scan(d, 1024)

    assert r["mean_line_len"] < LINE
    assert r["mean_segment_len"] == 2 * LINE + 5000


def test_two_segments_of_one_contig_stay_separate(tmp_path):
    """N-splitting produces several short lines under the same contig id."""
    d = _raw(tmp_path, [("chr1", 300), ("chr1", 400), ("chr1", 500)])

    r = wl.scan(d, 1024)

    assert r["segments"] == 3


def test_a_contig_change_closes_the_segment(tmp_path):
    """Even mid-wrap, a new id starts a new segment."""
    d = _raw(tmp_path, [("chr1", LINE), ("chr2", 700)])

    r = wl.scan(d, 1024)

    assert r["segments"] == 2


def test_window_arithmetic_stays_per_line(tmp_path):
    """It has to match raw_to_parquet, which chunks each line on its own."""
    d = _raw(tmp_path, [("chr1", 1500), ("chr1", 1500)])

    r = wl.scan(d, 1024)

    # per line: 1 window each, 476 wasted each. Per segment (3000) it would be 2
    # windows and 952 wasted -- the same here, but the units must not be mixed.
    assert r["windows"] == 2
    assert r["remainder_bases"] == 2 * (1500 - 1024)


def test_a_short_line_is_counted_as_a_short_line_not_a_short_segment(tmp_path):
    """The loss breakdown is a line-level statement and stays labelled so."""
    d = _raw(tmp_path, [("chr1", 900)])

    r = wl.scan(d, 1024)

    assert r["lines_too_short"] == 1
    assert r["bases_in_short_lines"] == 900
    assert r["segments_yielding_no_window"] == 1


def test_a_long_segment_yields_windows_even_though_its_last_line_is_short(tmp_path):
    """The wrap's short tail must not make the whole segment look barren."""
    d = _raw(tmp_path, [("chr1", LINE), ("chr1", 10)])

    r = wl.scan(d, 1024)

    assert r["segments"] == 1
    assert r["segments_yielding_no_window"] == 0
