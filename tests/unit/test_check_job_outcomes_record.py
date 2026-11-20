"""A submission record is read whichever shape it has.

The checker exists so a later session can ask how the jobs it submitted ended without
remembering anything about them. That breaks if it also has to remember which column
the id is in: the first submitters wrote `job_id<TAB>label`, later ones write a TSV
whose first column is the time of submission.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import check_job_outcomes as K  # noqa: E402


def test_the_old_two_column_record_is_read(tmp_path):
    path = tmp_path / "old.tsv"
    path.write_text("375244\tbert-small-mb8\n375245\tbert-small-mb16\n", encoding="utf-8")
    assert K.read_record(str(path)) == ["375244", "375245"]


def test_a_record_with_a_header_takes_the_job_id_column(tmp_path):
    path = tmp_path / "new.tsv"
    path.write_text(
        "submitted_at\tarch\tsize\tjob_id\tcommit\n"
        "2026-11-20T19:07:54+09:00\tbert\tsmall\t375244\tdeadbeef\n"
        "2026-11-20T19:07:54+09:00\tgpt2\txl\t375274\tdeadbeef\n",
        encoding="utf-8")
    assert K.read_record(str(path)) == ["375244", "375274"]


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "commented.tsv"
    path.write_text("#! report-tsv 1\n# note: whatever\n\n"
                    "submitted_at\tjob_id\n2026-11-20T19:07:54+09:00\t375293\n",
                    encoding="utf-8")
    assert K.read_record(str(path)) == ["375293"]


def test_a_missing_record_is_empty_rather_than_an_error(tmp_path):
    assert K.read_record(str(tmp_path / "nothing.tsv")) == []
