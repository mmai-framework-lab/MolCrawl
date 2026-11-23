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


# --- the handled list: the count at the top has to mean what it says ---

def test_handled_reads_ids_and_reasons(tmp_path):
    path = tmp_path / "handled.tsv"
    path.write_text("job_id\twhy\n"
                    "384933\tresubmitted as 385354\n"
                    "384997\tresubmitted as 385358\n"
                    "# a comment\n"
                    "\n", encoding="utf-8")
    assert K.read_handled(str(path)) == {"384933": "resubmitted as 385354",
                                        "384997": "resubmitted as 385358"}


def test_handled_tolerates_no_reason_and_no_file(tmp_path):
    path = tmp_path / "bare.tsv"
    path.write_text("384933\n", encoding="utf-8")
    assert K.read_handled(str(path)) == {"384933": "(理由なし)"}
    assert K.read_handled(str(tmp_path / "absent.tsv")) == {}
    assert K.read_handled(None) == {}


def test_a_handled_job_is_not_counted_among_the_ones_needing_action(tmp_path, capsys,
                                                                    monkeypatch):
    """The defect this closes: the rows were filtered but the headline still said 9."""
    rows = {
        "384933": {"job_id": "384933", "name": "dead-arm-L1", "state": "FAILED",
                   "elapsed": "00:00:29", "end": "2026-11-23T12:46:21"},
        "385999": {"job_id": "385999", "name": "fresh-trouble", "state": "FAILED",
                   "elapsed": "00:01:00", "end": "2026-11-23T14:00:00"},
    }
    monkeypatch.setattr(K, "sacct", lambda ids: rows)
    handled = tmp_path / "handled.tsv"
    handled.write_text("job_id\twhy\n384933\tresubmitted as 385354\n", encoding="utf-8")

    K.main(["--job", "384933", "--job", "385999", "--handled", str(handled)])
    out = capsys.readouterr().out
    assert "★対処が要る: 1 件" in out
    assert "既に対処済み: 1 件" in out
    assert "resubmitted as 385354" in out
    # the real one is still there, and not buried in a standing list
    assert "385999" in out.split("既に対処済み")[0]


def test_without_a_handled_list_nothing_changes(capsys, monkeypatch):
    rows = {"384933": {"job_id": "384933", "name": "dead-arm-L1", "state": "FAILED",
                       "elapsed": "00:00:29", "end": "2026-11-23T12:46:21"}}
    monkeypatch.setattr(K, "sacct", lambda ids: rows)
    K.main(["--job", "384933"])
    out = capsys.readouterr().out
    assert "★対処が要る: 1 件" in out
    assert "既に対処済み" not in out


def test_a_spare_that_is_also_handled_is_counted_once(tmp_path, capsys, monkeypatch):
    """--logs and --handled answer different questions and can name the same job.

    A spare segment is a job that never needed doing anything about; a handled job is
    one somebody has already answered for. Counting a row under both would make the
    totals add up to more than the jobs that exist, so it is taken out as a spare,
    which is the stronger statement.
    """
    rows = {
        "400001": {"job_id": "400001", "name": "arm-L5", "state": "FAILED",
                   "elapsed": "00:00:06", "end": "2026-11-24T09:00:00"},
        "400002": {"job_id": "400002", "name": "arm-L1", "state": "FAILED",
                   "elapsed": "00:00:29", "end": "2026-11-24T09:01:00"},
        "400003": {"job_id": "400003", "name": "arm-L2", "state": "FAILED",
                   "elapsed": "00:05:00", "end": "2026-11-24T09:02:00"},
    }
    monkeypatch.setattr(K, "sacct", lambda ids: rows)
    logs = tmp_path / "logs"
    logs.mkdir()
    # 400001 is a spare and also listed as handled; 400002 is only handled.
    (logs / "arm-L5-400001.out").write_text(
        "it finished the run, so there is nothing to continue\n", encoding="utf-8")
    handled = tmp_path / "handled.tsv"
    handled.write_text("job_id\twhy\n400001\tlisted by mistake\n"
                       "400002\tresubmitted as 400009\n", encoding="utf-8")

    K.main(["--job", "400001", "--job", "400002", "--job", "400003",
            "--logs", str(logs), "--handled", str(handled)])
    out = capsys.readouterr().out
    assert "★対処が要る: 1 件" in out        # only 400003
    assert "既に対処済み: 1 件" in out        # only 400002
    assert "断った: 1 件" in out              # only 400001
    spare_block = out.split("断った")[1]
    assert "400001" in out.split("既に対処済み")[0] or "400001" in spare_block
    assert "400001" not in out.split("既に対処済み")[1].split("断った")[0]
