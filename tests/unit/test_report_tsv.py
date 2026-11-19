"""The writer that puts the column convention into a report TSV.

Every case here is a rule from docs/07-reports/report_tsv_columns.md that a
script can break without noticing: a name that does not declare the convention,
a `value` column with nothing saying what the value is, a missing number written
as `NaN`, CRLF from csv's default line terminator. The point of the module is
that the rule is checked where the file is written, so the rules are tested at
that boundary rather than on the files afterwards.
"""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "report_tsv.py"
    spec = importlib.util.spec_from_file_location("report_tsv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prov(mod):
    return {"model": "compounds bert small",
            "source": "a-run/checkpoint-45000/trainer_state.json",
            "generated": mod.stamp("scripts/example.py")}


def body(path: Path):
    """The lines below the provenance block."""
    return [l for l in path.read_text(encoding="utf-8").split("\n")
            if l and not l.startswith("#")]


def test_the_name_declares_the_convention(mod, tmp_path, prov):
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "curve.tsv", ["step"], [{"step": 1}], prov)
    assert not (tmp_path / "curve.tsv").exists()


def test_the_version_in_the_name_and_in_the_body_agree(mod, tmp_path, prov):
    """R13. Both are written from one constant, so they cannot drift apart."""
    out = tmp_path / f"rt{mod.VERSION}-curve.tsv"
    mod.write_tsv(out, ["step"], [{"step": 1}], prov)
    assert out.read_text(encoding="utf-8").split("\n")[0] == f"#! report-tsv {mod.VERSION}"


@pytest.mark.parametrize("missing", ["model", "source", "generated"])
def test_the_three_required_provenance_keys(mod, tmp_path, prov, missing):
    del prov[missing]
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step"], [{"step": 1}], prov)


def test_a_provenance_value_that_claims_to_be_unknown_is_refused(mod, tmp_path, prov):
    """R15. `不明` is what the files written before the convention say."""
    prov["model"] = "不明"
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step"], [{"step": 1}], prov)


def test_a_value_column_needs_a_metric_column(mod, tmp_path, prov):
    """R12, which is what the curve tables written before 2026-11-10 broke."""
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step", "value"],
                      [{"step": 1, "value": 2}], prov)


def test_a_value_column_with_a_metric_column_is_allowed(mod, tmp_path, prov):
    out = tmp_path / "rt1-x.tsv"
    mod.write_tsv(out, ["step", "metric", "value"],
                  [{"step": 1, "metric": "val_loss", "value": 2}], prov)
    assert body(out)[0] == "step\tmetric\tvalue"


def test_a_repeated_column_name_is_refused(mod, tmp_path, prov):
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step", "step"], [], prov)


def test_a_column_name_outside_ascii_is_refused(mod, tmp_path, prov):
    """R3. A Japanese column name defeats grep and splits the wording in two."""
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["段"], [{"段": 1}], prov)


def test_a_missing_value_is_an_empty_cell(mod, tmp_path, prov):
    out = tmp_path / "rt1-x.tsv"
    mod.write_tsv(out, ["step", "train_loss"], [{"step": 1, "train_loss": None}], prov)
    assert body(out)[1] == "1\t"


def test_a_column_a_row_does_not_carry_is_an_empty_cell(mod, tmp_path, prov):
    out = tmp_path / "rt1-x.tsv"
    mod.write_tsv(out, ["step", "train_loss"], [{"step": 1}], prov)
    assert body(out)[1] == "1\t"


@pytest.mark.parametrize("marker", ["NaN", "NA", "None", "-", "null"])
def test_a_missing_value_marker_is_refused(mod, tmp_path, prov, marker):
    """R10. Each of these reads as a string to one reader and a number to another."""
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step", "train_loss"],
                      [{"step": 1, "train_loss": marker}], prov)


def test_a_tab_inside_a_field_is_refused(mod, tmp_path, prov):
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["note"], [{"note": "a\tb"}], prov)


def test_a_row_with_a_field_the_header_lacks_is_refused(mod, tmp_path, prov):
    """R2. Silently dropping it would lose a column nobody then looks for."""
    with pytest.raises(SystemExit):
        mod.write_tsv(tmp_path / "rt1-x.tsv", ["step"], [{"step": 1, "lr": "1e-4"}], prov)


def test_no_bom_and_lf_endings(mod, tmp_path, prov):
    """R5 and R6. csv writes CRLF unless told otherwise, and a BOM lands inside
    the first column's name."""
    out = tmp_path / "rt1-x.tsv"
    mod.write_tsv(out, ["step"], [{"step": 1}, {"step": 2}], prov)
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"2\n")


def test_the_block_is_written_in_the_conventions_order(mod, prov):
    prov["job"] = "123456"
    prov["note"] = "something"
    keys = [l[2:].split(":")[0] for l in mod.provenance_block(prov) if l.startswith("# ")]
    assert keys == ["model", "source", "job", "generated", "note"]


def test_two_notes_become_two_lines(mod, prov):
    prov["note"] = ["first", "second"]
    lines = mod.provenance_block(prov)
    assert lines.count("# note: first") == 1
    assert lines.count("# note: second") == 1


def test_an_optional_key_with_nothing_in_it_is_left_out(mod, prov):
    """Outside a Slurm job there is no job id, and `# job: None` would claim one."""
    prov["job"] = None
    assert not [l for l in mod.provenance_block(prov) if l.startswith("# job:")]


def test_the_stamp_names_the_script_and_the_offset(mod):
    moment = datetime.datetime(2026, 11, 10, 15, 4, 5, tzinfo=mod.JST)
    assert mod.stamp("scripts/x.py", now=moment) == "2026-11-10T15:04:05+09:00 by scripts/x.py"


def test_write_tsv_returns_the_number_of_data_rows(mod, tmp_path, prov):
    assert mod.write_tsv(tmp_path / "rt1-x.tsv", ["step"],
                         [{"step": 1}, {"step": 2}], prov) == 2
