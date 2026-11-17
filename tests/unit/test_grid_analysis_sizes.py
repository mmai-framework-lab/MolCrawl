"""A size the figures cannot draw has to stop the run, not vanish from the figures.

SIZES listed only small/medium/large, so every xl arm was dropped from all three
relationship figures and from the fit through them, and nothing said so. The
loss-against-compute figure drew three sizes out of a four-size grid, and the
slope of the learning-rate boundary changed from -0.14 to -0.81 once xl was
included. A missing size has to be an error, not a silent omission.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "plot_grid_analysis.py"

HEADER = ("modality\trun\tsize\tlr\tparams\tparams_nonembed\tlast_tokens\t"
          "last_loss\tbest_loss\tcollapsed\tturned_tokens\n")
ROW = "rna\trna-{size}-lr1e4\t{size}\t0.0001\t100\t90\t1000\t2.7\t2.6\t0\t\n"

MATPLOTLIB_MISSING = "No module named 'matplotlib'"


def skip_without_matplotlib(out):
    """The script imports matplotlib at module scope, and the CI unit-test job has
    it nowhere: environment.yaml pins matplotlib, pyproject does not list it. There
    the subprocess dies on the import line, so its stderr says nothing about sizes
    and the assertions below would read that as the guard having failed.

    Decided from the process's own output rather than by probing for the import:
    that covers a matplotlib which is installed but will not load, and it cannot
    disagree with what the subprocess actually did.
    """
    if MATPLOTLIB_MISSING in out.stderr:
        pytest.skip("plot_grid_analysis needs matplotlib, which is not installed here")


def write_summary(tmp_path, sizes):
    p = tmp_path / "summary.tsv"
    p.write_text(HEADER + "".join(ROW.format(size=s) for s in sizes), encoding="utf-8")
    return p


def run(summary, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--summary", str(summary),
         "--config", str(tmp_path / "no-such-config.json"), "--out-dir", str(tmp_path)],
        capture_output=True, text=True)


# The exact stderr the GitHub unit-test job produced on 2026-11-17, kept verbatim:
# the first attempt at this guard probed importlib.util.find_spec instead, and what
# the job actually prints is the only thing that settles whether the skip fires.
CI_IMPORT_FAILURE = (
    "Traceback (most recent call last):\n"
    '  File "/home/runner/work/MolCrawl/MolCrawl/scripts/plot_grid_analysis.py",'
    " line 25, in <module>\n"
    "    import matplotlib\n"
    "ModuleNotFoundError: No module named 'matplotlib'\n"
)


class _Out:
    def __init__(self, stderr):
        self.stderr = stderr


def test_the_skip_fires_on_the_import_failure_ci_actually_printed():
    with pytest.raises(BaseException) as caught:
        skip_without_matplotlib(_Out(CI_IMPORT_FAILURE))
    assert caught.typename == "Skipped"


def test_the_skip_does_not_fire_on_the_guard_message():
    """The guard's own output must reach the assertions, not be skipped past."""
    skip_without_matplotlib(_Out(
        "SystemExit: summary.tsv carries size(s) xxl, which the figures do not draw. "
        "Add them to SIZES and MARKERS in plot_grid_analysis.py"))


def test_xl_is_a_size_the_figures_draw():
    """Read the names rather than import them: importing pulls in matplotlib."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"xl"' in text.split("SIZES = ")[1].split("\n")[0]
    assert '"xl":' in text.split("MARKERS = ")[1].split("\n")[0]


def test_an_unknown_size_stops_the_run(tmp_path):
    out = run(write_summary(tmp_path, ["small", "xxl"]), tmp_path)
    skip_without_matplotlib(out)
    assert out.returncode != 0
    assert "xxl" in out.stderr
    assert "SIZES" in out.stderr


def test_the_four_known_sizes_pass_the_check(tmp_path):
    """It must fail later, on the missing config -- not on the size check."""
    out = run(write_summary(tmp_path, ["small", "medium", "large", "xl"]), tmp_path)
    skip_without_matplotlib(out)
    assert out.returncode != 0
    assert "SIZES" not in out.stderr
