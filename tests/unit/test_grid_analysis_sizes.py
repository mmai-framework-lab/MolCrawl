"""A size the figures cannot draw has to stop the run, not vanish from the figures.

SIZES listed only small/medium/large, so every xl arm was dropped from all three
relationship figures and from the fit through them, and nothing said so. The
loss-against-compute figure drew three sizes out of a four-size grid, and the
slope of the learning-rate boundary changed from -0.14 to -0.81 once xl was
included. A missing size has to be an error, not a silent omission.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "plot_grid_analysis.py"

HEADER = ("modality\trun\tsize\tlr\tparams\tparams_nonembed\tlast_tokens\t"
          "last_loss\tbest_loss\tcollapsed\tturned_tokens\n")
ROW = "rna\trna-{size}-lr1e4\t{size}\t0.0001\t100\t90\t1000\t2.7\t2.6\t0\t\n"

# plot_grid_analysis imports matplotlib at module scope, and the CI unit-test job
# installs it nowhere: environment.yaml pins matplotlib, pyproject does not list it.
# The two tests that run the script are skipped there. The one that reads the names
# out of the file is not, so the regression itself stays covered everywhere.
needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="plot_grid_analysis imports matplotlib, which is not installed here",
)


def write_summary(tmp_path, sizes):
    p = tmp_path / "summary.tsv"
    p.write_text(HEADER + "".join(ROW.format(size=s) for s in sizes), encoding="utf-8")
    return p


def run(summary, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--summary", str(summary),
         "--config", str(tmp_path / "no-such-config.json"), "--out-dir", str(tmp_path)],
        capture_output=True, text=True)


def test_xl_is_a_size_the_figures_draw():
    """Read the names rather than import them: importing pulls in matplotlib."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"xl"' in text.split("SIZES = ")[1].split("\n")[0]
    assert '"xl":' in text.split("MARKERS = ")[1].split("\n")[0]


@needs_matplotlib
def test_an_unknown_size_stops_the_run(tmp_path):
    out = run(write_summary(tmp_path, ["small", "xxl"]), tmp_path)
    assert out.returncode != 0
    assert "xxl" in out.stderr
    assert "SIZES" in out.stderr


@needs_matplotlib
def test_the_four_known_sizes_pass_the_check(tmp_path):
    """It must fail later, on the missing config -- not on the size check."""
    out = run(write_summary(tmp_path, ["small", "medium", "large", "xl"]), tmp_path)
    assert out.returncode != 0
    assert "SIZES" not in out.stderr
