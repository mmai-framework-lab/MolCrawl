"""The launch plan of workflows/submit-rna-batch-sweep.sh, checked without queueing.

The submitter runs days after the probe, with nobody watching the step that joins the
two files. A mistake in that join is a wrong link count on 68 chains, which shows up
as arms that stop short of the budget. So the plan is exercised here against a
synthetic table, with DRY_RUN set, and the link counts are checked against the
arithmetic the directive fixes.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import rna_batch_sweep_arms as A  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBMITTER = os.path.join(REPO, "workflows", "submit-rna-batch-sweep.sh")
#: One arm of batch 2,560 would run this many steps for the whole budget, and the
#: wall clock of every arm is this times its rate whatever its batch.
REFERENCE_STEPS = 120_960
SEGMENT_HOURS = 96


def _write(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def plan(tmp_path):
    """The plan the submitter prints for the whole grid at a known set of rates."""
    rates = {("bert", "small"): 4.48, ("bert", "medium"): 7.55,
             ("bert", "large"): 12.12, ("bert", "xl"): 21.69,
             ("gpt2", "small"): 9.63, ("gpt2", "medium"): 8.31,
             ("gpt2", "large"): 9.60, ("gpt2", "xl"): 12.77}
    arm_rows = A.arms("current", A.GRACE_BLOCKS, True)
    arms_tsv = _write(tmp_path / "arms.tsv", list(arm_rows[0]), arm_rows)

    probe_rows = [{
        "arch": arch, "size": size,
        "micro": A.MICRO_BATCH["current"][arch][size],
        "global": 256, "accum": 0, "job_id": "0", "status": "OK",
        "s_per_step": f"{rate / 10:.3f}", "s_per_step_at_2560": f"{rate:.3f}",
        "log": "x.out",
    } for (arch, size), rate in rates.items()]
    probe_tsv = _write(tmp_path / "probe.tsv", list(probe_rows[0]), probe_rows)

    out = subprocess.run(
        ["bash", SUBMITTER], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "ARMS": arms_tsv, "PROBE_TABLE": probe_tsv,
             "LEARNING_SOURCE_DIR": str(tmp_path), "MODEL_OUTPUT_ROOT": str(tmp_path),
             "SLURM_ACCOUNT": "test", "DRY_RUN": "1"})
    assert out.returncode == 0, out.stderr
    return rates, out.stdout


def test_the_plan_covers_all_sixty_eight_arms(plan):
    _, text = plan
    assert re.search(r"=== rna batch-size grid: 68 arms,", text), text[:400]
    assert len(re.findall(r"^  (bert|gpt2) ", text, re.M)) == 68


def test_each_arm_gets_the_links_its_measured_rate_needs(plan):
    rates, text = plan
    seen = 0
    for line in text.splitlines():
        hit = re.match(r"^  (bert|gpt2) (\w+)-b(\d+)-lr\S+: (\d+) link\(s\), ~(\d+) h",
                       line)
        if not hit:
            continue
        arch, size, _, links, hours = (hit.group(1), hit.group(2), hit.group(3),
                                       int(hit.group(4)), int(hit.group(5)))
        want_hours = REFERENCE_STEPS * rates[(arch, size)] / 3600
        assert abs(hours - want_hours) <= 1, line
        need = -(-want_hours // SEGMENT_HOURS)        # ceiling
        assert links == need + 1, f"{line}: expected {need:.0f} + 1 spare"
        seen += 1
    assert seen == 68


def test_the_wall_clock_does_not_depend_on_the_batch(plan):
    """Steps fall as the batch rises and the step time rises with it."""
    _, text = plan
    by_arm: dict[tuple[str, str], set[int]] = {}
    for line in text.splitlines():
        hit = re.match(r"^  (bert|gpt2) (\w+)-b(\d+)-lr\S+: \d+ link\(s\), ~(\d+) h",
                       line)
        if hit:
            by_arm.setdefault((hit.group(1), hit.group(2)), set()).add(int(hit.group(4)))
    assert by_arm
    for key, hours in by_arm.items():
        assert len(hours) == 1, f"{key}: {hours}"


def test_a_filter_narrows_the_plan(tmp_path):
    rows = A.arms("current", A.GRACE_BLOCKS, True)
    arms_tsv = _write(tmp_path / "arms.tsv", list(rows[0]), rows)
    probe_rows = [{"arch": a, "size": s,
                   "micro": A.MICRO_BATCH["current"][a][s], "global": 256,
                   "accum": 0, "job_id": "0", "status": "OK", "s_per_step": "1.0",
                   "s_per_step_at_2560": "10.0", "log": "x.out"}
                  for a in ("bert", "gpt2") for s in A.SIZES]
    probe_tsv = _write(tmp_path / "probe.tsv", list(probe_rows[0]), probe_rows)
    out = subprocess.run(
        ["bash", SUBMITTER], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "ARMS": arms_tsv, "PROBE_TABLE": probe_tsv,
             "LEARNING_SOURCE_DIR": str(tmp_path), "MODEL_OUTPUT_ROOT": str(tmp_path),
             "SLURM_ACCOUNT": "test", "DRY_RUN": "1",
             "ONLY_ARCH": "gpt2", "ONLY_BATCHES": "256 4096"})
    assert out.returncode == 0, out.stderr
    assert re.search(r"grid: 8 arms,", out.stdout), out.stdout[:400]
    assert " bert " not in out.stdout


def test_a_dry_run_says_which_configs_are_not_written_yet(tmp_path):
    """The plan is read before the configs exist; that is what it is for.

    The config column is pointed at a name nothing writes, because the real ones are
    on disk once the grid has been generated -- this test read "the configs do not
    exist yet" as the state of the repository and started failing the day they did.
    """
    rows = [dict(r, config="molcrawl/tasks/pretrain/configs/rna/_absent_probe_arm.py")
            for r in A.arms("current", A.GRACE_BLOCKS, True)
            if r["arch"] == "gpt2" and r["size"] == "small"]
    arms_tsv = _write(tmp_path / "arms.tsv", list(rows[0]), rows)
    probe_rows = [{"arch": "gpt2", "size": "small",
                   "micro": A.MICRO_BATCH["current"]["gpt2"]["small"], "global": 256,
                   "accum": 0, "job_id": "0", "status": "OK", "s_per_step": "1.0",
                   "s_per_step_at_2560": "10.0", "log": "x.out"}]
    probe_tsv = _write(tmp_path / "probe.tsv", list(probe_rows[0]), probe_rows)
    out = subprocess.run(
        ["bash", SUBMITTER], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "ARMS": arms_tsv, "PROBE_TABLE": probe_tsv,
             "LEARNING_SOURCE_DIR": str(tmp_path), "MODEL_OUTPUT_ROOT": str(tmp_path),
             "SLURM_ACCOUNT": "test", "DRY_RUN": "1"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("(not written yet:") == len(rows)


def test_an_arm_with_no_measured_rate_stops_the_plan(tmp_path):
    rows = A.arms("current", A.GRACE_BLOCKS, True)
    arms_tsv = _write(tmp_path / "arms.tsv", list(rows[0]), rows)
    probe_rows = [{"arch": "bert", "size": "small",
                   "micro": A.MICRO_BATCH["current"]["bert"]["small"], "global": 256,
                   "accum": 0, "job_id": "0", "status": "OK", "s_per_step": "1.0",
                   "s_per_step_at_2560": "10.0", "log": "x.out"}]
    probe_tsv = _write(tmp_path / "probe.tsv", list(probe_rows[0]), probe_rows)
    out = subprocess.run(
        ["bash", SUBMITTER], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "ARMS": arms_tsv, "PROBE_TABLE": probe_tsv,
             "LEARNING_SOURCE_DIR": str(tmp_path), "MODEL_OUTPUT_ROOT": str(tmp_path),
             "SLURM_ACCOUNT": "test", "DRY_RUN": "1"})
    assert out.returncode != 0
    assert "NO RATE" in out.stderr


def test_the_probe_table_and_the_arms_agree_on_the_accumulation():
    """§9 is printed twice, by two scripts, and the two must not disagree.

    They did: the probe table applied BERT's rule to both frameworks. BERT reaches its
    batch as micro x accumulation x world_size and nanoGPT as micro x accumulation,
    because it divides the accumulation across the ranks, so one rule understates
    GPT-2's accumulation by the rank count. The configs come from the arms script, so
    the table was the wrong one -- and a reader comparing a config against that table
    would have found a fourfold difference and no way to tell which was right.
    """
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    for row in A.arms("current", A.GRACE_BLOCKS, True):
        arch, micro, batch = row["arch"], int(row["micro_batch"]), int(row["batch"])
        assert batch % (micro * probe.GPU_COUNT) == 0, row["config"]
        assert batch // (micro * probe.ACCUMULATION_DIVISOR[arch]) \
            == int(row["grad_accum"]), row["config"]


# --- the re-measurement: repeats, and the check the first probe failed ---

def _probe_rows(measurements):
    """Rows in the shape the probe table writes, one per (shape, repeat)."""
    return [{"arch": arch, "size": size, "micro": str(micro), "global": "256",
             "accum": "0", "job_id": "0", "status": "OK", "repeat": str(rep),
             "s_per_step": f"{rate / 10:.3f}", "s_per_step_at_2560": f"{rate:.3f}",
             "log": "x.out"}
            for arch, size, micro, rep, rate in measurements]


def test_repeats_collapse_to_the_fastest_run_of_each_shape():
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    rows = probe.collapse_repeats(_probe_rows([
        ("gpt2", "small", 32, 1, 9.10),   # measured against competition
        ("gpt2", "small", 32, 2, 4.97),
        ("gpt2", "small", 32, 3, 5.20),
        ("gpt2", "small", 64, 1, 5.50),
    ]))
    by = {(r["size"], int(r["micro"])): r for r in rows}
    assert len(rows) == 2
    assert float(by[("small", 32)]["s_per_step_at_2560"]) == 4.97
    assert by[("small", 32)]["repeats"] == 3
    assert by[("small", 32)]["spread"] == "1.83"   # flagged: never measured quietly
    assert by[("small", 64)]["repeats"] == 1


def test_a_bigger_model_cannot_be_faster_than_a_smaller_one():
    """The 2026-11-20 probe put the 328M model 2.3x ahead of the 104M one."""
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    rows = _probe_rows([("gpt2", "small", 32, 1, 6.29),
                        ("gpt2", "medium", 64, 1, 2.73)])
    problems = probe.size_order_holds({"gpt2": {"small": 32, "medium": 64}}, rows)
    assert len(problems) == 1
    assert "medium" in problems[0] and "2.30x" in problems[0]

    ok = _probe_rows([("gpt2", "small", 32, 1, 4.97),
                      ("gpt2", "medium", 64, 1, 5.80)])
    assert probe.size_order_holds({"gpt2": {"small": 32, "medium": 64}}, ok) == []


def test_the_size_order_gate_reads_the_chosen_shape_not_the_fastest_row():
    """A size is judged on the block count it will run, not its best measurement."""
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    rows = _probe_rows([("gpt2", "small", 32, 1, 4.97),
                        ("gpt2", "medium", 8, 1, 9.00),
                        ("gpt2", "medium", 64, 1, 2.00)])
    assert probe.size_order_holds({"gpt2": {"small": 32, "medium": 8}}, rows) == []
    assert probe.size_order_holds({"gpt2": {"small": 32, "medium": 64}}, rows) != []
