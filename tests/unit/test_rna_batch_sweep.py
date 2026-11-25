"""What the rna batch-size grid of 2026-11-20 must not get wrong.

1. The arms derived from the directive's rules match the tables the same directive
   prints. A grid of 68 arms is built from a handful of numbers, and a rate or a
   step count copied into the wrong row is invisible once the runs start.
2. The budget is spent in blocks, so every batch runs the same data and every arm
   carries 225 evaluation points at the same block positions.
3. Each arm's micro-batch, accumulation count and rank count multiply out to its
   batch -- in BERT's arithmetic, which includes the GPU count, and in GPT-2's,
   which does not.
4. The collapse detector's grace window is in steps and is compared against the
   absolute step count, so one number cannot serve five batches that differ
   16-fold. It has to come out to the same block position on every arm.
5. The generated configs declare what the directive fixes: the judged metric, the
   fixed evaluation mask, one evaluation micro-batch, expected_global_batch, and
   early stopping off. None of them declares the eval subset seed, which lives in
   one constant in main.py and would move an arm's rows if a config named it.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import rna_batch_sweep_arms as A  # noqa: E402
import rna_batch_sweep_configs as C  # noqa: E402

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                           "molcrawl", "tasks", "pretrain", "configs", "rna")


@pytest.fixture(params=sorted(A.MICRO_BATCH))
def inventory(request):
    return request.param, A.arms(request.param, None, False)


def test_the_derived_arms_match_the_printed_tables(inventory):
    which, rows = inventory
    problems = A.check(rows)
    assert problems == [], f"micro-batch set {which}: " + "; ".join(problems)


def test_there_are_sixty_eight_arms(inventory):
    _, rows = inventory
    assert len(rows) == 68
    assert sum(1 for r in rows if r["arch"] == "bert") == 48
    assert sum(1 for r in rows if r["arch"] == "gpt2") == 20


def test_every_arm_spends_the_same_blocks(inventory):
    _, rows = inventory
    for row in rows:
        assert int(row["blocks"]) == A.BUDGET_BLOCKS
        assert int(row["steps"]) * int(row["batch"]) == A.BUDGET_BLOCKS
        assert int(row["eval_points"]) == 225
        assert int(row["eval_interval"]) * int(row["batch"]) == A.EVAL_EVERY_BLOCKS


def test_the_shape_multiplies_out_to_the_batch(inventory):
    _, rows = inventory
    for row in rows:
        product = int(row["micro_batch"]) * int(row["grad_accum"]) * int(row["world_size"])
        assert product == int(row["batch"]), row["config"]


def test_one_gpu_block_count_is_held_across_the_batches_of_a_size(inventory):
    _, rows = inventory
    for arch in A.ARCHES:
        for size in A.SIZES:
            micros = {int(r["micro_batch"]) for r in rows
                      if r["arch"] == arch and r["size"] == size}
            assert len(micros) == 1, f"{arch} {size} changes micro-batch between batches"


def test_every_candidate_micro_batch_can_build_the_smallest_batch():
    for which, table in A.MICRO_BATCH.items():
        for arch, sizes in table.items():
            for size, micro in sizes.items():
                assert min(A.BATCHES) % (micro * A.GPU_COUNT) == 0, \
                    f"{which}: {arch} {size} micro {micro} cannot build 256"


def test_the_unit_multiplier_arm_recovers_the_adopted_rate():
    for arch in A.ARCHES:
        for size in A.SIZES:
            got = A.scaled_rate(arch, size, A.BASE_BATCH, 1.0)
            assert got == pytest.approx(A.BASE_LR_2560[arch][size], rel=5e-3)


def test_gpt2_floors_and_decays_on_its_own_run(inventory):
    _, rows = inventory
    for row in rows:
        if row["arch"] != "gpt2":
            continue
        assert int(row["lr_decay_iters"]) == int(row["steps"])
        assert float(row["min_lr"]) == pytest.approx(float(row["lr"]) / 10, rel=1e-6)


def test_bert_is_judged_on_the_mask_loss_at_one_evaluation_batch(inventory):
    _, rows = inventory
    bert = [r for r in rows if r["arch"] == "bert"]
    assert {r["judge_metric"] for r in bert} == {"eval_loss_mask"}
    assert {int(r["eval_batch"]) for r in bert} == {A.BERT_EVAL_BATCH}
    assert {r["judge_metric"] for r in rows if r["arch"] == "gpt2"} == {"val_loss"}


def test_the_grace_window_is_the_same_block_position_on_every_arm():
    grace_blocks = 25_165_824  # a block position, whatever it is measured to be
    rows = A.arms("current", grace_blocks, True)
    for row in rows:
        if row["arch"] != "bert":
            assert row["degenerate_grace_steps"] == ""
            continue
        position = int(row["degenerate_grace_steps"]) * int(row["batch"])
        assert position == grace_blocks, row["config"]


def test_the_approved_grace_position_divides_every_batch_and_clears_the_slowest_arm():
    """The position approved on 2026-11-20, and why that number and not another.

    Measured (job 374267): of the four adopted arms the last to fall below the 9.0
    threshold is xl, at 6,656,000 blocks. The grace position is twice that, so a
    healthy arm is well clear of it, and it divides all five batches exactly, so
    every arm waits out the same share of its own schedule.
    """
    slowest_healthy_crossing = 6_656_000
    assert A.GRACE_BLOCKS == 13_312_000
    assert A.GRACE_BLOCKS >= 2 * slowest_healthy_crossing

    shares = set()
    for row in A.arms("current", A.GRACE_BLOCKS, True):
        if row["arch"] != "bert":
            continue
        batch, grace, steps = (int(row["batch"]), int(row["degenerate_grace_steps"]),
                               int(row["steps"]))
        assert grace * batch == A.GRACE_BLOCKS, row["config"]
        assert grace > slowest_healthy_crossing // batch, row["config"]
        shares.add(round(grace / steps, 6))
    assert len(shares) == 1, shares


def test_the_detector_refuses_to_be_written_without_a_block_position():
    rows = A.arms("current", None, True)
    assert {r["degenerate_grace_steps"] for r in rows} == {""}


def test_every_arm_has_a_parent_config_on_disk():
    for (arch, size), parent in C.PARENT.items():
        path = os.path.join(CONFIGS_DIR, f"{parent}.py")
        assert os.path.exists(path), f"{arch} {size}: no {parent}.py to import"


def test_the_rendered_config_declares_what_the_directive_fixes():
    rows = A.arms("current", 98_304, True)
    for row in rows:
        text = C.render(row)
        assert f"from molcrawl.tasks.pretrain.configs.rna.{C.PARENT[(row['arch'], row['size'])]} import *" in text
        assert f"expected_global_batch = {row['batch']}" in text
        assert "early_stopping = False" in text
        assert "eval_subset_seed" not in text
        assert f"batch_size = {row['micro_batch']}\n" in text
        assert f"gradient_accumulation_steps = {row['grad_accum']}\n" in text
        if row["arch"] == "bert":
            assert 'judge_on = "eval_loss_mask"' in text
            assert 'metric_for_best_model = "eval_loss_mask"' in text
            assert "fixed_eval_mask = True" in text
            assert "eval_mask_seed = 42" in text
            assert f"per_device_eval_batch_size = {row['eval_batch']}" in text
            assert f"max_steps = {row['steps']}" in text
            assert f"save_steps = {row['save_interval']}" in text
            assert "save_on_improve = True" in text
            assert f"degenerate_loss_threshold = {A.DEGENERATE_THRESHOLD}" in text
            assert f'-b{row["batch"]}-lr{row["lr_slug"]}' in text
        else:
            assert f"max_iters = {row['steps']}" in text
            assert f"lr_decay_iters = {row['steps']}" in text
            assert "degenerate_loss_threshold" not in text
            assert row["out_dir"] in text


def test_a_config_without_the_detector_does_not_mention_it():
    rows = A.arms("current", None, False)
    for row in rows:
        assert "degenerate_loss_threshold" not in C.render(row)


def test_the_rendered_rate_is_the_rate_in_the_inventory():
    for row in A.arms("current", None, False):
        text = C.render(row)
        assert f"learning_rate = {float(row['lr']):.6g}\n" in text


# --- the measured block counts, which arrive from the probe rather than from source ---

PROBE_COLUMNS = ["arch", "size", "micro", "global", "accum", "job_id", "status",
                 "s_per_step", "s_per_step_at_2560", "log"]


def _probe_table(tmp_path, measurements):
    """A table in the shape scripts/rna_microbatch_probe_table.py writes."""
    import csv as _csv

    path = tmp_path / "probe-table.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("#! report-tsv 1\n")
        writer = _csv.DictWriter(handle, fieldnames=PROBE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for arch, size, micro, rate in measurements:
            writer.writerow({
                "arch": arch, "size": size, "micro": micro, "global": 256,
                "accum": 256 // (micro * 4), "job_id": "0",
                "status": "OK" if rate else "FAILED (OOM)",
                "s_per_step": f"{rate / 10:.3f}" if rate else "",
                "s_per_step_at_2560": f"{rate:.3f}" if rate else "",
                "log": "x.out"})
    return str(path)


def test_the_fastest_shape_is_chosen_and_a_shape_that_did_not_fit_is_not(tmp_path):
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    path = _probe_table(tmp_path, [
        ("bert", "small", 8, 4.48), ("bert", "small", 16, 3.10),
        ("bert", "small", 32, 2.95), ("bert", "small", 64, 0.0),
    ])
    chosen = probe.fastest_micro(probe.read_table(path))
    assert chosen["bert"]["small"] == 32


def test_a_tie_takes_the_smaller_block_count(tmp_path):
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    path = _probe_table(tmp_path, [
        ("gpt2", "xl", 4, 12.77), ("gpt2", "xl", 8, 12.77),
    ])
    assert probe.fastest_micro(probe.read_table(path))["gpt2"]["xl"] == 4


def test_the_measured_set_builds_arms_with_the_measured_shape(tmp_path):
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    rows = [(arch, size, micro, 10.0 - micro / 100)
            for arch in ("bert", "gpt2")
            for size in A.SIZES
            for micro in (8, 16)]
    measured = probe.fastest_micro(probe.read_table(_probe_table(tmp_path, rows)))
    assert all(measured[a][s] == 16 for a in ("bert", "gpt2") for s in A.SIZES)

    A.MICRO_BATCH["measured"] = measured
    try:
        arms = A.arms("measured", A.GRACE_BLOCKS, True)
        assert len(arms) == 68
        for row in arms:
            assert int(row["micro_batch"]) == 16, row["config"]
            product = (int(row["micro_batch"]) * int(row["grad_accum"])
                       * int(row["world_size"]))
            assert product == int(row["batch"]), row["config"]
    finally:
        del A.MICRO_BATCH["measured"]


def test_two_tables_merge_with_the_later_one_winning(tmp_path):
    """BERT was measured on 2026-11-20 and GPT-2 on 2026-11-23, so the set is two files.

    The GPT-2 rows of the first table are the contaminated ones; passing the newer
    table second has to replace them rather than race with them.
    """
    import subprocess

    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    first = _probe_table(tmp_path / "a", [
        (arch, size, micro, 10.0 - micro / 100)
        for arch in ("bert", "gpt2") for size in A.SIZES for micro in (8, 16)])
    second = _probe_table(tmp_path / "b", [
        ("gpt2", size, micro, 10.0 - micro / 1000)
        for size in A.SIZES for micro in (32, 64)])
    out = tmp_path / "arms.tsv"
    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    done = subprocess.run(
        [sys.executable, os.path.join(repo, "scripts", "rna_batch_sweep_arms.py"),
         "--micro-batch-set", "measured", "--measured-from", first,
         "--measured-from", second, "--out", str(out)],
        cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rows = list(csv.DictReader(
        (ln for ln in open(out, encoding="utf-8") if not ln.startswith("#")),
        delimiter="\t"))
    assert len(rows) == 68
    by_arch = {r["arch"]: int(r["micro_batch"]) for r in rows}
    assert by_arch["bert"] == 16     # only the first table measured BERT
    assert by_arch["gpt2"] == 64     # the second table replaced its rows


def test_a_framework_no_table_measured_is_refused(tmp_path):
    import subprocess

    (tmp_path / "c").mkdir()
    only_bert = _probe_table(tmp_path / "c", [
        ("bert", size, micro, 5.0) for size in A.SIZES for micro in (8, 16)])
    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    done = subprocess.run(
        [sys.executable, os.path.join(repo, "scripts", "rna_batch_sweep_arms.py"),
         "--micro-batch-set", "measured", "--measured-from", only_bert],
        cwd=repo, capture_output=True, text=True)
    assert done.returncode == 2
    assert "gpt2" in done.stderr


def _probe_runs(shapes):
    """Rows in the shape main() builds, before the repeats are collapsed."""
    return [{"arch": arch, "size": size, "micro": micro, "repeat": str(repeat),
             "global": 256, "accum": 256 // (micro * 4), "job_id": "0",
             "status": "OK", "s_per_step": f"{rate / 10:.3f}",
             "s_per_step_at_2560": f"{rate:.3f}", "log": "x.out"}
            for arch, size, micro, repeat, rate in shapes]


def test_the_printed_rate_is_the_shape_s_own_and_not_the_first_run_s():
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    # Every shape has one quiet run, but never the first one listed.
    runs = _probe_runs([
        ("gpt2", "xl", 4, 1, 11.85), ("gpt2", "xl", 4, 2, 13.76),
        ("gpt2", "xl", 8, 1, 10.31), ("gpt2", "xl", 8, 2, 10.23),
        ("gpt2", "xl", 16, 1, 11.20), ("gpt2", "xl", 16, 2, 9.61),
    ])
    shapes = probe.collapse_repeats(runs)
    cells = probe.rates_by_micro(shapes, "gpt2", "xl")

    assert cells == {4: 11.85, 8: 10.23, 16: 9.61}
    # The choice and the cell it is read off have to be the same figure.
    chosen = probe.fastest_micro(shapes)["gpt2"]["xl"]
    assert chosen == 16
    assert cells[chosen] == min(cells.values())


def test_a_block_count_that_did_not_fit_is_told_apart_from_one_never_measured():
    import importlib

    probe = importlib.import_module("rna_microbatch_probe_table")
    runs = _probe_runs([("bert", "small", 16, 1, 3.10)])
    oom = dict(runs[0], micro=64, s_per_step="", s_per_step_at_2560="",
               status="FAILED (OOM)")
    shapes = probe.collapse_repeats(runs + [oom])

    # Collapsing drops the arm with no rate, so it must not show up as a figure,
    # and the whole record is what says it was measured at all.
    assert probe.rates_by_micro(shapes, "bert", "small") == {16: 3.10}
    assert [r for r in runs + [oom] if r["micro"] == 64]
