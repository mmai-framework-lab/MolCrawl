"""What the §5 GPT-2 seed inventory must not get wrong.

1. The config's declaration is not in the manifest. It is in the job log, where the
   configurator echoed the file, and that echo is repeated once per rank; one value
   has to come out of several identical copies, and two different values have to be
   refused rather than averaged or taken first.
2. A seed given on the command line beat the config. The row says so in its own
   column instead of letting `config_seed` stand for what the run used.
3. `manual_seed_rank0` is the value rank 0 was seeded with; the per-rank rule lives
   in its own column, not in four rows per run.
4. A run with no manifest records no seed at all, and says so.
5. The file follows the report TSV convention: version line, provenance with
   `model`/`source`/`generated`, ASCII column names, no BOM, LF endings.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from gpt2_seed_inventory import (  # noqa: E402
    COLUMNS,
    index_logs,
    inventory_row,
    provenance,
    read_declared_seed,
    write_tsv,
)

CONFIG_PATH = ("/data1/x/MolCrawl-wt-genome/molcrawl/tasks/pretrain/configs/"
               "genome_sequence/gpt2_small_subset.py")


def write_log(path, ranks=2, config_seed=45, cli_seed=None, config=CONFIG_PATH):
    """A log shaped like the real ones: the config echoed once per rank."""
    lines = []
    for _ in range(ranks):
        if config is not None:
            lines.append(f"Overriding config with {config}:")
            lines.append('"""Reference config for training GPT-2."""')
            lines.append("# Training seed (sequentially assigned 2026-08-03).")
            if config_seed is not None:
                lines.append(f"seed = {config_seed}")
            lines.append("    seed = 9  # inside a conditional, not the declaration")
            lines.append("data_seed = 7")
    if cli_seed is not None:
        lines.append(f"Overriding: seed = {cli_seed}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def test_the_config_echo_repeated_per_rank_yields_one_value(tmp_path):
    log = write_log(str(tmp_path / "logs" / "mc-gen-gpt2-152868.out"), ranks=4)
    config, config_seed, cli_seed, source = read_declared_seed([log])
    assert config_seed == 45
    assert cli_seed is None
    assert source == "job log"
    assert config == "genome_sequence/gpt2_small_subset.py"


def test_an_indented_seed_and_a_data_seed_are_not_the_declaration(tmp_path):
    log = write_log(str(tmp_path / "logs" / "mc-x-1.out"), ranks=1, config_seed=45)
    _config, config_seed, _cli, _source = read_declared_seed([log])
    assert config_seed == 45  # not 9 from the indented line, not 7 from data_seed


def test_two_different_seeds_in_one_log_are_refused(tmp_path):
    path = str(tmp_path / "logs" / "mc-x-2.out")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Overriding config with {CONFIG_PATH}:\nseed = 45\nseed = 9\n")
    _config, config_seed, _cli, source = read_declared_seed([path])
    assert config_seed is None
    assert "more than one seed" in source


def test_a_command_line_seed_is_its_own_column(tmp_path):
    log = write_log(str(tmp_path / "logs" / "mc-x-3.out"), config_seed=45, cli_seed=1001)
    _config, config_seed, cli_seed, _source = read_declared_seed([log])
    assert (config_seed, cli_seed) == (45, 1001)


def test_no_log_is_said_rather_than_guessed():
    _config, config_seed, cli_seed, source = read_declared_seed([])
    assert (config_seed, cli_seed) == (None, None)
    assert source == "no job log found"


def test_the_job_id_is_read_off_the_log_name(tmp_path):
    write_log(str(tmp_path / "mc-gen-gpt2-ep9-global_random_seed2-152868.out"))
    write_log(str(tmp_path / "not-a-job.out.gz"))
    index = index_logs([str(tmp_path)])
    assert set(index) == {"152868"}


def write_manifest(run_dir, used_seed=None, job_id="152868", lr=None):
    os.makedirs(run_dir, exist_ok=True)
    manifest = {"run": {"job_id": job_id}, "schedule": {}, "sources": {}}
    if lr is not None:
        manifest["schedule"]["learning_rate"] = lr
    if used_seed is not None:
        manifest["seed"] = {"seed": used_seed, "per_rank_offset": "seed + ddp_rank"}
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def test_the_row_puts_the_rank_rule_in_its_own_column(tmp_path):
    run_dir = str(tmp_path / "gpt2-small-global_random_seed2-ep9")
    write_manifest(run_dir, used_seed=45, lr=0.0001)
    log = write_log(str(tmp_path / "logs" / "mc-x-152868.out"), config_seed=45)
    row = inventory_row(
        {"modality": "genome_sequence", "run": "gpt2-small-global_random_seed2-ep9",
         "run_dir": run_dir, "job_id": "152868", "size": "small",
         "learning_rate": 0.0001, "steps_run": 150319},
        "learning_source_genome_runs", index_logs([os.path.dirname(log)]))
    assert row["manual_seed_rank0"] == 45
    assert row["per_rank_offset"] == "seed + ddp_rank"
    assert row["config_seed"] == 45
    assert row["seed_agrees"] == "yes"
    assert row["production"] == "yes"


def test_a_command_line_seed_makes_the_row_disagree(tmp_path):
    run_dir = str(tmp_path / "r")
    write_manifest(run_dir, used_seed=1001)
    log = write_log(str(tmp_path / "logs" / "mc-x-152868.out"),
                    config_seed=45, cli_seed=1001)
    row = inventory_row(
        {"modality": "protein_sequence", "run": "retrain2-gpt2-medium-lr6e-4-s1001",
         "run_dir": run_dir, "job_id": "152868"},
        "learning_source_protein_runs", index_logs([os.path.dirname(log)]))
    assert row["cli_seed"] == 1001
    assert row["config_seed"] == 45
    assert row["seed_agrees"] == "yes"  # what ran matches what the CLI asked for
    assert "command line" in row["note"]


def test_a_run_with_no_manifest_records_no_seed(tmp_path):
    run_dir = str(tmp_path / "r")
    os.makedirs(run_dir, exist_ok=True)
    row = inventory_row(
        {"modality": "rna", "run": "rna-small", "run_dir": run_dir, "job_id": None},
        "learning_source_rna_runs", {})
    assert row["manifest_seed"] == ""
    assert row["manual_seed_rank0"] == ""
    assert row["seed_agrees"] == ""
    assert "not recorded" in row["note"]
    assert row["production"] == "no"  # no job id


def test_the_tsv_follows_the_convention(tmp_path):
    out = str(tmp_path / "rt1-gpt2-seed-inventory.tsv")
    row = {c: "" for c in COLUMNS}
    row.update(modality="rna", run="rna-small", production="no")
    write_tsv(out, [row], provenance([("rna", "/x/learning_source_rna_runs")],
                                     ["/x/workflows/slurm-logs"],
                                     "scripts/gpt2_seed_inventory.py"))
    raw = open(out, "rb").read()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert text.splitlines()[0] == "#! report-tsv 1"
    assert any(line.startswith("# model:") for line in text.splitlines())
    assert any(line.startswith("# source:") for line in text.splitlines())
    assert any(line.startswith("# generated:") for line in text.splitlines())
    header = [line for line in text.splitlines() if not line.startswith("#")][0]
    assert header.split("\t") == COLUMNS
    assert all(c.isascii() for c in header)


def test_an_inherited_seed_is_resolved_in_the_tree_the_run_used(monkeypatch):
    """The echo carries one file; the number can be in the file it imports."""
    import gpt2_seed_inventory as mod

    tree = {
        "molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_large_1500_lr1p2e3.py":
            "from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_large_1500"
            " import *  # noqa\nlearning_rate = 1.2e-3\n",
        "molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_large_1500.py":
            "seed = 42\nmax_iters = 1500\n",
    }
    monkeypatch.setattr(mod, "git_show", lambda commit, path: tree.get(path))
    seed, how = mod.resolve_seed_at_commit(
        "4a3d9f4", "molecule_nat_lang/gpt2_large_1500_lr1p2e3.py")
    assert seed == 42
    assert "inherited" in how


def test_a_commit_that_is_gone_is_said_so(monkeypatch):
    import gpt2_seed_inventory as mod
    monkeypatch.setattr(mod, "git_show", lambda commit, path: None)
    seed, how = mod.resolve_seed_at_commit("deadbee", "rna/gpt2_small.py")
    assert seed is None
    assert "not in the tree" in how


def test_no_commit_is_not_guessed():
    import gpt2_seed_inventory as mod
    assert mod.resolve_seed_at_commit(None, "rna/gpt2_small.py") == (
        None, "no commit recorded")


def test_lines_spliced_between_ranks_are_still_read(tmp_path):
    """Four ranks print to one file without a lock. Job 87220's log looks like this."""
    path = str(tmp_path / "mc-prot-rt2-87220.out")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f"Overriding config with {CONFIG_PATH}:\nseed = 42\n"
            "Overriding: min_lr = 6e-05Overriding: seed = 1001Overriding:"
            " min_lr = 6e-05Overriding: seed = 1001\n"
            "Overriding: init_from = scratchOverriding: seed = 1001\n")
    config, config_seed, cli_seed, source = read_declared_seed([path])
    assert cli_seed == 1001
    assert config_seed == 42
    assert config == "genome_sequence/gpt2_small_subset.py"
    assert source == "job log"
