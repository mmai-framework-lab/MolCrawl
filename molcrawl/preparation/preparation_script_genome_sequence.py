"""Run genome preprocessing from a YAML configuration."""

import argparse
import shlex
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to genome preparation YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the command without starting preprocessing",
    )
    args = parser.parse_args()

    try:
        with open(args.config) as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        parser.error(str(error))

    if not isinstance(document, dict):
        parser.error("YAML must contain a data_preparation mapping")

    config = document.get("data_preparation")
    if not isinstance(config, dict):
        parser.error("YAML must contain a data_preparation mapping")

    if config.get("pipeline") != "single_nucleotide":
        parser.error("Set data_preparation.pipeline to single_nucleotide")

    allowed = {
        "pipeline", "input_dir", "output_dir",
        "num_worker", "num_shards", "pilot_files",
    }
    unknown = set(config) - allowed
    if unknown:
        parser.error("Unknown settings: " + ", ".join(sorted(unknown)))

    for name in ("input_dir", "output_dir"):
        value = config.get(name)
        if not isinstance(value, str) or not value.strip():
            parser.error(f"{name} must be a nonempty path string")

    for name, default, minimum in (
        ("num_worker", 8, 1),
        ("num_shards", 64, 1),
        ("pilot_files", 0, 0),
    ):
        value = config.get(name, default)
        if type(value) is not int or value < minimum:
            parser.error(f"{name} must be an integer >= {minimum}")
        config[name] = value

    command = [
        sys.executable, "-u", "-m",
        "molcrawl.preparation.genome.parallel",
        "--input", config["input_dir"],
        "--output", config["output_dir"],
        "--workers", str(config["num_worker"]),
        "--shards", str(config["num_shards"]),
        "--pilot", str(config["pilot_files"]),
    ]
    print(shlex.join(command), flush=True)

    if args.dry_run:
        return 0

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
