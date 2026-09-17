"""Build (and only with --submit, run) the sbatch command for a DeepSpeed training job.

Dry run by default: prints the command, the placement and the checks it made.

    RUNS_ROOT=... LEARNING_SOURCE_DIR=... python scripts/submit_deepspeed.py \\
        --arch gpt2 --model-config molcrawl/tasks/pretrain/configs/rna/gpt2_small.py \\
        --run-name rna-gpt2-small-ds-1n --nodes 1 --time 01:00:00 --account <account>

Resources come from here, not from the sbatch file (DeepSpeed order §8.3: account,
nodes, GPUs, CPUs and time are overridable at launch, and none is fixed in code).
``--cpus-per-task`` defaults to what the model config's ``dataloader_num_workers``
needs (one per worker, one for the process, one spare); GPT-2 reads batches in the
training process, so it gets the minimum.

Before printing anything it checks what would otherwise fail on the node: the model
config and its ``.deepspeed.json`` exist, the DeepSpeed file passes the contract in
``_deepspeed_config``, and -- for ``global_fixed`` -- the batch divides at the
requested world size. The batch check reads the config with ``ast`` and never
executes it.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from molcrawl.models import _deepspeed_config as dsc  # noqa: E402
from molcrawl.models._batch_policy import (  # noqa: E402
    BatchPolicyError,
    from_legacy_bert,
    from_legacy_gpt2,
    resolve_global_fixed,
    scaling_feasibility,
)
from molcrawl.models._launcher import GPUS_PER_NODE, cpus_per_task_for, sbatch_args  # noqa: E402

SBATCH_FILE = "workflows/deepspeed-train.sbatch"


def config_ints(path: str) -> Dict[str, int]:
    """Module-level integer assignments of a config, by ``ast``; the config is not run."""
    env: Dict[str, int] = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.FloorDiv)):
            left, right = value(node.left), value(node.right)
            if left is None or right is None:
                return None
            return {ast.Mult: left * right, ast.Add: left + right, ast.Sub: left - right,
                    ast.FloorDiv: left // right if right else None}[type(node.op)]
        return None

    for stmt in ast.parse(open(path).read()).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, node = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, node = stmt.target.id, stmt.value
        else:
            continue
        v = value(node)
        if v is None:
            env.pop(name, None)
        else:
            env[name] = v
    return env


def plan(args: argparse.Namespace, environ: Optional[Dict[str, str]] = None) -> Dict:
    environ = dict(os.environ if environ is None else environ)
    problems: List[str] = []
    for name in ("RUNS_ROOT", "LEARNING_SOURCE_DIR"):
        if not environ.get(name):
            problems.append(f"{name} is not set")

    model_config = args.model_config
    if not os.path.isfile(os.path.join(REPO, model_config)):
        raise SystemExit(f"model config {model_config} not found under {REPO}")
    ds_path = args.ds_config or dsc.config_path_for(model_config)
    ds_facts = None
    if not os.path.isfile(os.path.join(REPO, ds_path)):
        problems.append(f"DeepSpeed config {ds_path} does not exist")
    else:
        data, _ = dsc.load(os.path.join(REPO, ds_path))
        try:
            ds_facts = dsc.validate(data)
        except dsc.DeepSpeedConfigError as exc:
            problems.append(f"{ds_path}: {exc}")

    values = config_ints(os.path.join(REPO, model_config))
    world = args.nodes * GPUS_PER_NODE
    batch = {"world_size": world}
    try:
        if args.arch == "gpt2":
            canon = from_legacy_gpt2(values.get("batch_size"), values.get("gradient_accumulation_steps"))
        else:
            canon = from_legacy_bert(values.get("batch_size"), values.get("gradient_accumulation_steps"),
                                     values.get("expected_global_batch"))
        resolved = resolve_global_fixed(canon, world)
        batch.update(resolved.as_manifest())
        batch["scaling_feasibility"] = scaling_feasibility(canon)
    except BatchPolicyError as exc:
        problems.append(f"batch: {exc}")

    workers = values.get("dataloader_num_workers", 0) if args.arch == "bert" else 0
    cpus = args.cpus_per_task or cpus_per_task_for(workers)
    sbatch = ["sbatch"] + sbatch_args(
        nodes=args.nodes, cpus_per_task=cpus, time=args.time, job_name=args.job_name or args.run_name,
        account=args.account, partition=args.partition,
    )
    exports = {
        "ARCH": args.arch,
        "MODEL_CONFIG": model_config,
        "DS_CONFIG": ds_path,
        "RUN_NAME": args.run_name,
    }
    if args.extra_args:
        if "," in args.extra_args:
            # sbatch --export splits on commas, so the value would arrive cut.
            problems.append("--extra-args must not contain a comma (sbatch --export splits on it)")
        exports["EXTRA_ARGS"] = args.extra_args
    if args.expect_commit:
        exports["EXPECT_COMMIT"] = args.expect_commit
    sbatch.append("--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()))
    sbatch.append(SBATCH_FILE)
    return {
        "command": sbatch,
        "cwd": REPO,
        "nodes": args.nodes,
        "world_size": world,
        "cpus_per_task": cpus,
        "cpus_per_task_from": "--cpus-per-task" if args.cpus_per_task else f"dataloader_num_workers={workers}",
        "batch": batch,
        "deepspeed": {"path": ds_path, **(ds_facts or {})},
        "outputs": os.path.join(environ.get("RUNS_ROOT", "<RUNS_ROOT unset>"), args.run_name),
        "problems": problems,
    }


def parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=("gpt2", "bert"), required=True)
    p.add_argument("--model-config", required=True, help="path relative to the repository root")
    p.add_argument("--run-name", required=True)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--time", required=True)
    p.add_argument("--account", default=None)
    p.add_argument("--partition", default="gpu")
    p.add_argument("--cpus-per-task", type=int, default=None)
    p.add_argument("--ds-config", default=None)
    p.add_argument("--job-name", default=None)
    p.add_argument("--extra-args", default=None, help="further --key=value overrides, space separated")
    p.add_argument("--expect-commit", default=None)
    p.add_argument("--submit", action="store_true", help="run sbatch; without it this is a dry run")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse(argv)
    result = plan(args)
    print("cd " + shlex.quote(result["cwd"]))
    print(" ".join(shlex.quote(part) for part in result["command"]))
    print(f"placement: {result['nodes']} node(s), world_size {result['world_size']},"
          f" cpus-per-task {result['cpus_per_task']} ({result['cpus_per_task_from']})")
    batch = result["batch"]
    if "effective_global_batch" in batch:
        print(f"batch: {batch['formula']} -> effective {batch['effective_global_batch']};"
              f" max feasible nodes {batch['scaling_feasibility']['max_feasible_candidate_node_count']}")
    print(f"outputs: {result['outputs']}")
    for problem in result["problems"]:
        print(f"PROBLEM: {problem}")
    if result["problems"]:
        return 1
    if not args.submit:
        print("dry run: nothing submitted (pass --submit to run sbatch)")
        return 0
    return subprocess.call(result["command"], cwd=result["cwd"])


if __name__ == "__main__":
    sys.exit(main())
