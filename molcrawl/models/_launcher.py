"""The srun launcher for the DeepSpeed path: Slurm placement to torch.distributed env.

DeepSpeed order §8. The existing trainers are launched by
``torchrun --standalone --nproc_per_node=4`` inside one Slurm task, which drives
one node only; no workflow in the tree asks for ``--nodes``. The DeepSpeed path
instead runs **one Slurm task per GPU**, and each task becomes one training
process:

- ``sbatch_args`` builds the allocation request the order specifies --
  ``--nodes=N --ntasks=4N --ntasks-per-node=4 --gpus=4N --gpu-bind=none
  --cpus-per-task=C`` -- and never ``--gpus-per-node`` or ``--gres=gpu:N``.
  ``--cpus-per-task`` is always written: without it DataLoader workers queue
  behind each other and the input side becomes the bottleneck again.
- ``srun_args`` repeats the task shape for the step. Since Slurm 22.05 srun does
  not inherit ``--cpus-per-task`` from the allocation, so it is passed again.
- ``distributed_env`` maps ``SLURM_PROCID`` / ``SLURM_LOCALID`` / ``SLURM_NTASKS``
  to ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``, takes the first allocated node's
  IPv4 address as ``MASTER_ADDR`` and derives ``MASTER_PORT`` from the job id so
  two jobs sharing a node do not collide. It refuses a task layout that is not
  exactly one task per GPU.

``python -m molcrawl.models._launcher exec -- <command...>`` is what srun runs in
every task: it sets that environment, marks the process as launched by
``srun_deepspeed`` for the run manifest, and ``exec``s the command. Nothing here
chooses a network interface: NCCL's own detection is used (order §8.3).
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
from typing import Dict, List, Mapping, Optional

LAUNCHER_TYPE = "srun_deepspeed"
GPUS_PER_NODE = 4
PORT_BASE = 20000
PORT_SPAN = 20000
# Never used on this cluster (order §8): they select GPUs a different way.
FORBIDDEN_SBATCH = ("--gpus-per-node", "--gres")


class LaunchError(ValueError):
    """A placement that would not give exactly one process per GPU."""


def _positive(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LaunchError(f"{name} must be a positive int, got {value!r}")
    return value


def cpus_per_task_for(dataloader_num_workers: int, spare: int = 1, minimum: int = 2) -> int:
    """One CPU for the training process, one per DataLoader worker, plus a spare.

    Deliberately explicit rather than "whatever is left": worker 4 at 4 ranks per
    node already needs 16 worker processes on the node, and the manifest records
    both this request and what the process could actually use.
    """
    workers = int(dataloader_num_workers)
    if workers < 0:
        raise LaunchError(f"dataloader_num_workers must be >= 0, got {workers}")
    return max(int(minimum), 1 + workers + int(spare))


def sbatch_args(
    *,
    nodes: int,
    cpus_per_task: int,
    time: str,
    job_name: str,
    gpus_per_node: int = GPUS_PER_NODE,
    account: Optional[str] = None,
    partition: Optional[str] = "gpu",
    output: str = "workflows/slurm-logs/%x-%j.out",
) -> List[str]:
    """sbatch options for N nodes, one task per GPU."""
    n = _positive("nodes", nodes)
    gpn = _positive("gpus_per_node", gpus_per_node)
    cpt = _positive("cpus_per_task", cpus_per_task)
    if not re.fullmatch(r"(\d+-)?\d{1,2}(:\d{2}){0,2}", str(time)):
        raise LaunchError(f"time {time!r} is not a Slurm time limit (D-HH:MM:SS, HH:MM:SS, MM)")
    args = [
        f"--job-name={job_name}",
        f"--nodes={n}",
        f"--ntasks={n * gpn}",
        f"--ntasks-per-node={gpn}",
        f"--gpus={n * gpn}",
        "--gpu-bind=none",
        f"--cpus-per-task={cpt}",
        f"--time={time}",
        f"--output={output}",
        f"--error={output}",
    ]
    if partition:
        args.append(f"--partition={partition}")
    if account:
        args.append(f"--account={account}")
    for arg in args:
        if arg.startswith(FORBIDDEN_SBATCH):
            raise LaunchError(f"{arg} is not used on this cluster")
    return args


def srun_args(*, nodes: int, cpus_per_task: int, gpus_per_node: int = GPUS_PER_NODE) -> List[str]:
    """The step's task shape, restated because srun does not inherit --cpus-per-task."""
    n = _positive("nodes", nodes)
    gpn = _positive("gpus_per_node", gpus_per_node)
    return [
        "srun",
        f"--nodes={n}",
        f"--ntasks={n * gpn}",
        f"--ntasks-per-node={gpn}",
        f"--cpus-per-task={_positive('cpus_per_task', cpus_per_task)}",
        "--gpu-bind=none",
        "--kill-on-bad-exit=1",
    ]


def expand_nodelist(nodelist: str) -> List[str]:
    """``c[001-003,010],d7`` -> ``[c001, c002, c003, c010, d7]``, keeping zero padding.

    Covers the one-bracket-group form Slurm uses here. ``scontrol show hostnames``
    is preferred when available (``first_host``); this is its offline fallback.
    """
    hosts: List[str] = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", nodelist):
        match = re.fullmatch(r"([^\[]*)\[([^\]]*)\](.*)", part)
        if not match:
            if part:
                hosts.append(part)
            continue
        prefix, ranges, suffix = match.groups()
        for item in ranges.split(","):
            if "-" in item:
                lo, hi = item.split("-", 1)
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    hosts.append(f"{prefix}{str(i).zfill(width)}{suffix}")
            else:
                hosts.append(f"{prefix}{item}{suffix}")
    return hosts


def first_host(nodelist: str) -> str:
    try:
        done = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                              capture_output=True, text=True, timeout=10)
        names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        if done.returncode == 0 and names:
            return names[0]
    except (OSError, subprocess.SubprocessError):
        pass
    hosts = expand_nodelist(nodelist)
    if not hosts:
        raise LaunchError(f"could not read a host from SLURM_JOB_NODELIST={nodelist!r}")
    return hosts[0]


def ipv4_of(host: str) -> str:
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET)
    except socket.gaierror as exc:
        raise LaunchError(f"no IPv4 address for rendezvous host {host!r}: {exc}") from exc
    return infos[0][4][0]


def master_port_for(job_id: str) -> int:
    try:
        return PORT_BASE + int(job_id) % PORT_SPAN
    except (TypeError, ValueError) as exc:
        raise LaunchError(f"SLURM_JOB_ID {job_id!r} is not an integer") from exc


def distributed_env(environ: Mapping[str, str], resolve_ip=ipv4_of, host_of=first_host) -> Dict[str, str]:
    """RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* for this task, from Slurm's variables."""
    def need(name: str) -> str:
        if name not in environ:
            raise LaunchError(f"{name} is not set; exec must run inside srun")
        return environ[name]

    rank = int(need("SLURM_PROCID"))
    local = int(need("SLURM_LOCALID"))
    world = int(need("SLURM_NTASKS"))
    nodes = int(environ.get("SLURM_JOB_NUM_NODES") or environ.get("SLURM_NNODES") or 0)
    per_node = environ.get("SLURM_NTASKS_PER_NODE", "")
    if nodes and world != nodes * GPUS_PER_NODE:
        raise LaunchError(f"SLURM_NTASKS {world} is not {GPUS_PER_NODE} x {nodes} nodes: not one task per GPU")
    if per_node and per_node.split("(")[0] != str(GPUS_PER_NODE):
        raise LaunchError(f"SLURM_NTASKS_PER_NODE={per_node!r}; expected {GPUS_PER_NODE}")
    if not 0 <= local < GPUS_PER_NODE:
        raise LaunchError(f"SLURM_LOCALID {local} outside 0..{GPUS_PER_NODE - 1}")
    if not 0 <= rank < world:
        raise LaunchError(f"SLURM_PROCID {rank} outside 0..{world - 1}")
    host = host_of(need("SLURM_JOB_NODELIST"))
    return {
        "RANK": str(rank),
        "LOCAL_RANK": str(local),
        "WORLD_SIZE": str(world),
        "LOCAL_WORLD_SIZE": str(GPUS_PER_NODE),
        "MASTER_ADDR": resolve_ip(host),
        "MASTER_PORT": str(master_port_for(need("SLURM_JOB_ID"))),
        "MOLCRAWL_LAUNCHER": LAUNCHER_TYPE,
        "MOLCRAWL_RENDEZVOUS_HOST": host,
    }


def _exec(argv: List[str]) -> int:
    if not argv:
        raise LaunchError("exec needs a command after --")
    env = dict(os.environ)
    env.update(distributed_env(os.environ))
    if env["RANK"] == "0":
        print(f"[launcher] world_size={env['WORLD_SIZE']} master={env['MOLCRAWL_RENDEZVOUS_HOST']}"
              f" ({env['MASTER_ADDR']}:{env['MASTER_PORT']})", flush=True)
    print(f"[launcher] rank={env['RANK']} local_rank={env['LOCAL_RANK']} host={socket.gethostname()}"
          f" CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    os.execvpe(argv[0], argv, env)
    return 0  # not reached


def record_launch_failure(out_dir: str, exit_code: int, environ: Optional[Mapping[str, str]] = None) -> Dict:
    """Order §4.4: a failure the trainer never got to record.

    When srun fails before Python starts, or the trainer dies without reaching its
    own failure hook (a SIGKILL, a node loss), there is either no manifest or one
    still saying ``starting`` / ``running``. This writes a preliminary manifest in
    the first case and marks the existing one failed in the second. A manifest
    already ``completed`` or ``failed`` is left alone.
    """
    from molcrawl.models._provenance import (
        SCHEMA_VERSION,
        TERMINAL_STATUSES,
        launcher_record,
        now_iso,
        slurm_info,
        update_json_atomic,
        write_json_atomic,
    )

    env = os.environ if environ is None else environ
    path = os.path.join(out_dir, "run_manifest.json")
    entry = {"status": "failed", "at": now_iso(), "phase": "launch"}
    failure = {
        "phase": "launch",
        "error_summary": f"srun step exited with {exit_code}; the trainer did not record an outcome",
        "slurm_step_exit_code": int(exit_code),
    }
    if not os.path.exists(path):
        document = {
            "schema_version": SCHEMA_VERSION,
            "written": now_iso(),
            "preliminary": True,
            "run": {
                "status": "failed",
                "status_history": [entry],
                "exit_code": int(exit_code),
                "failure": failure,
                "out_dir": os.path.abspath(out_dir),
                "job_id": env.get("SLURM_JOB_ID"),
            },
            "launcher": launcher_record(environ=env),
            "slurm": slurm_info(),
            "deepspeed": {"enabled": None, "reason": "trainer did not start"},
        }
        write_json_atomic(path, document)
        return document

    def _mutate(doc):
        run = doc.setdefault("run", {})
        if run.get("status") in TERMINAL_STATUSES:
            return
        run["status"] = "failed"
        run["exit_code"] = int(exit_code)
        run["end_time"] = entry["at"]
        run["failure"] = {**failure, "last_status": run.get("status_history", [{}])[-1].get("status")}
        run.setdefault("status_history", []).append(entry)

    return update_json_atomic(path, _mutate)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m molcrawl.models._launcher")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exec", help="set the distributed env from Slurm and exec the command after --")
    p_srun = sub.add_parser("srun-args", help="print the srun options for the step, one per line")
    p_srun.add_argument("--nodes", type=int, required=True)
    p_srun.add_argument("--cpus-per-task", type=int, required=True)
    p_fail = sub.add_parser("record-failure", help="mark the run failed if the trainer could not")
    p_fail.add_argument("--out-dir", required=True)
    p_fail.add_argument("--exit-code", type=int, required=True)
    args, rest = parser.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.cmd == "exec":
        return _exec(rest)
    if rest:
        parser.error(f"unexpected arguments: {rest}")
    if args.cmd == "srun-args":
        print("\n".join(srun_args(nodes=args.nodes, cpus_per_task=args.cpus_per_task)))
        return 0
    if args.cmd == "record-failure":
        record_launch_failure(args.out_dir, args.exit_code)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
