"""The parts of a run manifest that are the same whichever trainer wrote it.

``models/gpt2/_run_manifest.py`` and ``models/bert/_run_manifest.py`` each write
what their own framework knows -- nanoGPT's GPU-independent global batch, HF's
selection metric and collapse detector. Three things are not framework-specific
at all, and every one of them exists because a value was resolved somewhere the
run never wrote down:

``git_state``
    A commit hash does not say the run came from that commit. An edited working
    tree runs happily and records the hash it was based on. Both trainers
    recorded only the hash.
``value_sources``
    Both trainers resolve config the same way: snapshot the module's globals,
    ``exec`` a config file over them, then let ``--key=value`` overwrite again --
    and the config files themselves read ``os.environ`` on the way. By the time
    anything is saved there is one number and no history. genome's production 21
    ran at learning_rate 1e-4 against a 6e-4 default, and it took opening 384
    checkpoints in 2026-08 to find that out.
``placement``
    GPUs are asked for with ``--gpus=N`` and the scheduler decides the node
    placement, so a request for 4 can arrive as 2 nodes with 2 each. Both
    launchers hard-code ``--nproc_per_node`` with ``--standalone`` and no
    ``MASTER_ADDR``, so they would drive one node only. What that costs differs
    by framework -- HF multiplies the effective batch by ``world_size`` and
    nanoGPT divides the accumulation by it -- but neither run said how many GPUs
    it was actually driving, so neither could be checked afterwards.
``environment``
    The overrides live in environment variables that neither ``ckpt.pt`` nor
    ``training_args.bin`` has ever carried. ``SUBSET_BERT_EPOCHS`` sets the epoch
    count genome's BERT schedule is derived from; the 9-epoch run used it and
    said so nowhere.
``introduced_values``
    ``value_sources`` can only report names the trainer itself declares, because
    ``config_keys`` is snapshotted before the config file runs. A name a config
    *introduces* is in neither the resolved dict nor the defaults, so it is
    skipped in silence -- ``expected_global_batch`` (#164) is one, and so are
    ``eos_token_id``, ``pad_token_id`` and ``bos_token_id``, which 47, 35 and 35
    configs set and no manifest has ever carried. RNA's cell boundary is token 0,
    the same id as its pad; that fact lived only in the prep script.

Each trainer passes its own names -- the values worth tracking and the variables
its configs read differ -- but the shape of the answer should not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

# How many changed paths to name before the list stops being useful in a manifest.
MAX_DIRTY_FILES = 50

# Read by config files across both trainers. Framework-specific names are added
# by the caller; these are the ones either side can hit.
COMMON_ENV = (
    "LEARNING_SOURCE_DIR",
    # Where generated model output goes when a run does not want it inside the
    # corpus (see core.paths.get_model_output_root). Unset is the old behaviour,
    # so the manifest has to say which of the two a run got.
    "MODEL_OUTPUT_ROOT",
    "GENOME_SUBSET",
    "HARD_MAX_STEPS_OVERRIDE",
)


def placement(world: int) -> Dict[str, Any]:
    """What the scheduler handed out, beside what the processes actually saw.

    ``world_size_matches_allocation`` is ``None``, not ``False``, when the
    allocation is unknown: outside SLURM there is nothing to disagree with, and
    ``False`` there would warn on every local run.
    """
    def _int(name: str) -> Optional[int]:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError, TypeError):
            return None

    nodes = _int("SLURM_JOB_NUM_NODES") or _int("SLURM_NNODES")
    # --gpus=N sets SLURM_GPUS; SLURM_GPUS_ON_NODE counts only this node's share.
    gpus_total = _int("SLURM_GPUS")
    gpus_here = _int("SLURM_GPUS_ON_NODE")
    if gpus_total is None and gpus_here is not None and nodes is not None:
        gpus_total = gpus_here * nodes

    return {
        "nodes": nodes,
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "gpus_allocated": gpus_total,
        "gpus_on_this_node": gpus_here,
        "world_size": world,
        "world_size_matches_allocation": (
            None if gpus_total is None else gpus_total == world
        ),
    }


def _git(*args: str) -> str:
    try:
        done = subprocess.run(("git",) + args, capture_output=True, text=True, timeout=10)
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> Dict[str, Any]:
    """Commit and branch, plus whether the tree that produced this run was clean.

    ``dirty`` is the field that matters: with it False the commit identifies the
    code, and with it True the commit is only where the code started from.
    """
    dirty = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "--short", "HEAD") or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(dirty),
        "dirty_files": [line[3:] for line in dirty.splitlines()][:MAX_DIRTY_FILES],
    }


def dirty_tree_warning(state: Mapping[str, Any]) -> Optional[str]:
    """The line to print when a run starts from an edited checkout, or None.

    Deliberately a warning and not a refusal: the three workstreams launch from
    checkouts they have just edited, and failing here would block them. The
    manifest carries the file list either way, so a run that turns out to be odd
    can be checked against it afterwards.
    """
    if not state.get("dirty"):
        return None
    return (
        "Working tree had uncommitted changes at launch: this run is NOT "
        f"reproducible from commit {state.get('commit')}. "
        f"{len(state.get('dirty_files') or [])} file(s) differ; "
        "they are listed in run_manifest.json."
    )


def value_sources(
    config: Mapping[str, Any],
    defaults: Mapping[str, Any],
    tracked: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Per value: what it is, whether it is the default, and what the default was.

    ``defaults`` is the trainer's globals snapshotted before its configurator
    runs. That separates "something set this" from "nothing touched it" -- the
    distinction both checkpoint formats lose. It does not separate a config file
    from an environment variable from ``--key=value``: all three overwrite the
    same globals in place, and by the time we can look there is nothing left to
    tell them apart. ``resolved_by`` says so rather than guessing.

    A value equal to its default is reported as "default" rather than omitted, so
    the manifest states the agreement instead of leaving a reader to assume it.
    """
    sources: Dict[str, Dict[str, Any]] = {}
    for key in tracked:
        if key not in config:
            continue
        value = config[key]
        default = defaults.get(key)
        same = value == default
        sources[key] = {
            "value": value,
            "from": "default" if same else "overridden",
            "default": default,
            "resolved_by": None if same else "config file, env var or --key=value (indistinguishable)",
        }
    return sources


def environment(extra: Iterable[str] = ()) -> Dict[str, str]:
    """The tracked environment variables that are actually set.

    Absent names are left out rather than recorded as null, so the manifest shows
    what shaped the run and not the whole list of things that could have.
    """
    names = list(COMMON_ENV) + list(extra)
    return {name: os.environ[name] for name in names if name in os.environ}


def configurator_locals(configurator_path: str) -> frozenset:
    """The names configurator.py binds in the namespace it is exec'd into.

    ``exec(open(configurator_path).read())`` runs at module level, so the
    configurator's own loop variables -- ``arg``, ``key``, ``val`` and the rest --
    become globals indistinguishable from anything a config file set. They are
    read out of its source rather than listed here, because a hard-coded list is
    exactly the kind that stops matching: TRACKED carried three names main.py had
    never declared until a test was written to hold the two in step.
    """
    import ast

    names = set()

    def walk(body):
        for node in body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.For):
                targets = [node.target]
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, (ast.If, ast.While, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    walk(getattr(node, attr, []) or [])
                continue
            elif isinstance(node, ast.ExceptHandler):
                walk(node.body)
                continue
            else:
                continue
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)

    try:
        walk(ast.parse(open(configurator_path).read()).body)
    except (OSError, SyntaxError):
        return frozenset()
    return frozenset(names)


def introduced_values(
    before: Iterable[str],
    after: Mapping[str, Any],
    configurator_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scalars a config file added, which value_sources cannot see.

    ``before`` is the trainer's ``config_keys``, taken before the configurator
    ran; ``after`` is the same filter applied to globals once it has. What is left
    is what a config introduced rather than overrode -- there is no default to
    compare against, which is why these are reported separately from ``sources``
    instead of being folded into it with a null default.

    Names the configurator itself binds are removed. Known names are *not*: an
    exclusion list of things already reported elsewhere would be one more list to
    forget to update, and a value appearing twice costs nothing.
    """
    excluded = configurator_locals(configurator_path) if configurator_path else frozenset()
    return {
        name: value
        for name, value in sorted(after.items())
        if name not in set(before) and name not in excluded
    }


# --------------------------------------------------------------------------- #
# Schema 2: what the run-manifest order (2026-09-17h) adds on top of the above.
#
# Everything below is additive. The helpers above keep their signatures, so the
# manifests both trainers already write keep every field they had (order §2.10).
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = 2

# Environment variables are captured by prefix, not by list. A list is what lost
# SUBSET_BERT_EPOCHS and SUBSET_BERT_MAX_LENGTH: the genome 9-epoch saturation run
# was launched through them and its manifest did not say so, because nobody had
# added the names yet. A prefix catches the next variable a config starts reading.
#
# ``config``: read by config files and the trainers' own settings.
# ``runtime``: shape how the processes ran (placement, threads, NCCL, caches).
ENV_PREFIXES = {
    "config": (
        "SUBSET_",
        "SMOKE_",
        "BERT_",
        "GPT2_",
        "GENOME_",
        "HARD_MAX_",
        "LEARNING_SOURCE_",
        "MODEL_OUTPUT_",
        "MOLCRAWL_",
        "USE_WANDB",
        "WANDB_",
    ),
    "runtime": (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_",
        "TORCH_",
        "TORCHDYNAMO_",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "PYTHONHASHSEED",
        "HF_",
        "DS_",
    ),
}

# A captured name matching this is recorded as present with its value withheld.
_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|APIKEY|AUTH|PRIVATE_KEY|ACCESS_KEY)",
    re.IGNORECASE,
)
REDACTED = "<redacted>"


def now_iso() -> str:
    """Timezone-aware ISO 8601, to the second."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def is_rank_zero() -> bool:
    """True on the one process that may write the manifest.

    Reads ``RANK`` rather than asking torch.distributed, so it answers the same
    before and after the process group exists -- the manifest is first written
    before initialisation. Absent ``RANK`` means a single process.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:  # torch absent or not initialisable here
        pass
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _json_default(value: Any) -> Any:
    """Normalise what json cannot take, rather than failing the write."""
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def write_json_atomic(path: str, obj: Any) -> None:
    """Write through a temporary file in the same directory, then rename.

    A reader never sees half a manifest, and a crash mid-write leaves the
    previous version in place.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_json_atomic(path: str, mutate) -> Optional[Dict[str, Any]]:
    """Read, apply ``mutate(dict)`` in place, write back atomically.

    Returns the updated dict, or None when there is no file to update. Only the
    caller decides whether this process may write; see ``is_rank_zero``.
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    mutate(data)
    write_json_atomic(path, data)
    return data


def sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def environment_by_prefix(
    prefixes: Mapping[str, Iterable[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Every set variable whose name starts with a listed prefix, secrets withheld.

    ``redacted`` lists the names whose values were withheld, so a reader can see
    that e.g. ``WANDB_API_KEY`` was set without the manifest carrying it.
    """
    prefixes = ENV_PREFIXES if prefixes is None else prefixes
    environ = os.environ if environ is None else environ
    out: Dict[str, Any] = {"prefixes": {k: list(v) for k, v in prefixes.items()}, "redacted": []}
    for group, group_prefixes in prefixes.items():
        found = {}
        for name in sorted(environ):
            if any(name.startswith(p) for p in group_prefixes):
                if _SECRET_NAME.search(name):
                    found[name] = REDACTED
                    out["redacted"].append(name)
                else:
                    found[name] = environ[name]
        out[group] = found
    out["redacted"] = sorted(set(out["redacted"]))
    return out


def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py<3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def runtime_versions() -> Dict[str, Any]:
    """Versions the running process has, not what environment.yaml declares.

    ``torch`` keeps its build suffix (``2.8.0+cu129``): on aarch64 the plain
    wheel is CPU-only, so ``2.8.0`` alone does not say whether CUDA was there.
    Library versions come from package metadata so recording them imports
    nothing new; torch's CUDA and NCCL are read only if torch is already loaded.
    """
    out: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": _dist_version("torch"),
        "torch_cuda": None,
        "nccl": None,
        "transformers": _dist_version("transformers"),
        "accelerate": _dist_version("accelerate"),
        "datasets": _dist_version("datasets"),
        "pyarrow": _dist_version("pyarrow"),
        "numpy": _dist_version("numpy"),
        "deepspeed": _dist_version("deepspeed"),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch"] = getattr(torch, "__version__", out["torch"])
        out["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            nccl = torch.cuda.nccl.version()
            out["nccl"] = ".".join(str(p) for p in nccl) if isinstance(nccl, tuple) else str(nccl)
        except Exception:
            out["nccl"] = None
    return out


def _nvidia_smi_driver() -> Dict[str, Any]:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        return {"driver_version": lines[0] if lines else None, "error": None if lines else done.stderr.strip() or "no output"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"driver_version": None, "error": f"{type(exc).__name__}: {exc}"}


def gpu_info(query_driver: bool = True) -> Dict[str, Any]:
    """This node's visible GPUs. Written per rank-0 node; other nodes via ``nodes``.

    Optional throughout: a failure is recorded with its reason, never raised.
    """
    out: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_count": None,
        "devices": [],
        "driver_version": None,
        "error": None,
    }
    torch = sys.modules.get("torch")
    if torch is None:
        out["error"] = "torch not loaded in this process"
    else:
        try:
            if torch.cuda.is_available():
                out["visible_count"] = torch.cuda.device_count()
                for i in range(out["visible_count"]):
                    props = torch.cuda.get_device_properties(i)
                    out["devices"].append(
                        {"index": i, "name": props.name, "total_memory_bytes": int(props.total_memory)}
                    )
            else:
                out["visible_count"] = 0
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
    if query_driver:
        smi = _nvidia_smi_driver()
        out["driver_version"] = smi["driver_version"]
        if smi["error"] and out["error"] is None:
            out["driver_error"] = smi["error"]
    return out


def cpu_info() -> Dict[str, Any]:
    """CPUs asked for and CPUs this process can actually be scheduled on.

    The two differ when --cpus-per-task is missing from a launcher, and then
    DataLoader workers queue behind each other: worker 4 at 4 ranks per node
    wants 16 CPUs on the node.
    """
    try:
        usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable = None
    requested = os.environ.get("SLURM_CPUS_PER_TASK")
    return {
        "cpus_per_task_requested": int(requested) if requested and requested.isdigit() else None,
        "cpus_usable_by_process": usable,
        "cpus_on_node": os.cpu_count(),
    }


def slurm_info() -> Dict[str, Any]:
    """Scheduler identity and shape, as the process sees it."""
    env = os.environ

    def _int(name):
        try:
            return int(env[name])
        except (KeyError, ValueError):
            return None

    return {
        "job_id": env.get("SLURM_JOB_ID"),
        "step_id": env.get("SLURM_STEP_ID"),
        "account": env.get("SLURM_JOB_ACCOUNT"),
        "partition": env.get("SLURM_JOB_PARTITION"),
        "qos": env.get("SLURM_JOB_QOS"),
        "nodelist": env.get("SLURM_JOB_NODELIST"),
        "nodes": _int("SLURM_JOB_NUM_NODES") or _int("SLURM_NNODES"),
        "ntasks": _int("SLURM_NTASKS"),
        "ntasks_per_node": env.get("SLURM_NTASKS_PER_NODE"),
        "gpus": env.get("SLURM_GPUS"),
        "procid": _int("SLURM_PROCID"),
        "localid": _int("SLURM_LOCALID"),
        "hostname": socket.gethostname(),
    }


class ArgvOrderError(ValueError):
    """Command-line arguments the configurator would resolve surprisingly."""


def resolve_config_path(arg: str, trainer_file: Optional[str]) -> str:
    """The same fallback configurator.py applies to a relative config path."""
    if os.path.isabs(arg) or os.path.exists(arg) or not trainer_file:
        return os.path.abspath(arg)
    parent = os.path.dirname(os.path.dirname(os.path.abspath(trainer_file)))
    candidate = os.path.join(parent, arg)
    return os.path.abspath(candidate if os.path.exists(candidate) else arg)


def parse_argv(argv: Iterable[str], trainer_file: Optional[str] = None) -> Dict[str, Any]:
    """Split argv the way configurator.py does, and say whether its order is safe.

    configurator.py applies arguments left to right, so a config file placed
    after ``--key=value`` silently overwrites the CLI value, and a second config
    file overwrites the first. Neither is used by any launcher in the tree; both
    are refused by ``validate_argv``.
    """
    args = list(argv)
    config_files: List[Dict[str, Any]] = []
    overrides: List[Dict[str, Any]] = []
    problems: List[str] = []
    for position, arg in enumerate(args):
        if "=" not in arg:
            if overrides:
                problems.append(f"config file {arg!r} at position {position} comes after --key=value overrides")
            path = resolve_config_path(arg, trainer_file)
            config_files.append({"argument": arg, "path": path, "sha256": sha256_file(path), "position": position})
        else:
            key, _, value = arg.partition("=")
            overrides.append({"key": key[2:] if key.startswith("--") else key, "value": value, "position": position})
    if len(config_files) > 1:
        problems.append(f"{len(config_files)} config files given; at most one is allowed")
    return {"argv": args, "config_files": config_files, "cli_overrides": overrides, "problems": problems}


def validate_argv(argv: Iterable[str], trainer_file: Optional[str] = None) -> Dict[str, Any]:
    """``parse_argv``, raising when the order would let a later argument win silently."""
    parsed = parse_argv(argv, trainer_file)
    if parsed["problems"]:
        raise ArgvOrderError("; ".join(parsed["problems"]))
    return parsed


def config_file_record(parsed: Mapping[str, Any]) -> Dict[str, Any]:
    """The training config a run executed: absolute path and content hash.

    The hash is not optional decoration: configs outside git have been run, and
    for those a commit identifies nothing.
    """
    files = parsed.get("config_files") or []
    if not files:
        return {"path": None, "sha256": None, "reason": "no config file argument"}
    first = files[0]
    return {"path": first["path"], "sha256": first["sha256"], "argument": first["argument"]}


# Scalars a snapshot keeps: the same filter both trainers use for config_keys.
SNAPSHOT_TYPES = (int, float, bool, str)


def scalar_snapshot(namespace: Mapping[str, Any], exclude: Iterable[str] = ()) -> Dict[str, Any]:
    excluded = set(exclude)
    return {
        k: v
        for k, v in namespace.items()
        if not k.startswith("_") and k not in excluded and isinstance(v, SNAPSHOT_TYPES)
    }


STAGES = ("defaults", "after_config_file", "after_cli", "after_deepspeed", "resolved")


def stage_sources(snapshots: Mapping[str, Optional[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Per key: its value at each stage, the final value, and the stage that set it.

    A stage whose snapshot is None did not happen (no DeepSpeed) and is skipped
    rather than treated as having removed every key. ``set_by`` is the last stage
    at which the value changed; "defaults" means nothing after the code's own
    default touched it. Environment variables read inside a config file are part
    of ``after_config_file`` and cannot be separated from it here -- the
    environment is recorded beside this for that reason.
    """
    present = [(name, snapshots.get(name)) for name in STAGES if snapshots.get(name) is not None]
    keys = sorted(set().union(*[set(snap) for _, snap in present])) if present else []
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        values: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []
        previous_value, previous_present, set_by = None, False, None
        for name, snap in present:
            if key in snap:
                value = snap[key]
                values[name] = value
                if not previous_present or value != previous_value:
                    if previous_present:
                        history.append({"stage": name, "from": previous_value, "to": value})
                    set_by = name
                previous_value, previous_present = value, True
        out[key] = {
            "stages": values,
            "final": previous_value,
            "set_by": set_by,
            "overridden": history,
        }
    return out


def names_read_by(source_paths: Iterable[str]) -> frozenset:
    """Global names a trainer's source can read, found statically.

    Counts plain name loads, ``globals()[...]`` / ``globals().get(...)`` with a
    literal key, and attribute-free string keys passed to those. Static, so it
    says "no code path reads this name" -- not that a path which does was taken.
    """
    import ast

    names = set()
    for path in source_paths:
        try:
            tree = ast.parse(open(path).read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                is_globals_get = (
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id == "globals"
                )
                if is_globals_get and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names.add(node.args[0].value)
            elif isinstance(node, ast.Subscript):
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "globals"
                ):
                    key = node.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names.add(key.value)
    return frozenset(names)


def mark_used(values: Mapping[str, Any], read_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Each value with ``used``: whether the trainer's code reads that name at all.

    configurator.py refuses unknown ``--key`` arguments but ``exec``s a config
    file whole, so a name only a config file sets and no code reads is silently
    inert. Here it keeps its value and is marked ``used: false``.
    """
    read = set(read_names)
    return {name: {"value": value, "used": name in read} for name, value in sorted(values.items())}
