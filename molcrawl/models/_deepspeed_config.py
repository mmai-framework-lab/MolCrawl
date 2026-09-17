"""The DeepSpeed config file: what it may say, and what MolCrawl fills in.

DeepSpeed order §5.3 fixes the shape of the file handed to DeepSpeed: one per
model config, beside it in ``molcrawl/tasks/pretrain/configs/<modality>/``, the
same file for one node and for many, with the GPU-count-dependent batch values
left as ``"auto"`` and no ``optimizer``. This module holds that contract, so
the values it names have exactly one authoritative source:

batch (``train_batch_size``, ``train_micro_batch_size_per_gpu``,
``gradient_accumulation_steps``)
    MolCrawl's batch policy (``_batch_policy``). The file must say ``"auto"``.
    For the HF Trainer the ``"auto"`` stays and transformers fills it from
    ``TrainingArguments`` (integrations/deepspeed.py:135-158); for the custom
    GPT-2 loop MolCrawl fills it here.
optimizer, scheduler
    The trainer's own code (order §2, §5): GPT-2's ``configure_optimizers`` with
    its fused AdamW and weight-decay groups, BERT's Trainer ``adamw_torch`` with
    embeddings excluded from decay, and each trainer's learning-rate schedule.
    A file that names either is refused. A DeepSpeed ``optimizer`` would receive
    one flat parameter group and decay embeddings, LayerNorm and biases.
precision
    The trainer's config (``bf16`` for BERT, ``dtype`` for GPT-2), and mapped to
    DeepSpeed's ``torch_autocast`` -- *not* to DeepSpeed's ``bf16`` block. In
    DeepSpeed 0.19.5 ``bf16.enabled`` casts the module's parameters to bfloat16
    (engine.py:1714-1721) and wraps the optimizer in BF16_Optimizer with fp32
    master weights; the existing runs keep fp32 parameters and autocast the
    forward. ``torch_autocast`` keeps fp32 parameters and wraps the forward in
    ``torch.autocast`` (engine.py:2779), and at ZeRO stage 0 gradients are
    all-reduced in their own dtype (engine.py:174-191). The file must not set
    ``bf16``, ``fp16``, ``amp`` or ``torch_autocast``; they are written here.
gradient clipping
    The trainer's config (``grad_clip`` / ``max_grad_norm``). ``"auto"`` or the
    same value.
ZeRO and offload
    The file. Stage 0 and no offload are the initial specification (order §5);
    anything else is refused unless the caller passes ``allow_zero_stage`` /
    ``allow_offload``, because the order requires a report before using them.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

from molcrawl.models._batch_policy import BatchResolution
from molcrawl.models._provenance import sha256_file

SUFFIX = ".deepspeed.json"
AUTO = "auto"
BATCH_KEYS = ("train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps")
FORBIDDEN_KEYS = ("optimizer", "scheduler")
PRECISION_KEYS = ("bf16", "bfloat16", "fp16", "amp", "torch_autocast")
PRECISIONS = ("fp32", "bf16", "fp16")
FRAMEWORKS = ("hf", "custom")


class DeepSpeedConfigError(ValueError):
    """A DeepSpeed config that would give a value a second source, or change a condition."""


def config_path_for(model_config_path: str) -> str:
    """``configs/rna/bert_small.py`` -> ``configs/rna/bert_small.deepspeed.json``."""
    base, ext = os.path.splitext(model_config_path)
    if ext != ".py":
        raise DeepSpeedConfigError(f"expected a .py model config, got {model_config_path!r}")
    return base + SUFFIX


def default_config() -> Dict[str, Any]:
    """The initial specification (order §5) as a file body."""
    return {
        "train_batch_size": AUTO,
        "train_micro_batch_size_per_gpu": AUTO,
        "gradient_accumulation_steps": AUTO,
        "gradient_clipping": AUTO,
        "zero_optimization": {"stage": 0},
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }


def load(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The parsed file and its identity record (absolute path, SHA-256, raw text)."""
    abspath = os.path.abspath(path)
    with open(abspath) as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise DeepSpeedConfigError(f"{abspath}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DeepSpeedConfigError(f"{abspath}: top level must be an object")
    return data, {"path": abspath, "sha256": sha256_file(abspath), "content": text}


def _zero(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    zero = cfg.get("zero_optimization") or {}
    if not isinstance(zero, dict):
        raise DeepSpeedConfigError("zero_optimization must be an object")
    return zero


def _offload_device(zero: Mapping[str, Any], key: str) -> Optional[str]:
    block = zero.get(key)
    if not block:
        return None
    device = block.get("device") if isinstance(block, dict) else None
    return None if device in (None, "none") else str(device)


def validate(cfg: Mapping[str, Any], *, allow_zero_stage: bool = False, allow_offload: bool = False) -> Dict[str, Any]:
    """Refuse what the file may not say; return the facts a manifest needs."""
    problems = []
    for key in FORBIDDEN_KEYS:
        if key in cfg:
            problems.append(f"'{key}' must not be set: the trainer's own code is its only source")
    for key in PRECISION_KEYS:
        if key in cfg:
            problems.append(f"'{key}' must not be set: precision comes from the model config and is written by MolCrawl")
    for key in BATCH_KEYS:
        if key in cfg and cfg[key] != AUTO:
            problems.append(f"'{key}' must be \"auto\" (it depends on the GPU count), got {cfg[key]!r}")
    zero = _zero(cfg)
    stage = int(zero.get("stage", 0))
    if stage != 0 and not allow_zero_stage:
        problems.append(f"ZeRO stage {stage} requires the report in DeepSpeed order §5 before use")
    offload = {"optimizer": _offload_device(zero, "offload_optimizer"), "param": _offload_device(zero, "offload_param")}
    if any(offload.values()) and not allow_offload:
        problems.append(f"offload {offload} is off in the initial specification (order §5)")
    if problems:
        raise DeepSpeedConfigError("; ".join(problems))
    return {"zero_stage": stage, "offload": offload}


def precision_block(precision: str) -> Dict[str, Any]:
    """DeepSpeed keys for a MolCrawl precision, keeping fp32 parameters in every case."""
    if precision not in PRECISIONS:
        raise DeepSpeedConfigError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    autocast = {"enabled": False}
    if precision == "bf16":
        autocast = {"enabled": True, "dtype": "bfloat16"}
    elif precision == "fp16":
        autocast = {"enabled": True, "dtype": "float16"}
    return {"bf16": {"enabled": False}, "fp16": {"enabled": False}, "torch_autocast": autocast}


def resolve(
    cfg: Mapping[str, Any],
    *,
    framework: str,
    batch: BatchResolution,
    precision: str,
    gradient_clipping: float,
    allow_zero_stage: bool = False,
    allow_offload: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The dict to hand to DeepSpeed, and a record of where each managed value came from.

    ``framework="hf"`` leaves the batch keys as ``"auto"`` for transformers to fill
    from ``TrainingArguments``; ``check_effective`` then compares what the engine
    ended up with against ``batch``. ``framework="custom"`` writes the numbers.
    """
    if framework not in FRAMEWORKS:
        raise DeepSpeedConfigError(f"unknown framework {framework!r}; expected one of {FRAMEWORKS}")
    facts = validate(cfg, allow_zero_stage=allow_zero_stage, allow_offload=allow_offload)
    out = copy.deepcopy(dict(cfg))
    sources: Dict[str, str] = {}

    clip = out.get("gradient_clipping", AUTO)
    if clip != AUTO and float(clip) != float(gradient_clipping):
        raise DeepSpeedConfigError(
            f"gradient_clipping {clip} in the DeepSpeed config does not match the model config's {gradient_clipping}"
        )
    out["gradient_clipping"] = AUTO if framework == "hf" else float(gradient_clipping)
    sources["gradient_clipping"] = "model config (max_grad_norm / grad_clip)"

    expected = {
        "train_micro_batch_size_per_gpu": batch.micro_batch_size_per_gpu,
        "gradient_accumulation_steps": batch.gradient_accumulation_steps_per_rank,
        "train_batch_size": batch.effective_global_batch,
    }
    for key, value in expected.items():
        out[key] = AUTO if framework == "hf" else int(value)
        sources[key] = (
            f"batch policy {batch.policy} ({batch.formula}); filled by transformers from TrainingArguments"
            if framework == "hf" else f"batch policy {batch.policy} ({batch.formula})"
        )

    out.update(precision_block(precision))
    for key in ("bf16", "fp16", "torch_autocast"):
        sources[key] = f"model config precision {precision}, as torch_autocast (fp32 parameters)"

    out.setdefault("zero_optimization", {"stage": 0})
    record = {
        "framework": framework,
        "zero_stage": facts["zero_stage"],
        "offload": facts["offload"],
        "precision": precision,
        "parameter_dtype": "float32",
        "optimizer_created_by": "trainer (client optimizer); DeepSpeed config has no optimizer",
        "scheduler_created_by": "trainer; DeepSpeed config has no scheduler",
        "expected": {**expected, "gradient_clipping": float(gradient_clipping)},
        "sources": sources,
        "resolved": out,
    }
    return out, record


def check_effective(engine_values: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare what the engine reports against what was intended; raise on any difference.

    ``engine_values`` is read off the initialised engine by the caller --
    ``train_batch_size()``, ``train_micro_batch_size_per_gpu()``,
    ``gradient_accumulation_steps()``, ``gradient_clipping()``,
    ``bfloat16_enabled()``, ``fp16_enabled()``, ``torch_autocast_enabled()``,
    ``torch_autocast_dtype()`` -- so this stays testable without DeepSpeed.
    """
    expected = dict(record["expected"])
    precision = record["precision"]
    expected.update({
        "bfloat16_enabled": False,
        "fp16_enabled": False,
        "torch_autocast_enabled": precision != "fp32",
    })
    mismatches = []
    for key, want in expected.items():
        got = engine_values.get(key)
        if isinstance(want, float) or isinstance(got, float):
            same = got is not None and float(got) == float(want)
        else:
            same = got == want
        if not same:
            mismatches.append(f"{key}: engine {got!r} vs intended {want!r}")
    if precision != "fp32":
        want_dtype = "bfloat16" if precision == "bf16" else "float16"
        got_dtype = str(engine_values.get("torch_autocast_dtype", "")).replace("torch.", "")
        if got_dtype != want_dtype:
            mismatches.append(f"torch_autocast_dtype: engine {got_dtype!r} vs intended {want_dtype!r}")
    if mismatches:
        raise DeepSpeedConfigError("DeepSpeed engine differs from the resolved run: " + "; ".join(mismatches))
    return {"checked": sorted(expected) + (["torch_autocast_dtype"] if precision != "fp32" else []), "ok": True}
