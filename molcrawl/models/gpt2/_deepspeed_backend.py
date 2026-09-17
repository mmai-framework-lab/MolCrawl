"""The DeepSpeed backend for gpt2/train.py, kept to what the legacy loop already does.

DeepSpeed order §7 asks for the engine inside the existing custom loop without
changing a research condition (§2). Each piece below is chosen so that a run with
``deepspeed_config`` set trains the same numbers the DDP path would, and differs
only in who holds the process group and reduces the gradients:

optimizer
    ``GPT.configure_optimizers`` still builds it -- fused AdamW on CUDA, embeddings
    and 1-D tensors out of weight decay -- and it is handed to
    ``deepspeed.initialize`` as a client optimizer. At ZeRO stage 0 with fp32
    parameters DeepSpeed uses it unwrapped (engine.py:1974-1977, 2033-2034);
    ``initialize`` checks that ``engine.optimizer is optimizer`` and refuses
    otherwise. Its ``param_groups`` are therefore the ones train.py sets the
    learning rate on, and its ``state_dict`` is what checkpoints already save.
precision
    fp32 parameters, forward under ``torch.autocast`` -- by DeepSpeed's
    ``torch_autocast``, since the engine switches off an autocast opened outside
    it (torch_autocast.py:107-128). ``float16`` is refused: the legacy loop pairs
    it with its own GradScaler, which this path does not reproduce. GB200 runs
    resolve ``dtype`` to ``bfloat16``.
loss and gradients
    The loss is divided by the accumulation count exactly as before and passed to
    ``engine.backward(loss, scale_wrt_gas=False)``, so DeepSpeed does not divide
    again. Gradients are all-reduced at the accumulation boundary only, as DDP's
    ``require_backward_grad_sync`` did.
clipping
    ``torch.nn.utils.clip_grad_norm_`` at the boundary, after the reduction and
    before ``engine.step()``, as before. DeepSpeed's own clipping is set to 0: its
    ``clip_grad_norm_`` (runtime/utils.py:359) computes the norm differently.
checkpoints
    Unchanged. At stage 0 the model and optimizer are the client objects, so
    ``checkpoint-N/`` and ``ckpt.pt`` keep their nanoGPT format, no DeepSpeed
    checkpoint is written, and existing evaluators read them as before.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from molcrawl.models import _deepspeed_config as dsc
from molcrawl.models._batch_policy import from_legacy_gpt2, resolve_global_fixed

PRECISION_FOR_DTYPE = {"bfloat16": "bf16", "float32": "fp32"}


def check_preconditions(*, deepspeed_config: str, ddp: bool, device_type: str, dtype: str) -> str:
    """Refuse, before any data is loaded, a run this backend would not train faithfully."""
    problems = []
    if not ddp:
        problems.append("it needs a distributed launch (RANK set): use workflows/deepspeed-train.sbatch")
    if device_type != "cuda":
        problems.append(f"it runs on CUDA only, device_type is {device_type!r}")
    if dtype not in PRECISION_FOR_DTYPE:
        problems.append(
            f"dtype {dtype!r} is not supported: float16 would need the legacy GradScaler, which this path does not reproduce"
        )
    if problems:
        raise SystemExit(f"deepspeed_config={deepspeed_config!r} cannot be used: " + "; ".join(problems))
    try:
        import deepspeed  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"deepspeed_config={deepspeed_config!r} is set but deepspeed is not importable: {exc}") from exc
    return PRECISION_FOR_DTYPE[dtype]


def _engine_values(engine) -> Dict[str, Any]:
    return {
        "train_batch_size": engine.train_batch_size(),
        "train_micro_batch_size_per_gpu": engine.train_micro_batch_size_per_gpu(),
        "gradient_accumulation_steps": engine.gradient_accumulation_steps(),
        "gradient_clipping": engine.gradient_clipping(),
        "bfloat16_enabled": engine.bfloat16_enabled(),
        "fp16_enabled": engine.fp16_enabled(),
        "torch_autocast_enabled": engine.torch_autocast_enabled(),
        "torch_autocast_dtype": str(engine.torch_autocast_dtype()),
    }


def initialize(
    model,
    optimizer,
    *,
    deepspeed_config: str,
    batch_size: int,
    gradient_accumulation_steps_configured: int,
    gradient_accumulation_steps_per_rank: int,
    world_size: int,
    precision: str,
    grad_clip: float,
    deepspeed_module=None,
) -> Tuple[Any, Dict[str, Any]]:
    """Wrap ``model`` in a DeepSpeed engine; return it and the manifest's ``deepspeed`` record."""
    if deepspeed_module is None:
        import deepspeed as deepspeed_module  # noqa: N813

    batch = resolve_global_fixed(from_legacy_gpt2(batch_size, gradient_accumulation_steps_configured), world_size)
    if batch.gradient_accumulation_steps_per_rank != gradient_accumulation_steps_per_rank:
        raise SystemExit(
            f"batch policy gives per-rank accumulation {batch.gradient_accumulation_steps_per_rank}"
            f" but train.py resolved {gradient_accumulation_steps_per_rank}; refusing to start"
        )
    raw, ident = dsc.load(deepspeed_config)
    config, record = dsc.resolve(
        raw, framework="custom", batch=batch, precision=precision,
        gradient_clipping=grad_clip, clipping_by="trainer",
    )
    engine, engine_optimizer, _, _ = deepspeed_module.initialize(model=model, optimizer=optimizer, config=config)
    if getattr(engine, "optimizer", None) is not optimizer:
        raise SystemExit(
            "DeepSpeed wrapped the client optimizer"
            f" ({type(getattr(engine, 'optimizer', None)).__name__}); the optimizer implementation and the"
            " checkpoint format would change. Only ZeRO stage 0 with fp32 parameters is supported here."
        )
    values = _engine_values(engine)
    checked = dsc.check_effective(values, record)
    manifest = {
        "enabled": True,
        "version": getattr(deepspeed_module, "__version__", None),
        "config_file": ident,
        **{k: v for k, v in record.items() if k != "resolved"},
        "config_passed_to_initialize": record["resolved"],
        "engine_effective": values,
        "engine_check": checked,
        "engine_config": getattr(engine, "config", None),
        "optimizer_is_client_optimizer": True,
        "loss_scaling": "loss / gradient_accumulation_steps in train.py; engine.backward(scale_wrt_gas=False)",
        "checkpoint": {
            "format": "unchanged nanoGPT: checkpoint-<step>/training_state.bin, ckpt.pt",
            "sharded": False,
            "shards": 1,
            "deepspeed_checkpoint_written": False,
            "resume_requires_same_world_size": False,
            "resume_condition": "the batch policy must divide at the new world size (global_fixed)",
            "conversion_for_evaluators": "not needed",
        },
    }
    return engine, manifest


def micro_steps(
    engine,
    X,
    Y,
    *,
    gradient_accumulation_steps: int,
    grad_clip: float,
    get_batch: Callable[[str], Tuple[Any, Any]],
    ctx,
    clip_grad_norm: Optional[Callable] = None,
):
    """One optimizer step through the engine: the legacy loop's arithmetic, in order.

    Returns the next ``X, Y`` and the last micro-step's (divided) loss, which
    train.py logs as before.
    """
    if clip_grad_norm is None:
        import torch

        clip_grad_norm = torch.nn.utils.clip_grad_norm_
    loss = None
    for _ in range(gradient_accumulation_steps):
        with ctx:
            _, loss = engine(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        engine.backward(loss, scale_wrt_gas=False)
        if engine.is_gradient_accumulation_boundary() and grad_clip != 0.0:
            clip_grad_norm(engine.module.parameters(), grad_clip)
        engine.step()
    return X, Y, loss
