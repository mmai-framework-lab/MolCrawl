"""One reading of "global batch" for both trainers, and what it allows.

The two trainers read the same two config names in opposite ways:

``gpt2/train.py`` (nanoGPT)
    ``gradient_accumulation_steps`` in a config is the *total* over all ranks.
    train.py asserts it divides by the world size and then does
    ``gradient_accumulation_steps //= world_size``, so the effective batch
    ``batch_size * gradient_accumulation_steps`` does not move with the GPU
    count -- and a count it does not divide is an AssertionError.
``bert/main.py`` (HF Trainer)
    ``gradient_accumulation_steps`` is *per rank*. The effective batch is
    ``batch_size * gradient_accumulation_steps * world_size`` and does move with
    the GPU count. Only a config that declares ``expected_global_batch`` notices.

A DeepSpeed launcher that places a run on N nodes has to know which number it
is holding fixed, and a manifest has to say so. This module is that one reading:

``global_fixed``
    Normal training. ``target_global_batch`` is authoritative and the per-rank
    accumulation is derived from it::

        per_rank = target_global_batch / (micro_batch_size_per_gpu * world_size)

    A quotient that is not a positive integer is refused, never rounded.
``per_gpu_fixed``
    Throughput benchmarks only, and only when asked for by name. The micro batch
    and the *per-rank* accumulation are fixed and the global batch grows with
    the world size. Such a run's loss is not a training result.

Legacy configs keep their meaning. They are read through ``from_legacy_gpt2`` /
``from_legacy_bert`` into the canonical pair (micro batch, target global batch);
no config file changes. For nanoGPT the canonical per-rank value is
``accum_configured / world_size`` -- the same division train.py does, with the
same condition for failing -- so every existing GPT-2 run resolves to the batch
it ran with. ``tests/unit/test_batch_policy_legacy_configs.py`` checks that for
every ``gpt2*.py`` config at every world size, rather than trusting the algebra.

A BERT config that does not declare ``expected_global_batch`` has no target to
read, and one is not invented from ``batch_size * accum * 4``: which world size
a config was written for is exactly what nothing records. ``from_legacy_bert``
refuses instead, and the caller reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional

GLOBAL_FIXED = "global_fixed"
PER_GPU_FIXED = "per_gpu_fixed"
POLICIES = (GLOBAL_FIXED, PER_GPU_FIXED)

# Rikyu GB200 nodes carry 4 GPUs each.
DEFAULT_GPUS_PER_NODE = 4
# The node counts a DeepSpeed scaling series is drawn from.
DEFAULT_CANDIDATE_NODES = (1, 2, 4, 8)


class BatchPolicyError(ValueError):
    """A batch that cannot be resolved without rounding or guessing."""


@dataclass(frozen=True)
class CanonicalBatch:
    """The two numbers a run's batch is defined by, and where they came from."""

    micro_batch_size_per_gpu: int
    target_global_batch: int
    source: str
    legacy: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResolution:
    """A batch resolved for one world size, with the inputs the formula used."""

    policy: str
    micro_batch_size_per_gpu: int
    world_size: int
    gradient_accumulation_steps_per_rank: int
    effective_global_batch: int
    target_global_batch: Optional[int]
    formula: str
    source: Optional[str] = None
    legacy: Dict[str, Any] = field(default_factory=dict)

    def global_tokens_per_optimizer_step(self, sequence_length: int) -> int:
        return self.effective_global_batch * int(sequence_length)

    def as_manifest(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "batch_policy": self.policy,
            "target_global_batch": self.target_global_batch,
            "micro_batch_size_per_gpu": self.micro_batch_size_per_gpu,
            "world_size": self.world_size,
            "gradient_accumulation_steps_per_rank": self.gradient_accumulation_steps_per_rank,
            "effective_global_batch": self.effective_global_batch,
            "formula": self.formula,
            "target_source": self.source,
            "legacy": dict(self.legacy),
        }
        if sequence_length is not None:
            out["sequence_length"] = int(sequence_length)
            out["global_tokens_per_optimizer_step"] = self.global_tokens_per_optimizer_step(sequence_length)
        return out


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchPolicyError(f"{name} must be an int, got {value!r} ({type(value).__name__})")
    if value < 1:
        raise BatchPolicyError(f"{name} must be >= 1, got {value}")
    return value


def from_legacy_gpt2(batch_size: int, gradient_accumulation_steps_configured: int) -> CanonicalBatch:
    """nanoGPT config -> canonical. The configured accumulation is the total."""
    micro = _positive_int("batch_size", batch_size)
    accum = _positive_int("gradient_accumulation_steps", gradient_accumulation_steps_configured)
    return CanonicalBatch(
        micro_batch_size_per_gpu=micro,
        target_global_batch=micro * accum,
        source="legacy gpt2: batch_size x gradient_accumulation_steps (configured total)",
        legacy={"batch_size": micro, "gradient_accumulation_steps_configured": accum},
    )


def from_legacy_bert(
    batch_size: int,
    gradient_accumulation_steps: int,
    expected_global_batch: Optional[int],
) -> CanonicalBatch:
    """HF config -> canonical, only when the config states its global batch."""
    micro = _positive_int("batch_size", batch_size)
    accum = _positive_int("gradient_accumulation_steps", gradient_accumulation_steps)
    if expected_global_batch is None:
        raise BatchPolicyError(
            "this BERT config does not declare expected_global_batch, so it has no"
            " target global batch to hold fixed. It is not derived from"
            f" batch_size x gradient_accumulation_steps x world_size ({micro} x {accum} x ?):"
            " the world size the config was written for is not recorded anywhere."
            " Declaring it is a research decision outside this code."
        )
    target = _positive_int("expected_global_batch", expected_global_batch)
    return CanonicalBatch(
        micro_batch_size_per_gpu=micro,
        target_global_batch=target,
        source="legacy bert: expected_global_batch",
        legacy={
            "batch_size": micro,
            "gradient_accumulation_steps_configured": accum,
            "expected_global_batch": target,
        },
    )


def _per_rank(target: int, micro: int, world_size: int) -> Fraction:
    return Fraction(target, micro * world_size)


def resolve_global_fixed(canonical: CanonicalBatch, world_size: int) -> BatchResolution:
    """Hold the global batch; derive the per-rank accumulation or refuse."""
    world = _positive_int("world_size", world_size)
    micro = canonical.micro_batch_size_per_gpu
    target = canonical.target_global_batch
    quotient = _per_rank(target, micro, world)
    if quotient.denominator != 1 or quotient < 1:
        raise BatchPolicyError(
            f"global_fixed: target_global_batch {target} / (micro_batch_size_per_gpu {micro}"
            f" x world_size {world}) = {float(quotient):g}, which is not a positive integer."
            " The accumulation is not rounded; choose a world size or micro batch that divides."
        )
    per_rank = int(quotient)
    return BatchResolution(
        policy=GLOBAL_FIXED,
        micro_batch_size_per_gpu=micro,
        world_size=world,
        gradient_accumulation_steps_per_rank=per_rank,
        effective_global_batch=micro * per_rank * world,
        target_global_batch=target,
        formula=f"{target} / ({micro} x {world}) = {per_rank}",
        source=canonical.source,
        legacy=canonical.legacy,
    )


def resolve_per_gpu_fixed(
    micro_batch_size_per_gpu: int,
    gradient_accumulation_steps_per_rank: int,
    world_size: int,
) -> BatchResolution:
    """Benchmark policy: the global batch is whatever the world size makes it."""
    micro = _positive_int("micro_batch_size_per_gpu", micro_batch_size_per_gpu)
    per_rank = _positive_int("gradient_accumulation_steps_per_rank", gradient_accumulation_steps_per_rank)
    world = _positive_int("world_size", world_size)
    effective = micro * per_rank * world
    return BatchResolution(
        policy=PER_GPU_FIXED,
        micro_batch_size_per_gpu=micro,
        world_size=world,
        gradient_accumulation_steps_per_rank=per_rank,
        effective_global_batch=effective,
        target_global_batch=None,
        formula=f"{micro} x {per_rank} x {world} = {effective}",
        source="per_gpu_fixed: micro batch and per-rank accumulation held fixed",
    )


def resolve(
    policy: str,
    world_size: int,
    *,
    canonical: Optional[CanonicalBatch] = None,
    micro_batch_size_per_gpu: Optional[int] = None,
    gradient_accumulation_steps_per_rank: Optional[int] = None,
) -> BatchResolution:
    """Dispatch on the policy name. Unknown names are refused, not defaulted."""
    if policy == GLOBAL_FIXED:
        if canonical is None:
            raise BatchPolicyError("global_fixed needs a canonical batch (micro batch and target global batch)")
        return resolve_global_fixed(canonical, world_size)
    if policy == PER_GPU_FIXED:
        if micro_batch_size_per_gpu is None or gradient_accumulation_steps_per_rank is None:
            raise BatchPolicyError(
                "per_gpu_fixed needs micro_batch_size_per_gpu and gradient_accumulation_steps_per_rank"
            )
        return resolve_per_gpu_fixed(micro_batch_size_per_gpu, gradient_accumulation_steps_per_rank, world_size)
    raise BatchPolicyError(f"unknown batch_policy {policy!r}; expected one of {POLICIES}")


def scaling_feasibility(
    canonical: CanonicalBatch,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    candidate_nodes: Iterable[int] = DEFAULT_CANDIDATE_NODES,
) -> Dict[str, Any]:
    """Which node counts a global_fixed series can reach, and why the rest cannot.

    Computed before a scaling series is planned and written into every manifest
    of that series, so a series that stops at 4 nodes says it stopped because
    8 would need 2.5 accumulation steps -- not because nobody tried.
    """
    gpn = _positive_int("gpus_per_node", gpus_per_node)
    candidates: List[Dict[str, Any]] = []
    for nodes in candidate_nodes:
        n = _positive_int("nodes", nodes)
        world = n * gpn
        quotient = _per_rank(canonical.target_global_batch, canonical.micro_batch_size_per_gpu, world)
        if quotient.denominator == 1 and quotient >= 1:
            feasible, reason, per_rank = True, None, int(quotient)
        elif quotient < 1:
            feasible, per_rank = False, None
            reason = f"per-rank accumulation {float(quotient):g} < 1"
        else:
            feasible, per_rank = False, None
            reason = f"per-rank accumulation {float(quotient):g} is not an integer"
        candidates.append(
            {
                "nodes": n,
                "world_size": world,
                "gradient_accumulation_steps_per_rank": per_rank,
                "quotient": float(quotient),
                "feasible": feasible,
                "reason": reason,
            }
        )
    feasible_nodes = [c["nodes"] for c in candidates if c["feasible"]]
    return {
        "batch_policy": GLOBAL_FIXED,
        "target_global_batch": canonical.target_global_batch,
        "micro_batch_size_per_gpu": canonical.micro_batch_size_per_gpu,
        "gpus_per_node": gpn,
        "candidates": candidates,
        "max_feasible_candidate_node_count": max(feasible_nodes) if feasible_nodes else None,
    }
