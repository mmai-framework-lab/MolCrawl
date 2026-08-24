"""Diagnostics for MLM pretraining.

Background (investigation of 2026-08-06):

``DataCollatorForLanguageModeling`` replaces 80% of the selected positions with
``[MASK]``, 10% with a random token, and leaves the remaining 10% **unchanged**.
That last 10% has the answer sitting in the input, so the model can copy it —
measured at loss 0.029 on a run whose ``[MASK]`` positions scored 2.453. A single
blended ``eval_loss`` therefore reads 2.13 for a model that has learned almost no
context, which is how a total training failure went unnoticed.

Scoring is split by position type here, and a run is judged on the ``[MASK]``
positions alone. Reference points for the compounds packed data:

    unigram 2.635 / degenerate (unigram + copy) 2.395 / left-neighbour table 2.066

A run whose ``[MASK]`` loss never drops below the degenerate 2.395 has not learned
context and can be treated as failed.

Collapse into that degenerate solution is also **unrecoverable**: lowering the
learning rate, redoing warmup and rebuilding the optimizer for 800 further steps
moved the loss by 0.0005. A collapsed run only burns GPU time, so it is stopped.
"""

from __future__ import annotations

import torch

try:
    from transformers import TrainerCallback
except ImportError:  # documentation-only environments
    TrainerCallback = object  # type: ignore[assignment,misc]

IGNORE_INDEX = -100


def split_mlm_loss(logits, labels, input_ids, mask_token_id):
    """Aggregate the MLM loss separately for each type of scored position.

    Returns ``(sums, counts)``, both keyed by ``"mask"`` / ``"copy"`` / ``"random"``.
    Sums and counts rather than means, so distributed ranks can be combined.

    - ``mask``   : input is ``[MASK]``; the model must infer from context.
    - ``copy``   : the answer is still in the input (the 10% left unchanged).
    - ``random`` : the input was replaced with a random token.
    """
    scored = labels != IGNORE_INDEX
    if not bool(scored.any()):
        zero = {k: 0.0 for k in ("mask", "copy", "random")}
        return zero, {k: 0 for k in ("mask", "copy", "random")}

    flat_logits = logits[scored]
    flat_labels = labels[scored]
    flat_inputs = input_ids[scored]
    ce = torch.nn.functional.cross_entropy(
        flat_logits.float(), flat_labels, reduction="none"
    )

    is_mask = flat_inputs == mask_token_id
    is_copy = (~is_mask) & (flat_inputs == flat_labels)
    is_random = (~is_mask) & (flat_inputs != flat_labels)

    sums, counts = {}, {}
    for name, sel in (("mask", is_mask), ("copy", is_copy), ("random", is_random)):
        sums[name] = float(ce[sel].sum()) if bool(sel.any()) else 0.0
        counts[name] = int(sel.sum())
    return sums, counts


class MlmBreakdownMixin:
    """Trainer mixin that reports the per-position-type eval breakdown.

    ``prediction_step`` accumulates the split losses and ``evaluate`` adds
    ``eval_loss_mask`` / ``eval_loss_copy`` / ``eval_loss_random`` to the returned
    metrics and to the logs. ``eval_loss`` itself is left as HF computes it, so
    comparisons with earlier runs still hold.
    """

    mlm_mask_token_id: int | None = None

    def _reset_mlm_breakdown(self):
        self._mlm_sums = {k: 0.0 for k in ("mask", "copy", "random")}
        self._mlm_counts = {k: 0 for k in ("mask", "copy", "random")}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        out = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        if self.mlm_mask_token_id is None:
            return out
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        if labels is None or input_ids is None:
            return out
        if not hasattr(self, "_mlm_sums"):
            self._reset_mlm_breakdown()
        with torch.no_grad():
            # Forward with everything the collator supplied except the labels, so this
            # runs the model the way training ran it. Naming input_ids and
            # attention_mask alone used to drop position_ids, which DocumentMaskingCollator
            # resets per document: the model then saw 0..N-1 across a packed block it had
            # been trained to read per molecule, and the reported [MASK] loss came out
            # about six times too high. Measured on the W3 checkpoint: 0.141 with
            # position_ids, 0.869 without, against 0.871 in that run's own log - while
            # HF's own eval_loss, which does forward properly, stayed at 0.129. The two
            # numbers in that log could never have been true at once.
            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            logits = model(**forward_inputs).logits
            sums, counts = split_mlm_loss(logits, labels, input_ids, self.mlm_mask_token_id)
        for k in sums:
            self._mlm_sums[k] += sums[k]
            self._mlm_counts[k] += counts[k]
        return out

    def evaluate(self, *args, **kwargs):
        self._reset_mlm_breakdown()
        metrics = super().evaluate(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")
        extra = {}
        for k in ("mask", "copy", "random"):
            n = self._mlm_counts.get(k, 0)
            if n:
                extra[f"{prefix}_loss_{k}"] = self._mlm_sums[k] / n
        if extra:
            metrics.update(extra)
            self.log(extra)
        return metrics


class CollapseDetectionCallback(TrainerCallback):
    """Detect collapse into the degenerate solution and stop training.

    Collapse is unrecoverable, so continuing only wastes compute. Two signals:

    1. The first logged ``grad_norm`` exceeding ``max_grad_norm`` by
       ``grad_norm_factor`` — the signature of a warmup-starved start (6.4 on the
       run that collapsed, 2.5 on the one that survived). This only **warns**:
       deep models can start large and still recover.
    2. ``eval_loss_mask`` staying above the degenerate threshold for ``patience``
       consecutive evaluations, once past ``grace_steps``. This **stops** training.

    ``degenerate_threshold`` is modality-specific — roughly the unigram entropy of
    the maskable tokens scaled by the non-copy fraction; 2.395 for compounds
    packed. Stopping is disabled when it is not set.

    **Why the grace period.** MLM does not descend from the start: it sits at the
    degenerate level and then breaks out. Measured plateaus are 1,500-2,000 steps
    (compounds seq 128, mol_nl) and 2,750 steps (compounds packed with document
    masking). ``patience`` alone cannot express that, because it counts evaluations
    and the eval interval differs per config — at protein's ``log_interval=100`` the
    old default of 3 fired after 300 steps, a tenth of the way into a plateau every
    healthy run has to cross. ``grace_steps`` is in optimizer steps, so it means the
    same thing whatever the eval cadence, and defaults past the longest plateau
    observed with margin.
    """

    #: Optimizer steps before the degenerate check starts counting. Longest observed
    #: plateau is 2,750 steps; this leaves ~1.8x margin.
    DEFAULT_GRACE_STEPS = 5000

    def __init__(
        self,
        degenerate_threshold=None,
        patience=5,
        grad_norm_factor=3.0,
        grace_steps=DEFAULT_GRACE_STEPS,
    ):
        self.degenerate_threshold = degenerate_threshold
        self.patience = patience
        self.grad_norm_factor = grad_norm_factor
        self.grace_steps = grace_steps
        self._over = 0
        self._warned_grad = False
        self._last_checked_step = None
        self._stopped = False

    def _check_degenerate(self, value, state, control):
        """Count consecutive evals above the threshold and stop once patience runs out.

        Guarded by ``global_step`` so an eval seen through both hooks is counted once,
        and by ``grace_steps`` so the plateau every healthy run crosses is not read as
        collapse.

        A level threshold cannot tell a stalled run from a merely slow one, which
        is why genome leaves this off. Measured there over 8,000 steps
        (2026-08-24, jobs 35233 / 35234 / 35235), all three learning rates cross
        the degenerate line early -- 1e-4 at step 300, 3e-5 at 400, 1e-5 at 700 --
        so crossing it says nothing. Afterwards 1e-5 does not move at all while
        3e-5 keeps improving, yet at step 5,000 they sit only 0.008 apart
        (1.3510 against 1.3430): any threshold that catches the stalled one also
        catches the slow one.

        What separates them is the slope, not the level. Improvement from step
        6,000 to 8,000 was +0.0330 (1e-4), +0.0072 (3e-5), -0.0010 (1e-5) --
        ordered, and unambiguous. This detector only reads levels, so acting on
        that would mean changing what it consumes. Recorded here as the input for
        whoever next revisits it.
        """
        if not self.degenerate_threshold or value is None or self._stopped:
            return control
        step = getattr(state, "global_step", None) if state is not None else None
        if step is not None and step == self._last_checked_step:
            return control
        self._last_checked_step = step

        if self.grace_steps and step is not None and step < self.grace_steps:
            # Still inside the plateau every healthy run has to cross.
            return control

        if value > self.degenerate_threshold:
            self._over += 1
            print(
                f"WARNING: eval_loss_mask={value:.4f} is above the degenerate "
                f"threshold {self.degenerate_threshold} ({self._over}/{self.patience})",
                flush=True,
            )
            if self._over >= self.patience:
                print(
                    "STOPPING: the run has collapsed into the degenerate solution. "
                    "No recovery from this state has been observed — lowering the LR, "
                    "redoing warmup and rebuilding the optimizer for 800 steps moved "
                    "the loss by 0.0005.",
                    flush=True,
                )
                control.should_training_stop = True
                self._stopped = True
        else:
            self._over = 0
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        if not self._warned_grad:
            gn = logs.get("grad_norm")
            if gn is not None:
                self._warned_grad = True
                limit = args.max_grad_norm * self.grad_norm_factor
                if gn > limit:
                    print(
                        f"WARNING: first grad_norm={gn:.3f} exceeds clip={args.max_grad_norm} "
                        f"by more than {self.grad_norm_factor}x. That is what a warmup-starved "
                        f"start looks like (6.4 on the run that collapsed, 2.5 on the one that "
                        f"survived) — watch eval_loss_mask.",
                        flush=True,
                    )

        # The breakdown reaches the callbacks here, not through on_evaluate.
        # Trainer.evaluate() fires on_evaluate with its own metrics dict and only
        # then returns it, so MlmBreakdownMixin.evaluate() — which adds
        # eval_loss_mask to the returned dict afterwards — is too late for that
        # hook. The mixin's own log(extra) call is the first point at which a
        # callback can see the value.
        if "eval_loss_mask" in logs:
            control = self._check_degenerate(logs.get("eval_loss_mask"), state, control)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Kept so the detector still works if the metric ever lands in the eval
        # metrics before this hook runs; the step guard stops it double counting.
        if not metrics:
            return control
        return self._check_degenerate(metrics.get("eval_loss_mask"), state, control)
