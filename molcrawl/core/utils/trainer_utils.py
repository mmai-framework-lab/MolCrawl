"""Utilities for HuggingFace Trainer workarounds."""


def install_tie_weights_on_resume(trainer):
    """Patch trainer to re-tie embedding weights after checkpoint resume.

    HuggingFace Trainer saves checkpoints using safetensors by default.
    safetensors cannot represent tied tensors (shared storage), so only one
    copy of the tied weight is written. When resuming, the decoder weight
    (e.g. ``cls.predictions.decoder.weight`` for BERT or
    ``lm_head.decoder.weight`` for ESM-2) is not restored from the checkpoint
    and the model keeps its randomly initialised value unless
    ``model.tie_weights()`` is called.

    This helper wraps ``trainer._load_from_checkpoint`` so that
    ``tie_weights()`` is invoked right after the state dict is loaded.
    Existing checkpoints that were saved without the tied key are therefore
    still usable: the decoder is restored from the embedding.
    """
    original_load = trainer._load_from_checkpoint

    def patched_load(resume_from_checkpoint, model=None):
        original_load(resume_from_checkpoint, model)
        target = model if model is not None else trainer.model
        if hasattr(target, "tie_weights"):
            target.tie_weights()
            if getattr(trainer.args, "local_rank", -1) in (-1, 0):
                print("tie_weights() re-applied after checkpoint load")

    trainer._load_from_checkpoint = patched_load
