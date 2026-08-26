# Resuming a run under torch >= 2.6

## The defect

torch 2.6 flipped the default of `torch.load` from `weights_only=False` to
`weights_only=True`. Every training loop here writes `numpy.random.get_state()`
into its checkpoint so a resumed run continues the same RNG stream, and numpy's
state contains arrays that the safe loader refuses to unpickle:

```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global:
    GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default.
```

A loop whose `torch.load` calls pass only `map_location` therefore **cannot read
the checkpoints it wrote itself**. Only `training_state.bin` and
`rng_state_*.pth` are affected; `pytorch_model.bin` is plain tensors and loads
either way.

## How it was found, and why so late

RNA `gpt2-xl` is the first production run in this project that did not fit in a
single job — 20.4 s/iter against a 4-day walltime — so it became the first to
execute a resume at all. The resubmission (job 28851, 2026-08-17) died 23 seconds
in. Every GPT-2 production run before it had finished inside one job, so the load
path had never been exercised, and the checkpoints written along the way looked
fine because writing them works.

The fix on the GPT-2 side is `9aaa92a`.

## Why the shim existed but was not wired up

`molcrawl/core/torch_compat.py` has handled exactly this since the HF Trainer
entry points hit it, and five of them call `enable_full_torch_load()` at import.
GPT-2 did not, because the shim's own docstring said it did not need to:

> GPT-2 (`molcrawl/models/gpt2/train.py`) is unaffected because it does not go
> through HF Trainer's resume path; it manages its own checkpoint load with
> explicit `weights_only=False` semantics.

GPT-2 never passed `weights_only` at all. The docstring described an intent that
was never implemented, and reading it was enough to conclude no work was needed.
It has been corrected.

## What to check when adding a training loop

Any loop that does not go through HF Trainer needs
`enable_full_torch_load()` at import, or `weights_only=False` at every
`torch.load` call site. As of 2026-08-24 the non-HF loops are
`models/gpt2/train.py` and `models/llama/train.py`; both are covered.

**Writing a checkpoint is not evidence that it can be read.** Before launching a
run whose walltime makes a resume likely, load the newest checkpoint back:

```bash
python - <<'PY'
import torch
from molcrawl.core.torch_compat import enable_full_torch_load
enable_full_torch_load()
ck = torch.load("<out_dir>/checkpoint-<step>/training_state.bin", map_location="cpu")
print(ck["iter_num"], ck["best_val_loss"], "rng_state" in ck)
PY
```

This check was skipped for the RNA ladder — the pre-flight confirmed checkpoints
were being saved every 1,000 steps and stopped there — which is why the failure
surfaced at resubmission time instead of before launch.

## Related

A requeue by SLURM (`Restarts=` in `scontrol show job`) restarts the batch script
from the top and takes the same resume path, so this defect also broke
requeue-driven restarts, not just manual resubmission. Note that a requeue also
truncates `%x-%j.out`: the earlier portion of the log is lost, and
`best_val_loss` has to be read from the checkpoint instead.
