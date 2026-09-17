"""gpt2 DeepSpeed backend without DeepSpeed: preconditions, initialisation, loop arithmetic.

DeepSpeed is not installed in this environment, so the engine is a stand-in that
reproduces the parts of DeepSpeed 0.19.5 the backend relies on:
``backward(loss, scale_wrt_gas)`` divides only when asked
(engine.py:3193, 2954-2955), the accumulation boundary is
``(micro_steps + 1) % gas == 0`` (engine.py:3218-3241), and ``step()`` applies the
client optimizer and clears gradients at the boundary only (engine.py:3281-3320).

The equivalence test is the point: the same tiny GPT, data and seed, one run
through the legacy loop's lines and one through ``micro_steps``, must end with
bit-identical parameters -- including with clipping active.
"""

import json
from contextlib import nullcontext

import pytest
import torch

from molcrawl.models import _deepspeed_config as dsc
from molcrawl.models.gpt2 import _deepspeed_backend as backend
from molcrawl.models.gpt2.model import GPT, GPTConfig


class FakeEngine(torch.nn.Module):
    def __init__(self, model, optimizer, config, wrap_optimizer=False):
        super().__init__()
        self.module = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1) if wrap_optimizer else optimizer
        self._config = config
        self.micro_steps = 0
        self.backward_scale_args = []
        self.config = config

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss, scale_wrt_gas=True):
        self.backward_scale_args.append(scale_wrt_gas)
        (loss / self.gradient_accumulation_steps() if scale_wrt_gas else loss).backward()

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % self.gradient_accumulation_steps() == 0

    def step(self):
        if self.is_gradient_accumulation_boundary():
            self.optimizer.step()
            for p in self.module.parameters():
                p.grad = None
        self.micro_steps += 1

    def train_batch_size(self):
        return self._config["train_batch_size"]

    def train_micro_batch_size_per_gpu(self):
        return self._config["train_micro_batch_size_per_gpu"]

    def gradient_accumulation_steps(self):
        return self._config["gradient_accumulation_steps"]

    def gradient_clipping(self):
        return self._config["gradient_clipping"]

    def bfloat16_enabled(self):
        return self._config["bf16"]["enabled"]

    def fp16_enabled(self):
        return self._config["fp16"]["enabled"]

    def torch_autocast_enabled(self):
        return self._config["torch_autocast"]["enabled"]

    def torch_autocast_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self._config["torch_autocast"].get("dtype"))


class FakeDeepSpeed:
    __version__ = "0.0-fake"

    def __init__(self, wrap_optimizer=False):
        self.wrap_optimizer = wrap_optimizer
        self.passed_config = None

    def initialize(self, model, optimizer, config):
        self.passed_config = config
        engine = FakeEngine(model, optimizer, config, wrap_optimizer=self.wrap_optimizer)
        return engine, engine.optimizer, None, None


# ------------------------------------------------------------ preconditions ---- #

@pytest.mark.parametrize(
    "kw,fragment",
    [(dict(ddp=False, device_type="cuda", dtype="bfloat16"), "distributed launch"),
     (dict(ddp=True, device_type="cpu", dtype="bfloat16"), "CUDA only"),
     (dict(ddp=True, device_type="cuda", dtype="float16"), "GradScaler")],
)
def test_preconditions_refuse_what_would_not_train_the_same(kw, fragment):
    with pytest.raises(backend.DeepSpeedBackendError, match=fragment):
        backend.check_preconditions(deepspeed_config="x.json", **kw)


def test_preconditions_report_a_missing_deepspeed(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "deepspeed", None)
    with pytest.raises(backend.DeepSpeedBackendError, match="not importable"):
        backend.check_preconditions(deepspeed_config="x.json", ddp=True, device_type="cuda", dtype="bfloat16")


# --------------------------------------------------------------- initialize ---- #

def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False))


def _ds_file(tmp_path, body=None):
    path = tmp_path / "gpt2_tiny.deepspeed.json"
    path.write_text(json.dumps(body if body is not None else dsc.default_config()))
    return str(path)


def _init(tmp_path, fake, **over):
    model = _tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    kw = dict(deepspeed_config=_ds_file(tmp_path), batch_size=16, gradient_accumulation_steps_configured=160,
              gradient_accumulation_steps_per_rank=40, world_size=4, precision="bf16", grad_clip=1.0,
              deepspeed_module=fake)
    kw.update(over)
    return model, optimizer, backend.initialize(model, optimizer, **kw)


def test_initialize_hands_deepspeed_the_resolved_config(tmp_path):
    fake = FakeDeepSpeed()
    model, optimizer, (engine, record) = _init(tmp_path, fake)
    cfg = fake.passed_config
    assert (cfg["train_micro_batch_size_per_gpu"], cfg["gradient_accumulation_steps"], cfg["train_batch_size"]) == (16, 40, 2560)
    assert cfg["gradient_clipping"] == 0.0
    assert cfg["torch_autocast"] == {"enabled": True, "dtype": "bfloat16"}
    assert cfg["bf16"] == {"enabled": False}
    assert "optimizer" not in cfg and "scheduler" not in cfg
    assert engine.optimizer is optimizer
    assert record["enabled"] is True and record["version"] == "0.0-fake"
    assert record["gradient_clipping"] == {"value": 1.0, "applied_by": "trainer"}
    assert record["engine_check"]["ok"] is True
    assert record["checkpoint"]["deepspeed_checkpoint_written"] is False
    assert len(record["config_file"]["sha256"]) == 64


def test_initialize_refuses_a_wrapped_optimizer(tmp_path):
    with pytest.raises(backend.DeepSpeedBackendError, match="wrapped the client optimizer"):
        _init(tmp_path, FakeDeepSpeed(wrap_optimizer=True))


def test_initialize_refuses_a_per_rank_accumulation_train_py_did_not_resolve(tmp_path):
    with pytest.raises(backend.DeepSpeedBackendError, match="per-rank accumulation"):
        _init(tmp_path, FakeDeepSpeed(), gradient_accumulation_steps_per_rank=160)


def test_initialize_refuses_an_optimizer_in_the_file(tmp_path):
    bad = tmp_path / "with_optimizer.deepspeed.json"
    bad.write_text(json.dumps({**dsc.default_config(), "optimizer": {"type": "AdamW"}}))
    with pytest.raises(dsc.DeepSpeedConfigError, match="optimizer"):
        _init(tmp_path, FakeDeepSpeed(), deepspeed_config=str(bad))


# ------------------------------------------------------------ loop arithmetic ---- #

def _batches(seed, n, batch=3, block=8, vocab=17):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, vocab, (n, batch, block + 1), generator=g)
    return [(row[:, :-1].contiguous(), row[:, 1:].contiguous()) for row in data]


def _legacy_run(steps, accum, grad_clip):
    """gpt2/train.py's non-DeepSpeed lines, single process, fp32."""
    model = _tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-2, (0.9, 0.95), "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    stream = iter(_batches(7, steps * accum + 1))
    X, Y = next(stream)
    for _ in range(steps):
        for _ in range(accum):
            _, loss = model(X, Y)
            loss = loss / accum
            X, Y = next(stream)
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return model


def _deepspeed_run(steps, accum, grad_clip):
    model = _tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-2, (0.9, 0.95), "cpu")
    engine = FakeEngine(model, optimizer, {"gradient_accumulation_steps": accum})
    stream = iter(_batches(7, steps * accum + 1))
    X, Y = next(stream)
    for _ in range(steps):
        X, Y, loss = backend.micro_steps(engine, X, Y, gradient_accumulation_steps=accum, grad_clip=grad_clip,
                                         get_batch=lambda split: next(stream), ctx=nullcontext())
    assert all(arg is False for arg in engine.backward_scale_args)
    return model


@pytest.mark.parametrize("accum", [1, 4])
@pytest.mark.parametrize("grad_clip", [0.0, 1.0, 1e-3])
def test_micro_steps_match_the_legacy_loop_bit_for_bit(accum, grad_clip):
    legacy = _legacy_run(steps=3, accum=accum, grad_clip=grad_clip)
    ds = _deepspeed_run(steps=3, accum=accum, grad_clip=grad_clip)
    for (name, a), (_, b) in zip(legacy.named_parameters(), ds.named_parameters()):
        assert torch.equal(a, b), name
    # The runs trained: parameters moved from their initial values.
    init = _tiny_model()
    assert any(not torch.equal(a, b) for a, b in zip(init.parameters(), ds.parameters()))


def test_clipping_happens_once_per_optimizer_step_at_the_boundary():
    model = _tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-2, (0.9, 0.95), "cpu")
    engine = FakeEngine(model, optimizer, {"gradient_accumulation_steps": 4})
    stream = iter(_batches(3, 9))
    X, Y = next(stream)
    calls = []

    def spy(params, value):
        calls.append((engine.micro_steps, value))
        return torch.nn.utils.clip_grad_norm_(params, value)

    backend.micro_steps(engine, X, Y, gradient_accumulation_steps=4, grad_clip=0.5,
                        get_batch=lambda split: next(stream), ctx=nullcontext(), clip_grad_norm=spy)
    assert calls == [(3, 0.5)]
