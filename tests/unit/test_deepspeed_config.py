"""The DeepSpeed config contract (order §5, §5.3) without DeepSpeed installed."""

import json

import pytest

from molcrawl.models import _deepspeed_config as dsc
from molcrawl.models._batch_policy import from_legacy_bert, from_legacy_gpt2, resolve_global_fixed


def _batch(world=4):
    return resolve_global_fixed(from_legacy_gpt2(8, 320), world)


def test_path_sits_beside_the_model_config():
    assert dsc.config_path_for("molcrawl/tasks/pretrain/configs/rna/bert_small.py") == \
        "molcrawl/tasks/pretrain/configs/rna/bert_small.deepspeed.json"
    with pytest.raises(dsc.DeepSpeedConfigError):
        dsc.config_path_for("x.json")


def test_default_config_passes_validation():
    facts = dsc.validate(dsc.default_config())
    assert facts == {"zero_stage": 0, "offload": {"optimizer": None, "param": None}}


@pytest.mark.parametrize("key", ["optimizer", "scheduler"])
def test_optimizer_and_scheduler_are_refused(key):
    cfg = {**dsc.default_config(), key: {"type": "AdamW"}}
    with pytest.raises(dsc.DeepSpeedConfigError, match=key):
        dsc.validate(cfg)


@pytest.mark.parametrize("key", ["bf16", "fp16", "amp", "torch_autocast"])
def test_precision_keys_in_the_file_are_refused(key):
    cfg = {**dsc.default_config(), key: {"enabled": True}}
    with pytest.raises(dsc.DeepSpeedConfigError, match="precision"):
        dsc.validate(cfg)


@pytest.mark.parametrize("key", dsc.BATCH_KEYS)
def test_batch_values_must_be_auto(key):
    cfg = {**dsc.default_config(), key: 16}
    with pytest.raises(dsc.DeepSpeedConfigError, match="auto"):
        dsc.validate(cfg)


def test_zero_stage_and_offload_need_explicit_permission():
    staged = {**dsc.default_config(), "zero_optimization": {"stage": 2}}
    with pytest.raises(dsc.DeepSpeedConfigError, match="ZeRO stage 2"):
        dsc.validate(staged)
    assert dsc.validate(staged, allow_zero_stage=True)["zero_stage"] == 2
    off = {**dsc.default_config(), "zero_optimization": {"stage": 0, "offload_optimizer": {"device": "cpu"}}}
    with pytest.raises(dsc.DeepSpeedConfigError, match="offload"):
        dsc.validate(off)
    none_device = {**dsc.default_config(), "zero_optimization": {"stage": 0, "offload_optimizer": {"device": "none"}}}
    assert dsc.validate(none_device)["offload"]["optimizer"] is None


def test_custom_framework_fills_numbers_from_the_batch_policy():
    cfg, record = dsc.resolve(dsc.default_config(), framework="custom", batch=_batch(4),
                              precision="bf16", gradient_clipping=1.0)
    assert (cfg["train_micro_batch_size_per_gpu"], cfg["gradient_accumulation_steps"], cfg["train_batch_size"]) == (8, 80, 2560)
    assert cfg["gradient_clipping"] == 1.0
    assert record["expected"]["train_batch_size"] == 2560
    # The same file resolves for 8 nodes without edits.
    cfg8, _ = dsc.resolve(dsc.default_config(), framework="custom", batch=_batch(32),
                          precision="bf16", gradient_clipping=1.0)
    assert (cfg8["gradient_accumulation_steps"], cfg8["train_batch_size"]) == (10, 2560)


def test_hf_framework_leaves_batch_auto_for_transformers():
    batch = resolve_global_fixed(from_legacy_bert(8, 80, 2560), 4)
    cfg, record = dsc.resolve(dsc.default_config(), framework="hf", batch=batch,
                              precision="fp32", gradient_clipping=1.0)
    for key in dsc.BATCH_KEYS + ("gradient_clipping",):
        assert cfg[key] == "auto"
    assert record["expected"]["gradient_accumulation_steps"] == 80


@pytest.mark.parametrize(
    "precision,autocast",
    [("fp32", {"enabled": False}),
     ("bf16", {"enabled": True, "dtype": "bfloat16"}),
     ("fp16", {"enabled": True, "dtype": "float16"})],
)
def test_precision_maps_to_torch_autocast_never_to_parameter_casting(precision, autocast):
    cfg, record = dsc.resolve(dsc.default_config(), framework="custom", batch=_batch(),
                              precision=precision, gradient_clipping=1.0)
    assert cfg["bf16"] == {"enabled": False}
    assert cfg["fp16"] == {"enabled": False}
    assert cfg["torch_autocast"] == autocast
    assert record["parameter_dtype"] == "float32"


def test_gradient_clipping_mismatch_is_refused():
    cfg = {**dsc.default_config(), "gradient_clipping": 0.5}
    with pytest.raises(dsc.DeepSpeedConfigError, match="gradient_clipping"):
        dsc.resolve(cfg, framework="custom", batch=_batch(), precision="fp32", gradient_clipping=1.0)
    same = {**dsc.default_config(), "gradient_clipping": 1.0}
    dsc.resolve(same, framework="custom", batch=_batch(), precision="fp32", gradient_clipping=1.0)


def test_resolve_does_not_mutate_the_loaded_file():
    original = dsc.default_config()
    snapshot = json.dumps(original, sort_keys=True)
    dsc.resolve(original, framework="custom", batch=_batch(), precision="bf16", gradient_clipping=1.0)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_load_records_path_hash_and_content(tmp_path):
    path = tmp_path / "gpt2_small.deepspeed.json"
    path.write_text(json.dumps(dsc.default_config()))
    data, ident = dsc.load(str(path))
    assert data["zero_optimization"]["stage"] == 0
    assert ident["path"] == str(path) and len(ident["sha256"]) == 64
    assert json.loads(ident["content"]) == data
    bad = tmp_path / "bad.deepspeed.json"
    bad.write_text("{")
    with pytest.raises(dsc.DeepSpeedConfigError, match="JSON"):
        dsc.load(str(bad))


def _engine_values(**over):
    values = {
        "train_micro_batch_size_per_gpu": 8, "gradient_accumulation_steps": 80, "train_batch_size": 2560,
        "gradient_clipping": 1.0, "bfloat16_enabled": False, "fp16_enabled": False,
        "torch_autocast_enabled": True, "torch_autocast_dtype": "torch.bfloat16",
    }
    values.update(over)
    return values


def test_check_effective_passes_when_the_engine_agrees():
    _, record = dsc.resolve(dsc.default_config(), framework="custom", batch=_batch(),
                            precision="bf16", gradient_clipping=1.0)
    assert dsc.check_effective(_engine_values(), record)["ok"] is True


@pytest.mark.parametrize(
    "override,fragment",
    [({"train_batch_size": 640}, "train_batch_size"),
     ({"bfloat16_enabled": True}, "bfloat16_enabled"),
     ({"torch_autocast_dtype": "torch.float16"}, "torch_autocast_dtype"),
     ({"gradient_accumulation_steps": 20}, "gradient_accumulation_steps")],
)
def test_check_effective_refuses_a_silent_difference(override, fragment):
    """The order's case: shown as 2,560 while the engine runs another batch."""
    _, record = dsc.resolve(dsc.default_config(), framework="custom", batch=_batch(),
                            precision="bf16", gradient_clipping=1.0)
    with pytest.raises(dsc.DeepSpeedConfigError, match=fragment):
        dsc.check_effective(_engine_values(**override), record)
