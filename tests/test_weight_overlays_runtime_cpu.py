"""No-GPU reference overlays: synthetic-overlay@cpu and source e448f070.

These synthetic operators and tiny serving fixtures do not qualify checkpoints.
"""


import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.experimental.weight_space.overlays import (
    InMemoryOverlayBank,
    QuantizationSpec,
    TensorTarget,
)
from mlx2.runtime.models.switch_layers import SwitchLinear
from mlx2.runtime.weight_overlays import WeightOverlayBinding, WeightOverlayManager
from mlx2.serving import request_apc_scope

CHECKPOINT = "synthetic-overlay@cpu"
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"


class Tiny(nn.Module):
    def __init__(self, expert=False, quantized=False):
        super().__init__()
        self.linear = SwitchLinear(32, 8, 3, bias=True) if expert else nn.Linear(32, 8)
        if quantized:
            self.linear = self.linear.to_quantized(group_size=32, bits=4)

    def __call__(self, x, *args):
        return self.linear(x, *args)


def manager_for(model, *, expert=None, budget=1024 * 1024):
    target = TensorTarget(0, expert, "linear")
    manager = WeightOverlayManager(model, {target: WeightOverlayBinding(model, "linear")},
        checkpoint=CHECKPOINT, max_materialized_bytes=budget)
    manager.bank = InMemoryOverlayBank(checkpoint=CHECKPOINT, source_revision=SOURCE,
        budget_bytes=1024 * 1024, source_validator=manager.source_validator)
    return manager, target


def create(manager, target, spec, name="trial"):
    source = manager.source(target)
    manager.bank.create(name, target, source["weights"], source_format=source["source_format"],
        source_tensor_fingerprint=source["source_tensor_fingerprint"], spec=spec)
    return manager.select(name)


@pytest.mark.parametrize("bits", range(1, 16))
def test_dense_all_bit_widths_execute_replacement_and_preserve_base(bits):
    model = Tiny()
    original = model.linear
    weights = np.array(original.weight)
    x = mx.arange(64, dtype=mx.float32).reshape(2, 32) / 32
    baseline = np.array(model(x))
    manager, target = manager_for(model)
    view = create(manager, target, QuantizationSpec(bits, group_size=17))
    manager.bind_uid(7, view)
    manager.bind_rows([7])
    expected = np.array(x) @ view.targets[target].tensor.dequantize().T + np.array(original.bias)
    np.testing.assert_allclose(np.array(model(x)), expected, atol=2e-6)
    assert manager.receipt(view, 7)["observed_used"] is True
    manager.clear_rows()
    np.testing.assert_array_equal(np.array(model(x)), baseline)
    np.testing.assert_array_equal(np.array(original.weight), weights)
    assert manager.status()["dequantized_bytes"] == weights.nbytes
    manager.release(view, 7)
    manager.restore()
    assert model.linear is original


@pytest.mark.parametrize("quantized", [False, True])
def test_expert_replacement_only_selected_rows_and_real_engagement(quantized):
    model = Tiny(expert=True, quantized=quantized)
    original = model.linear
    x = mx.arange(128, dtype=mx.float32).reshape(4, 1, 32) / 100
    ids = mx.array([0, 1, 2, 1], dtype=mx.uint32)
    baseline = np.array(model(x, ids))
    manager, target = manager_for(model, expert=1)
    view = create(manager, target, QuantizationSpec(4, group_size=16, format="fp4_e2m1_f32_scale"))
    manager.bind_uid(4, view)
    manager.bind_rows([4])
    all_other = mx.array([0, 0, 2, 2], dtype=mx.uint32)
    mx.eval(model(x, all_other))
    assert manager.receipt(view, 4)["observed_used"] is False
    result = np.array(model(x, ids))
    replacement = np.array(x) @ view.targets[target].tensor.dequantize().T + np.array(original.bias[1])
    expected = baseline.copy()
    expected[[1, 3]] = replacement[[1, 3]]
    np.testing.assert_allclose(result, expected, atol=2e-6)
    assert manager.receipt(view, 4)["applications"] == 1
    np.testing.assert_array_equal(result[[0, 2]], baseline[[0, 2]])
    manager.clear_rows()
    manager.release(view, 4)
    manager.restore()


def test_quantized_dense_source_format_and_materialization_budget():
    model = Tiny(quantized=True)
    manager, target = manager_for(model)
    source = manager.source(target)
    assert source["source_format"] == "mlx_affine_q4_g32"
    manager.bank.create("q1", target, source["weights"], source_format=source["source_format"],
                        spec=QuantizationSpec(1, group_size=32))
    manager.max_materialized_bytes = 1
    with pytest.raises(ValueError, match="budget"):
        manager.select("q1")
    assert manager.bank.snapshot()["active_requests"] == 0
    assert manager.status()["counts"]["rejected"] == 1
    manager.max_materialized_bytes = 8192
    view = manager.select("q1")
    manager.bind_uid(1, view)
    manager.bind_rows([1])
    output = np.array(model(mx.ones((1, 32))))
    assert np.isfinite(output).all()
    manager.clear_rows()
    manager.release(view, 1)


def test_frozen_views_revision_validation_and_width_refusal():
    model = Tiny()
    original = model.linear
    manager, target = manager_for(model)
    view = create(manager, target, QuantizationSpec(3))
    assert request_apc_scope({"_mlx2_weight_overlay_fingerprint": view.fingerprint}) != request_apc_scope({})
    with pytest.raises(ValueError, match="pinned"):
        manager.bank.delete("trial")
    manager.bind_uid(9, view)
    with pytest.raises(RuntimeError, match="width one"):
        manager.bind_rows([9, 10])
    manager.release(view, 9)
    original.weight = original.weight + 1
    with pytest.raises(ValueError, match="revision changed"):
        manager.select("trial")
    assert manager.bank.snapshot()["active_requests"] == 0


@pytest.mark.parametrize("key", ["weight_overlay", "layer_excision", "expert_mask"])
def test_http_generation_and_responses_preserve_registered_selection(key):
    from mlx2.openai_compat import responses_to_chat_request
    from mlx2.server import validate_request

    body = {"messages": [{"role": "user", "content": "hello"}], key: "candidate_1"}
    assert validate_request(body, chat=True)[key] == "candidate_1"
    translated, _ = responses_to_chat_request({"input": "hello", key: "candidate_1"})
    assert validate_request(translated, chat=True)[key] == "candidate_1"
    for value in ("/tmp/weights", 3, {}, "", "é"):
        with pytest.raises(ValueError, match="registered identifier"):
            validate_request({**body, key: value}, chat=True)


@pytest.mark.parametrize("experts", [0, 3])
def test_real_standard_qwen_adapter_explicit_dense_and_moe_mapping(experts):
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    kind = "qwen3_moe" if experts else "qwen3"
    model = Model(ModelArgs(model_type=kind, hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=32,
        num_key_value_heads=2, max_position_embeddings=128, rope_theta=10000,
        tie_word_embeddings=False, num_experts=experts, num_experts_per_tok=2 if experts else 0,
        moe_intermediate_size=64 if experts else 0))
    adapter = object.__new__(StandardDecoderAdapter)
    adapter.config, adapter.model = {"model_type": kind}, model
    bindings = adapter.weight_overlay_targets()
    target = TensorTarget(0, 1, "mlp.switch_mlp.down_proj") if experts else TensorTarget(0, None, "mlp.down_proj")
    assert target in bindings
    manager = WeightOverlayManager(model, bindings, checkpoint=CHECKPOINT)
    manager.bank = InMemoryOverlayBank(checkpoint=CHECKPOINT, source_revision=SOURCE,
        budget_bytes=1024 * 1024, source_validator=manager.source_validator)
    view = create(manager, target, QuantizationSpec(2))
    manager.bind_uid(3, view)
    manager.bind_rows([3])
    output = np.array(model(mx.array([[1, 2, 3]], dtype=mx.int32)))
    assert output.shape == (1, 3, 32)
    assert np.isfinite(output).all()
    manager.clear_rows()
    manager.release(view, 3)
    manager.restore()
    adapter.config = {"model_type": "llama"}
    assert adapter.weight_overlay_targets() == {}


def test_serving_freezes_overlay_cold_namespace_and_releases_pin(tmp_path, monkeypatch):
    from test_multi_lora import collect, make_engine, request

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": SOURCE})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    engine = make_engine(tmp_path, max_lanes=1, max_loras=0)
    try:
        target = TensorTarget(3, None, "q_proj")
        parent = engine.adapter.model.model.layers[3].self_attn
        engine.adapter.weight_overlay_targets = lambda: {target: WeightOverlayBinding(parent, "q_proj")}
        manager = engine.configure_weight_overlays()
        with pytest.raises(RuntimeError, match="model parking"):
            engine.park_model()
        with pytest.raises(RuntimeError, match="exported for restart"):
            engine.export_suspension("experimental")
        manager.bank = InMemoryOverlayBank(checkpoint=manager.checkpoint, source_revision=SOURCE,
            budget_bytes=1024 * 1024, source_validator=manager.source_validator)
        source = manager.source(target)
        manager.bank.create("q2", target, source["weights"], source_format=source["source_format"],
                            spec=QuantizationSpec(2))
        prompt = request(list(range(1, 20)), max_tokens=3)
        base, _ = collect(engine.submit(prompt))
        _, receipt = collect(engine.submit({**prompt, "weight_overlay": "q2"}))
        assert receipt["weight_overlay"]["observed_used"] is True
        assert receipt["cached_tokens"] == 0
        assert receipt["qualification"] == "unqualified"
        replay_job = engine.submit({**prompt, "weight_overlay": "q2"})
        _, replay = collect(replay_job)
        assert replay["cached_tokens"] == 0
        assert replay_job.request["skip_writing_prefix_cache"] is True
        assert manager.bank.snapshot()["active_requests"] == 0
        again, _ = collect(engine.submit(prompt))
        assert again == base
        assert engine.status()["weight_overlays"]["counts"]["applied"] > 0
    finally:
        engine.close()


def test_select_releases_bank_pin_on_any_exception(monkeypatch):
    from mlx2.experimental.weight_space import overlays

    manager, target = manager_for(Tiny())
    source = manager.source(target)
    manager.bank.create("trial", target, source["weights"], source_format=source["source_format"],
                        source_tensor_fingerprint=source["source_tensor_fingerprint"],
                        spec=QuantizationSpec(4, group_size=32))

    def exhausted(self):
        raise MemoryError("host allocation failed")

    monkeypatch.setattr(overlays.QuantizedTensor, "dequantize", exhausted)
    with pytest.raises(MemoryError):
        manager.select("trial")
    monkeypatch.undo()
    assert manager.bank.snapshot()["active_requests"] == 0
    assert manager.counts["rejected"] == 1 and not manager.views
    manager.bank.delete("trial")
