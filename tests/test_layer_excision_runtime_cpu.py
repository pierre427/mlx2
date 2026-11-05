"""Qwen3 synthetic-excision@cpu, source e448f070; no checkpoint qualification."""

from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.standard_decoder import StandardDecoderAdapter
from mlx2.experimental.weight_space.excision import BoundaryHealing, LayerExcisionPlan
from mlx2.runtime.layer_excision import LayerExcisionManager
from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.runtime.weight_overlays import (
    bind_weight_overlay_rows,
    clear_weight_overlay_rows,
)

CHECKPOINT = "synthetic-excision@cpu"
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"


def model_adapter():
    model = Model(ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=4,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=2, max_position_embeddings=128, rope_theta=10000,
        tie_word_embeddings=False))
    adapter = object.__new__(StandardDecoderAdapter)
    mx.eval(model.parameters())
    adapter.model, adapter.config = model, {"model_type": "qwen3"}
    return model, adapter


def plan(checkpoint=CHECKPOINT, *, heal=False):
    healing = (BoundaryHealing(2, np.ones((32, 1), np.float32) / 100,
        np.ones((1, 32), np.float32) / 50, "a" * 64, "b" * 64, "c" * 64) if heal else None)
    return LayerExcisionPlan(checkpoint, SOURCE, 4, (1,), 32, healing)


@pytest.mark.parametrize("heal", [False, True])
def test_actual_block_is_bypassed_zero_cache_advanced_and_base_reversible(monkeypatch, heal):
    model, adapter = model_adapter()
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    base = np.array(model(tokens))
    original_layers = tuple(model.layers)
    calls = Counter()
    layer_type = type(original_layers[0])
    original_call = layer_type.__call__

    def counted(self, *args, **kwargs):
        calls[id(self)] += 1
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(layer_type, "__call__", counted)
    manager = LayerExcisionManager(model, adapter.layer_excision_binding(), checkpoint=CHECKPOINT)
    selected = plan(heal=heal)
    manager.register("remove_one", selected)
    view = manager.select("remove_one")
    manager.bind_uid(1, view)
    context = bind_weight_overlay_rows(model, [1])
    cache = model.make_cache()
    output = np.array(model(tokens, cache=cache))
    assert calls[id(original_layers[1])] == 0
    assert all(item.offset == 4 for item in cache)
    assert np.count_nonzero(np.array(cache[1].state[0])) == 0
    # Direct edited-model oracle: same retained blocks and healing boundary.
    hidden = model.model.embed_tokens(tokens)
    from mlx2.runtime.models.base import create_attention_mask
    mask = create_attention_mask(hidden)
    for index, block in enumerate(original_layers):
        if index == 1:
            continue
        hidden = block(hidden, mask)
        if heal and index == 2:
            hidden = mx.array(selected.healing.apply_numpy(np.array(hidden)))
    expected = np.array(model.logits(model.model.norm(hidden)))
    np.testing.assert_allclose(output, expected, atol=1e-6)
    receipt = manager.receipt(view, 1)
    assert receipt["observed_used"] is True
    assert receipt["applications"]["skipped_blocks"] == 1
    if heal:
        assert receipt["applications"]["healing_applications"] == 1
    clear_weight_overlay_rows(context)
    np.testing.assert_array_equal(np.array(model(tokens)), base)
    manager.release(view, 1)
    manager.restore()
    assert all(a is b for a, b in zip(model.layers, original_layers))


def test_layer_binding_rejects_unsupported_cache_and_wrong_checkpoint():
    model, adapter = model_adapter()
    manager = LayerExcisionManager(model, adapter.layer_excision_binding(), checkpoint=CHECKPOINT)
    with pytest.raises(ValueError, match="checkpoint"):
        manager.register("wrong", plan("other@other"))
    manager.register("one", plan())
    view = manager.select("one")
    manager.bind_uid(1, view)
    with pytest.raises(RuntimeError, match="width one"):
        manager.bind_rows([1, 2])
    manager.bind_rows([1])
    from mlx2.runtime.models.cache import RotatingKVCache
    with pytest.raises(ValueError, match="full KV"):
        model.layers[1](mx.ones((1, 1, 32)), cache=RotatingKVCache(max_size=8))
    manager.clear_rows()
    manager.release(view, 1)


def test_actual_qwen_serving_excision_replays_private_without_publication(monkeypatch):
    from test_approximate_kv_serving import make_adapter
    from test_multi_lora import collect, request

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": SOURCE})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, adapter = model_adapter()
    factory = make_adapter(model, operations=None)
    factory.layer_excision_binding = lambda _self: adapter.layer_excision_binding()
    engine = serving.ServingEngine("tiny", adapter_factory=factory,
        qualification_mode=True, mtp=False, max_lanes=1, max_loras=0)
    try:
        assert engine.ready.wait(30), engine.error
        manager = engine.configure_layer_excision()
        manager.register("one", plan("tiny@tiny-hybrid", heal=True))
        prompt = request(list(range(1, 20)), max_tokens=4)
        base, _ = collect(engine.submit(prompt))
        selected = {**prompt, "layer_excision": "one"}
        cold, receipt = collect(engine.submit(selected))
        assert receipt["layer_excision"]["observed_used"] is True
        assert receipt["cached_tokens"] == 0
        assert receipt["qualification"] == "unqualified"
        warm, warm_receipt = collect(engine.submit(selected))
        assert warm == cold
        assert warm_receipt["cached_tokens"] == 0
        assert warm_receipt["layer_excision"]["apc_publication"] == "suppressed_unqualified_approximate"
        again, _ = collect(engine.submit(prompt))
        assert again == base
        assert not manager.views
    finally:
        engine.close()
