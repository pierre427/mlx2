"""Native Source router, synthetic payload fixture, no DS4.1 checkpoint load.

Checkpoint tiny@tiny-hybrid, source e448f070. The fake payload store records
native original IDs; this does not qualify real Source file reads or kernels.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters.deepseek_v41_source_masks import source_expert_mask_binding
from mlx2.adapters.deepseek_v41_source_weights import SourceExpertStore
from mlx2.experimental.weight_space.experts import ExpertMaskPlan, LayerExpertMask
from mlx2.runtime.expert_masks import ExpertMaskManager
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.deepseek_v4 import route_experts
from mlx2.runtime.models.deepseek_v41 import Block
from mlx2.runtime.weight_overlays import (
    bind_weight_overlay_rows,
    clear_weight_overlay_rows,
)

CHECKPOINT = "tiny@tiny-hybrid"
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"


class SyntheticSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(n_routed_experts=4, num_experts_per_tok=2,
            norm_topk_prob=True, routed_scaling_factor=1.5, hidden_size=8, vocab_size=64)
        self.embed = nn.Embedding(64, 8)
        self.gate = nn.Linear(8, 4, bias=False)
        self.head = nn.Linear(8, 64, bias=False)
        self.expert_values = mx.arange(32, dtype=mx.float32).reshape(4, 8) / 32
        self.pointwise_policy = "reference"
        block = object.__new__(Block)
        block.model, block.c, block.layer, block.prefix = self, self.args, 0, "blk.0."
        block.correction = mx.array([0.2, 0.0, -0.1, 0.1])
        self.layers = [block]
        self.received = []
        # No files, readers, leases or MLX payloads are initialized. Merely
        # prove the binding retains this native store identity unchanged.
        self.native_store = object.__new__(SourceExpertStore)
        self.weights = SimpleNamespace(expert_stream=self.native_store)
        mx.eval(self.parameters())

    def linear(self, x, _name, *, rounded):
        assert rounded is False
        return self.gate(x)

    def make_cache(self):
        return [KVCache()]

    def __call__(self, tokens, cache=None):
        hidden = self.embed(tokens)
        ids, scores = self.layers[0].router(hidden)
        # This is the unchanged payload-consumer seam: only the router owns
        # masking; dispatch still receives original checkpoint expert IDs.
        self.received.append(np.array(ids))
        routed = mx.sum(self.expert_values[ids] * scores[..., None], axis=-2)
        if cache is not None:
            zeros = mx.zeros((tokens.shape[0], 1, tokens.shape[1], 8))
            cache[0].update_and_fetch(zeros, zeros)
        return self.head(hidden + routed)


def binding(model, policy=None):
    adapter = SimpleNamespace(model=model, weights=model.weights, kernel_policy=policy or {},
        identity={"fingerprint": "a" * 64, "weight_sha256": "a" * 64},
        numeric_profile="source_effective_fixture")
    return source_expert_mask_binding(adapter)


def plan(manager, enabled=(1, 3), **changes):
    bound = manager.binding
    values = {"checkpoint": manager.checkpoint, "source_revision": SOURCE,
        "source_fingerprint": bound.source_fingerprint,
        "layers": (LayerExpertMask(0, 4, 2, enabled, True, 1.5),),
        "quantization_profile": bound.quantization_profile, "execution_profile": bound.execution_profile,
        "calibration_digest": "b" * 64, "tie_policy": "native_topk_after_mask"}
    return ExpertMaskPlan(**{**values, **changes})


def test_source_native_all_enabled_bypass_and_masked_original_ids():
    model = SyntheticSource()
    original_store = model.native_store
    x = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 10
    original = tuple(np.array(value) for value in model.layers[0].router(x))
    manager = ExpertMaskManager(model, binding(model), checkpoint=CHECKPOINT)
    manager.register("all", plan(manager, tuple(range(4))))
    manager.register("pruned", plan(manager))
    full = manager.select("all")
    manager.bind_uid(1, full)
    rows = bind_weight_overlay_rows(model, [1])
    actual = tuple(np.array(value) for value in model.layers[0].router(x))
    for left, right in zip(actual, original):
        np.testing.assert_array_equal(left, right)
    assert manager.receipt(full, 1)["observed_used"] is False
    clear_weight_overlay_rows(rows)
    manager.release(full, 1)
    view = manager.select("pruned")
    manager.bind_uid(2, view)
    rows = bind_weight_overlay_rows(model, [2])
    ids, weights = model.layers[0].router(x)
    expected = route_experts(model.gate(x), top_k=2, scale=1.5,
        correction=mx.array([-float("inf"), 0.0, -float("inf"), 0.1]), normalize=True)
    np.testing.assert_array_equal(np.array(ids), np.array(expected[0]))
    np.testing.assert_array_equal(np.array(weights), np.array(expected[1]))
    assert set(np.array(ids).flatten()) == {1, 3}
    assert manager.receipt(view, 2)["observed_used"] is True
    assert model.weights.expert_stream is original_store
    with pytest.raises(ValueError, match="text-only"):
        model.layers[0].router(x, image_mask=np.array([True, False, False]))
    clear_weight_overlay_rows(rows)
    manager.release(view, 2)
    manager.restore()
    restored = model.layers[0].router(x)
    np.testing.assert_array_equal(np.array(restored[0]), original[0])


def test_source_mask_refuses_wrong_source_law_and_unsupported_composition():
    model = SyntheticSource()
    for policy in ({"deepseek_source_route_bias": "resident_v1"},
                   {"deepseek_physical_batch": "b2"},
                   {"deepseek_source_experts": {"residency": "dynamic_precision_v1"}}):
        with pytest.raises(ValueError, match="ordinary fixed-precision"):
            binding(model, policy)
    manager = ExpertMaskManager(model, binding(model), checkpoint=CHECKPOINT)
    for altered in ({"source_fingerprint": "c" * 64}, {"execution_profile": "wrong"},
                    {"tie_policy": "stable_lowest_original_expert_id"}):
        with pytest.raises(ValueError, match="exact native"):
            manager.register("wrong", plan(manager, **altered))
    manager.register("one", plan(manager))
    view = manager.select("one")
    manager.bind_uid(1, view)
    with pytest.raises(RuntimeError, match="width one"):
        manager.bind_rows([1, 2])
    manager.release(view, 1)


def test_serving_source_router_mask_freezes_uids_and_never_publishes_approximate(monkeypatch):
    from test_approximate_kv_serving import make_adapter
    from test_multi_lora import collect, request

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": SOURCE})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model = SyntheticSource()
    factory = make_adapter(model, operations=None)
    factory.expert_mask_binding = lambda _self: binding(model)
    engine = serving.ServingEngine("tiny", adapter_factory=factory,
        qualification_mode=True, mtp=False, max_lanes=1, max_loras=0)
    try:
        assert engine.ready.wait(30), engine.error
        manager = engine.configure_expert_masks()
        manager.register("pruned", plan(manager))
        body = request(list(range(1, 15)), max_tokens=3)
        base, _ = collect(engine.submit(body))
        model.received.clear()
        first, receipt = collect(engine.submit({**body, "expert_mask": "pruned"}))
        assert receipt["expert_mask"]["observed_used"] is True
        assert receipt["expert_mask"]["apc_publication"] == "suppressed_unqualified_approximate"
        assert receipt["cached_tokens"] == 0
        assert all(set(row.flatten()) <= {1, 3} for row in model.received)
        second, receipt2 = collect(engine.submit({**body, "expert_mask": "pruned"}))
        assert second == first and receipt2["cached_tokens"] == 0
        assert not manager.views
        again, _ = collect(engine.submit(body))
        assert again == base
    finally:
        engine.close()
