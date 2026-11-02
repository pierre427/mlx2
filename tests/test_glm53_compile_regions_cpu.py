"""glm53_compile_regions: traced layer bodies equal eager, counters replay, gates decline."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import glm53_compile_regions as regions
from mlx2.runtime.models import glm53_gguf


class _Model:
    """The four layer hooks CompiledLayers drives, with a route ledger."""

    def __init__(self):
        rng = np.random.default_rng(7)
        self.w = mx.array(rng.normal(size=(8, 8)).astype(np.float32))
        self.calls = {"kda": 0, "mlp": 0, "mla": 0}
        self.weights = SimpleNamespace(scheduled_dense_calls={}, scheduled_grouped_calls={},
                                       scheduled_expert_calls={})
        self.layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
        self.math_policy, self.hc_mega, self.hc_fused = "ordinary", False, False
        self.args = SimpleNamespace(rms_norm_eps=1e-6)
        self.compile_regions = "v1"

    def _hc_in(self, h, half, norm):
        x = h.mean(-2)
        post = mx.full((*h.shape[:-1],), 0.5)
        comb = mx.broadcast_to(mx.eye(4) * 0.25, (*h.shape[:-2], 4, 4))
        return x, post, comb

    def _kda(self, x, cache, prefix):
        self.calls["kda"] += 1
        self.weights.scheduled_dense_calls["kblock_rows_dense"] = (
            self.weights.scheduled_dense_calls.get("kblock_rows_dense", 0) + 1)
        cache.conv = cache.conv * 0.5 + x[:, :, :1]
        cache.recurrent = cache.recurrent + (x * x).sum()
        cache.offset += 1
        return x @ self.w + cache.conv.sum()

    def _mlp(self, x, prefix, index):
        self.calls["mlp"] += 1
        return mx.maximum(x @ self.w.T, 0)

    def _mla(self, x, cache, prefix, *, frontier):
        self.calls["mla"] += 1
        return x * 2


def _eager(model, caches, h):
    for index, (layer, c) in enumerate(zip(model.layers, caches)):
        prefix = f"blk.{index}"
        x, post, comb = model._hc_in(h, prefix + ".hc_attn", "")
        branch = model._kda(x, c, prefix) if layer.is_linear else model._mla(x, c, prefix, frontier=0)
        h = glm53_gguf.hc_expand(branch, h, post, comb, fused=False)
        x, post, comb = model._hc_in(h, prefix + ".hc_ffn", "")
        h = glm53_gguf.hc_expand(model._mlp(x, prefix, index), h, post, comb, fused=False)
    return h


def _caches():
    return [SimpleNamespace(conv=mx.ones((1, 3, 1)), recurrent=mx.zeros((1, 2)), offset=0),
            SimpleNamespace()]


def test_traced_layers_equal_eager_and_replay_counters():
    eager_model, compiled_model = _Model(), _Model()
    compiled_model.w = eager_model.w
    eager_caches, compiled_caches = _caches(), _caches()
    layers = regions.CompiledLayers(compiled_model)
    rng = np.random.default_rng(3)
    for _ in range(4):
        h = mx.array(rng.normal(size=(1, 1, 4, 8)).astype(np.float32))
        expected = _eager(eager_model, eager_caches, h)
        actual = h
        for index, (layer, c) in enumerate(zip(compiled_model.layers, compiled_caches)):
            actual = layers.layer(index, layer, c, actual, 0)
        assert np.array_equal(np.array(expected), np.array(actual))
        assert np.array_equal(np.array(eager_caches[0].conv), np.array(compiled_caches[0].conv))
        assert np.array_equal(np.array(eager_caches[0].recurrent), np.array(compiled_caches[0].recurrent))
        assert eager_caches[0].offset == compiled_caches[0].offset
        assert eager_model.calls == compiled_model.calls
        assert (eager_model.weights.scheduled_dense_calls
                == compiled_model.weights.scheduled_dense_calls)
    # Two regions (the KDA layer, the MLA FFN half), each traced once.
    assert len(layers.regions) == 2 and all(r.delta is not None for r in layers.regions.values())


def test_policy_change_keys_a_fresh_trace():
    model = _Model()
    layers = regions.CompiledLayers(model)
    caches = _caches()
    h = mx.zeros((1, 1, 4, 8))
    layers.layer(0, model.layers[0], caches[0], h, 0)
    first = layers.regions[("kda", 0)]
    model.kda_small = "fused_v1"
    layers.layer(0, model.layers[0], caches[0], h, 0)
    assert layers.regions[("kda", 0)] is not first


def test_active_declines_off_policy_prompt_width_and_cpu():
    model = _Model()
    caches = _caches()
    model.compile_regions = "off"
    assert regions.active(model, mx.zeros((1, 1)), caches) is None
    model.compile_regions = "v1"
    assert regions.active(model, mx.zeros((1, 2)), caches) is None
    with mx.stream(mx.cpu):
        assert regions.active(model, mx.zeros((1, 1)), caches) is None
    with glm53_gguf.sparse_attention_phase("prompt"):
        assert regions.active(model, mx.zeros((1, 1)), caches) is None


def test_first_admitted_step_per_policy_runs_a_complete_eager_warmup():
    if not mx.metal.is_available():
        return
    model = _Model()
    with mx.stream(mx.gpu):
        if mx.default_device() != mx.gpu:
            return
        caches = _caches()
        h = mx.zeros((1, 1, 4, 8))
        warm = regions.active(model, mx.zeros((1, 1)), caches)
        assert isinstance(warm, regions._Warmup)
        # An interrupted warm-up (only layer 0 built) leaves the policy cold.
        warm.layer(0, model.layers[0], caches[0], h, 0)
        assert isinstance(regions.active(model, mx.zeros((1, 1)), caches), regions._Warmup)
        eager = _eager(_Model(), _caches(), h)
        warm = regions.active(model, mx.zeros((1, 1)), caches)
        out = h
        for index, (layer, c) in enumerate(zip(model.layers, _caches())):
            out = warm.layer(index, layer, c, out, 0)
        assert np.array_equal(np.array(out), np.array(eager))
        assert isinstance(regions.active(model, mx.zeros((1, 1)), caches), regions.CompiledLayers)
        assert regions.active(model, mx.zeros((1, 1)), caches, plain=False) is None
        model.router = "fused_v1"
        assert isinstance(regions.active(model, mx.zeros((1, 1)), caches), regions._Warmup)
        regions.release(model)
        assert "_compiled_layers" not in model.__dict__
