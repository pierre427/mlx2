"""Synthetic CPU operators/state contracts; no checkpoint/GPU execution."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import KVCache, load_prompt_cache, save_prompt_cache
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel, Glm53KDACache
from mlx2.runtime.models.glm53_gguf_indexer import pooled_selection


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class Weights:
    def __init__(self):
        self.rng = np.random.default_rng(75)
        self.arrays = {}

    def array(self, name):
        if name not in self.arrays:
            if "conv1d" in name:
                value = self.rng.normal(0, 0.2, (8192, 1, 4)).astype(np.float32)
            elif name.endswith("ssm_a"):
                value = self.rng.uniform(1, 8, 64).astype(np.float32)
            elif name.endswith("ssm_dt.bias"):
                value = self.rng.normal(-2, 0.2, 8192).astype(np.float32)
            elif name.endswith("ssm_norm.weight"):
                value = np.ones(128, dtype=np.float32)
            elif name.endswith("kpool_ape"):
                value = self.rng.normal(0, 0.1, (4, 128)).astype(np.float32)
            elif name.endswith("k_norm.weight"):
                value = np.ones(128, dtype=np.float32)
            elif name.endswith("k_norm.bias"):
                value = np.zeros(128, dtype=np.float32)
            else:
                raise AssertionError(name)
            self.arrays[name] = mx.array(value)
        return self.arrays[name]

    def linear(self, name):
        def call(x):
            if name not in self.arrays:
                if "indexer.attn_q_b" in name:
                    out = 4096
                elif "indexer.proj" in name:
                    out = 32
                elif "indexer" in name or name.endswith(
                    ("ssm_f_a.weight", "ssm_g_a.weight")
                ):
                    out = 128
                elif name.endswith("ssm_beta.weight"):
                    out = 64
                elif name.endswith("attn_output.weight"):
                    out = 4
                else:
                    out = 8192
                value = self.rng.normal(0, 0.05, (x.shape[-1], out)).astype(np.float32)
                self.arrays[name] = mx.array(value)
            return x @ self.arrays[name]

        return call


def test_kda_chunk_and_incremental_state_match():
    w = Weights()
    m = Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5},
        w,
        identity="synthetic",
        math_policy="reference",
    )
    x = mx.array(np.random.default_rng(8).normal(size=(1, 5, 4)).astype(np.float32))
    chunk = Glm53KDACache("synthetic")
    one = Glm53KDACache("synthetic")
    a = m._kda(x, chunk, "blk.0")
    b = mx.concatenate(
        [m._kda(x[:, i : i + 1], one, "blk.0") for i in range(5)], axis=1
    )
    mx.eval(a, b, chunk.state, one.state)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5, rtol=2e-5)
    for left, right in zip(chunk.state, one.state):
        np.testing.assert_allclose(
            np.array(left), np.array(right), atol=2e-6, rtol=2e-5
        )
    assert chunk.offset == one.offset == 5


def test_kda_checkpoint_roundtrip_and_wrong_abi_refused(tmp_path):
    c = Glm53KDACache("synthetic@source-policy")
    c.conv = mx.zeros((1, 3, 24576))
    c.recurrent = mx.zeros((1, 64, 128, 128))
    c.offset = 17
    path = tmp_path / "cache.safetensors"
    save_prompt_cache(str(path), [c])
    restored = load_prompt_cache(str(path))[0]
    assert type(restored) is Glm53KDACache
    assert restored.meta_state == c.meta_state
    assert restored.cache_operations_contract.rollback is None
    assert restored.size() == 17
    with pytest.raises(ValueError, match="cannot trim"):
        restored.trim(1)
    with pytest.raises(ValueError, match="geometry"):
        Glm53KDACache.from_state((mx.zeros((1, 1, 2)), c.recurrent), c.meta_state)
    with pytest.raises(ValueError, match="identity"):
        restored.restore_exact_decode_checkpoint(("other",))


def test_indexer_pool_boundary_and_per_query_causality():
    w = Weights()
    rng = np.random.default_rng(33)
    cache = KVCache()
    x = mx.array(rng.normal(size=(1, 2049, 4)).astype(np.float32))
    qr = mx.array(rng.normal(size=(1, 2049, 6)).astype(np.float32))
    ids = pooled_selection(x, qr, cache, w, "blk.3")
    a = np.array(ids)
    # Each query has its own tail and future complete pools remain hidden.
    for t in [0, 3, 100, 2047, 2048]:
        actual = a[0, t]
        actual = actual[actual >= 0]
        np.testing.assert_array_equal(np.sort(actual), np.arange(t + 1))
    more = mx.array(rng.normal(size=(1, 7, 4)).astype(np.float32))
    more_q = mx.array(rng.normal(size=(1, 7, 6)).astype(np.float32))
    selected = np.array(pooled_selection(more, more_q, cache, w, "blk.3"))
    for i in range(7):
        actual = selected[0, i]
        actual = actual[actual >= 0]
        assert np.all(actual <= 2049 + i)
        assert len(np.unique(actual)) == len(actual)
        assert len(actual) == 2048 + (2050 + i) % 4
    # Restore raw indexer state, then append: no pooled Python attribute survives.
    restored = KVCache.from_state(cache.state, cache.meta_state)
    next_x = more[:, :1]
    next_q = more_q[:, :1]
    left = pooled_selection(next_x, next_q, cache, w, "blk.3")
    right = pooled_selection(next_x, next_q, restored, w, "blk.3")
    np.testing.assert_array_equal(np.array(left), np.array(right))


def test_topology_and_memory_are_glm_specific_and_revision_bound():
    from mlx2.adapters.glm53_gguf import CACHE_LAYOUT, Glm53GGUFAdapter
    from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget
    from mlx2.runtime.attention_topology import apcv2_cache_layout_fingerprint
    from mlx2.runtime.cow_cache import _build_layer_segment_manifest
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    config = {
        "layer_types": [
            "linear_attention" if (i + 1) % 4 else "deepseek_sparse_attention"
            for i in range(45)
        ]
    }
    a = object.__new__(Glm53GGUFAdapter)
    a.config = config
    a.identity = {"fingerprint": "synthetic-revision"}
    a.layout = CACHE_LAYOUT
    assert apcv2_cache_layout_fingerprint(a)[0] == CACHE_LAYOUT
    budget = Glm53CacheBudget.from_config(config)
    ast = Glm53CacheBudget.from_config(config, tree_rows=16)
    assert ast.project(4096) > 8 * budget.project(4096)
    assert budget.snapshot_bytes(257) > budget.snapshot_bytes(256)
    c = Glm53KDACache("synthetic")
    c.offset = 17
    c.conv = mx.zeros((1, 3, 24576))
    c.recurrent = mx.zeros((1, 64, 128, 128))
    m = Glm53MLACache()
    for kv, dim in zip(m.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 17, dim)), mx.zeros((1, 1, 17, 0)))
    manifest = _build_layer_segment_manifest([c, m], None, covered_tokens=17)
    assert any(s.key.kind.value == "gdn_recurrent" for s in manifest.segments)
    with pytest.raises(ValueError):
        Glm53CacheBudget.from_config(config, mtp=True)


def test_composite_cache_keeps_abi_when_extracting_a_request(tmp_path):
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    c = Glm53MLACache()
    for kv, dim in zip(c.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 9, dim)), mx.zeros((1, 1, 9, 0)))
    extracted = c.extract(0)
    assert type(extracted) is Glm53MLACache
    assert extracted.offset == 9
    path = tmp_path / "hybrid.safetensors"
    save_prompt_cache(str(path), [extracted])
    loaded = load_prompt_cache(str(path))[0]
    assert type(loaded) is Glm53MLACache and loaded.offset == 9


def test_codec_copy_preserves_composite_class_and_operations_contract():
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache
    from mlx2.runtime.recurrent_state_codec import (
        RecurrentStateCodecPolicy,
        encode_prompt_cache,
    )

    cache = Glm53MLACache()
    encoded, _ = encode_prompt_cache([cache], RecurrentStateCodecPolicy(enabled=True))
    assert type(encoded[0]) is Glm53MLACache
    assert encoded[0] is not cache
    assert encoded[0].cache_operations_contract == cache.cache_operations_contract


def test_mla_checkpoint_refuses_wrong_shape_and_inconsistent_frontiers():
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    cache = Glm53MLACache()
    for kv, width in zip(cache.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 3, width)), mx.zeros((1, 1, 3, 0)))
    state = cache.state
    with pytest.raises(ValueError, match="geometry"):
        Glm53MLACache.from_state(
            [(mx.zeros((1, 1, 3, 511)), state[0][1]), state[1]], cache.meta_state
        )
    cache.caches[1].offset = 2
    with pytest.raises(ValueError, match="frontiers"):
        Glm53MLACache.from_state(state, cache.meta_state)


def test_preprojected_kda_and_separate_output_preserve_reference_state():
    weights = Weights()
    model = Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5},
        weights,
        identity="synthetic",
        math_policy="reference",
    )
    x = mx.array(np.random.default_rng(84).normal(size=(1, 3, 4)).astype(np.float32))
    ordinary = Glm53KDACache("synthetic")
    prepared = Glm53KDACache("synthetic")
    expected = model._kda(x, ordinary, "blk.0")
    project = lambda value, suffix: weights.linear("blk.0." + suffix)(value)
    projected = {
        "qkv": mx.concatenate(
            [project(x, "attn_" + name + ".weight") for name in ("q", "k", "v")],
            axis=-1,
        ),
        "a": project(project(x, "ssm_f_a.weight"), "ssm_f_b.weight"),
        "beta": project(x, "ssm_beta.weight"),
        "gate": project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight"),
    }
    core = model._kda(
        x, prepared, "blk.0", _projected=projected, _return_pre_output=True
    )
    actual = project(core, "attn_output.weight")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    for a, b in zip(ordinary.state, prepared.state):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    assert ordinary.offset == prepared.offset == 3
