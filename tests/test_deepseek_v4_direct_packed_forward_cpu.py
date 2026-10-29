"""Production forward/cache plumbing with CPU attention oracles, no GPU claim."""

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter, split_execution_policy
from mlx2.adapters.deepseek_v4_packed_memory import (
    DeepSeekV4DirectPackedCacheBudget,
    DeepSeekV4PackedCacheBudget,
)
from mlx2.runtime.models import deepseek_v4_packed_cache as cache_module
from mlx2.runtime.models import deepseek_v4_packed_visible_attention as packed_attention
from mlx2.runtime.models import deepseek_v4_visible_attention as visible
from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_direct_policy_requires_real_physical_fp8_pairing_before_weights():
    with pytest.raises(ValueError, match="fp8_mixed"):
        split_execution_policy({"deepseek_attention": "packed_visible_rows_fp32"})
    with pytest.raises(ValueError, match="fp8_mixed"):
        DeepSeekV4Model(None, None, attention_policy="packed_visible_rows_fp32")
    kernel, _ = split_execution_policy(
        {"deepseek_attention": "packed_visible_rows_fp32", "deepseek_kv": "fp8_mixed"}
    )
    assert kernel["deepseek_attention"] == "packed_visible_rows_fp32"
    assert split_execution_policy(None)[0]["deepseek_attention"] == "reference"


def test_direct_budget_removes_only_pool_unpack_after_selected_policy():
    config = {"num_hidden_layers": 3, "compress_ratios": (0, 4, 128)}
    old = DeepSeekV4PackedCacheBudget.from_config(config)
    direct = DeepSeekV4DirectPackedCacheBudget.from_config(config)
    for context in (0, 129, 4096, 262144):
        assert old.project_resident(context) == direct.project_resident(context)
        assert old.snapshot_bytes(context) == direct.snapshot_bytes(context)
        expected = (
            sum(old._pool_capacity(context, r) for r in old.compress_ratios if r)
            * 512
            * 4
        )
        assert (
            old.unpack_workspace_bytes(context) - direct.unpack_workspace_bytes(context)
            == expected
        )
        assert old.project(context) - direct.project(context) == expected
        assert (
            old.project_pool(context, resident_lanes=2, running_lanes=1)
            - direct.project_pool(context, resident_lanes=2, running_lanes=1)
            == expected
        )
        assert old.prefill_transient_bytes(
            context, 128
        ) == direct.prefill_transient_bytes(context, 128)
    assert direct.transient_gib_per_lane == old.transient_gib_per_lane == 1.76
    assert direct.as_dict()["pool_fp32_materialization"] is False
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.model = type("M", (), {"args": config})()
    adapter.kernel_policy = {
        "deepseek_kv": "fp8_mixed",
        "deepseek_attention": "packed_visible_rows_fp32",
    }
    assert isinstance(
        adapter.cache_budget(mtp=False), DeepSeekV4DirectPackedCacheBudget
    )
    adapter.kernel_policy["deepseek_attention"] = "visible_rows_fp32"
    assert type(adapter.cache_budget(mtp=False)) is DeepSeekV4PackedCacheBudget


def test_production_compressed_forward_bypasses_pool_getter_and_append_unpack(
    monkeypatch,
):
    from test_deepseek_v4_forward_cpu import SmallWeights, config

    c = replace(
        config((4, 128)),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        simulate_quantization=True,
        sliding_window=4,
    )
    weights = SmallWeights(c)

    def ordinary_cpu(q, local, pool, indices, sinks, **kwargs):
        out = visible.reference_visible_attention(
            np.asarray(q),
            np.asarray(local),
            None if pool is None else np.asarray(pool),
            None if indices is None else np.asarray(indices),
            np.asarray(sinks),
            **kwargs,
        )
        return mx.array(out)

    monkeypatch.setattr(visible, "visible_attention", ordinary_cpu)
    old = DeepSeekV4Model(
        c,
        weights,
        identity="cpu-packed-old",
        attention_policy="visible_rows_fp32",
        cache_policy="fp8_mixed",
    )
    old_cache = old.make_cache()
    tokens = mx.array(np.arange(138)[None] % c.vocab_size, dtype=mx.int32)
    chunks = ((0, 3), (3, 7), (7, 127), (127, 129), (129, 138))
    expected = []
    for start, end in chunks:
        out = old(tokens[:, start:end], old_cache)
        mx.eval(out)
        expected.append(np.asarray(out))
    original_unpack = cache_module.unpack_fp8_kv
    called = []

    def local_only_unpack(packed, *, xp):
        if packed.codes.ndim == 3:
            raise AssertionError("completed pool unpacked by production cache path")
        return original_unpack(packed, xp=xp)

    monkeypatch.setattr(cache_module, "unpack_fp8_kv", local_only_unpack)

    def no_pool_getter(self):
        raise AssertionError("production accessed FP32 pooled getter")

    monkeypatch.setattr(
        cache_module.DeepSeekV4PackedPoolCache,
        "pooled",
        property(no_pool_getter, cache_module.DeepSeekV4PackedPoolCache.pooled.fset),
    )

    def direct_cpu(q, local, packed, indices, sinks, **kwargs):
        called.append(
            (
                kwargs["ratio"],
                kwargs["offset"],
                0 if packed is None else packed.codes.shape[1],
            )
        )
        if packed is not None:
            assert packed.codes.dtype == mx.uint8
        from mlx2.runtime.models.deepseek_v4_packed_kv import PackedKV

        host = (
            None
            if packed is None
            else PackedKV(
                *(np.asarray(x) for x in (packed.codes, packed.scales, packed.rope))
            )
        )
        out = packed_attention.reference_packed_visible_attention(
            np.asarray(q),
            np.asarray(local),
            host,
            None if indices is None else np.asarray(indices),
            np.asarray(sinks),
            **kwargs,
        )
        return mx.array(out)

    monkeypatch.setattr(packed_attention, "packed_visible_attention", direct_cpu)
    direct = DeepSeekV4Model(
        c,
        weights,
        identity="cpu-packed-direct",
        attention_policy="packed_visible_rows_fp32",
        cache_policy="fp8_mixed",
    )
    state = direct.make_cache()
    for (start, end), answer in zip(chunks, expected):
        out = direct(tokens[:, start:end], state)
        mx.eval(out)
        np.testing.assert_array_equal(np.asarray(out), answer)
    assert {ratio for ratio, _, _ in called} == {4, 128}
    assert any(ratio == 128 and count == 1 for ratio, _, count in called)
    # State validation/snapshot/restore access raw planes, never the getter.
    snapshots = [item.snapshot() for item in state]
    for item, snapshot in zip(state, snapshots):
        item.restore(snapshot)
        item.validate_state()

    def equal(a, b):
        if a is None:
            return a is b
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return np.array_equal(
            np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)
        )

    assert all(equal(a.state, b.state) for a, b in zip(state, old_cache))
