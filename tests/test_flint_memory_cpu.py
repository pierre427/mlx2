"""Flint admission must follow its actual full/sliding cache geometry."""
import pytest

from mlx2.adapters.flint import FlintAdapter
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController


def adapter(dtype="bfloat16"):
    result = FlintAdapter.__new__(FlintAdapter)
    result.config = {
        "layer_types": ["full_attention"] * 14 + ["sliding_attention"] * 35,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sliding_window": 513,
        "torch_dtype": dtype,
    }
    return result


def test_m3_16k_geometry_fits_without_reducing_workspace(monkeypatch):
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "4")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "2048")
    budget = adapter().cache_budget(mtp=False)
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=36, advisory_gib=28.08, cache_estimator=budget.project,
    )
    generic = SelfMTPLaneAdmissionController(host_memory_gib=36, advisory_gib=28.08)
    context = 16384 + 128
    # The live M3 reading was about 5.77 GiB; generic dense K/V alone
    # exceeded it. Keep the same service, driver and workspace reserves.
    required = controller.hard_reserve_gib + controller.lane_gib(context, 0)
    generic_required = generic.hard_reserve_gib + generic.lane_gib(context, 0)
    assert required < 5.77 < generic_required
    assert budget.transient_gib_per_lane == generic.transient_gib_per_lane
    assert budget.project(context) >= 14 * context * 2 * 4 * 128 * 2
    assert [budget.project(n) for n in (0, 512, 2048, 16384, 65536)] == sorted(
        budget.project(n) for n in (0, 512, 2048, 16384, 65536)
    )


def test_bound_covers_live_cache_and_restore_windows(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "4")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "2048")
    # Preserve the real layer/window/chunk counts; smaller head dimensions
    # make the CPU allocation check cheap across a wrapped sliding window.
    model = adapter()
    model.config.update(num_key_value_heads=1, head_dim=4)
    budget = model.cache_budget(mtp=False)
    full, sliding = KVCache(), RotatingKVCache(max_size=513)
    with mx.stream(mx.cpu):
        for offset in range(0, 16384, 512):
            kv = mx.zeros((1, 1, 512, 4), dtype=mx.bfloat16)
            full.update_and_fetch(kv, kv)
            sliding.update_and_fetch(kv, kv)
            mx.eval(full.state, sliding.state)
            # Include four independently retained exact restore windows,
            # even before all four are due. Larger chunks are charged by
            # the separate prefill transient hook.
            actual = 14 * full.nbytes + 35 * sliding.nbytes
            snapshots = budget.sliding_snapshots(offset + 512) * budget.sliding_snapshot_bytes(offset + 512)
            assert budget.project(offset + 512) >= actual + snapshots
    assert budget.prefill_transient_bytes(16384, 2048) > 0
    assert budget.prefill_transient_bytes(16384, 256) == 0


def test_unknown_dtype_is_conservative_and_mtp_refused():
    bf16 = adapter().cache_budget(mtp=False)
    unknown = adapter("unknown").cache_budget(mtp=False)
    assert unknown.item_bytes == 4
    assert unknown.project(16384) > bf16.project(16384)
    with pytest.raises(ValueError, match="no native MTP"):
        adapter().cache_budget(mtp=True)
