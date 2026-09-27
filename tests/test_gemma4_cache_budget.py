"""Gemma 4 cache budget (CPU only; no model load).

The fixtures carry the geometry of the local 8-bit conversions
(~/mlx-models/gemma-4-{31B,26B-A4B}-MLX-8bit/config.json).  Without a budget,
admission charged the 0.44 GiB/1K full-attention envelope, 56.3 GiB for a 128K
31B request, against ~10 GiB of global K/V plus a flat sliding-window term.
"""

from dataclasses import replace

import pytest

from mlx2.adapters.gemma4 import Gemma431BAdapter, Gemma4A4BAdapter
from mlx2.adapters.mlx_vlm_memory import (
    DENSE_TRANSIENT_GIB_PER_LANE,
    GEMMA4_DENSE_TRANSIENT_GIB_PER_LANE,
    GEMMA4_MOE_TRANSIENT_GIB_PER_LANE,
    SlidingKVCacheBudget,
)
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

KIB = 1024
GIB = 1 << 30


def _text(*, sparse):
    layers = 30 if sparse else 60
    return {
        "model_type": "gemma4_text",
        "num_hidden_layers": layers,
        "hidden_size": 2816 if sparse else 5376,
        "enable_moe_block": sparse,
        "num_experts": 128 if sparse else None,
        "num_attention_heads": 16 if sparse else 32,
        "num_key_value_heads": 8 if sparse else 16,
        "num_global_key_value_heads": 2 if sparse else 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "dtype": "bfloat16",
        "max_position_embeddings": 262144,
        "layer_types": [
            "full_attention" if index % 6 == 5 else "sliding_attention"
            for index in range(layers)
        ],
    }


def _budget(*, sparse, **overrides):
    text = {**_text(sparse=sparse), **overrides}
    return SlidingKVCacheBudget.from_gemma4_config(text, mtp=False)


def _adapter(cls, *, sparse):
    adapter = object.__new__(cls)
    adapter.identity = {
        "config": {"model_type": "gemma4", "text_config": _text(sparse=sparse)}
    }
    return adapter


@pytest.mark.parametrize("cls,sparse", [
    (Gemma431BAdapter, False), (Gemma4A4BAdapter, True),
])
def test_both_adapters_declare_a_config_derived_cache_budget(cls, sparse):
    assert hasattr(cls, "cache_budget")
    adapter = _adapter(cls, sparse=sparse)
    budget = adapter.cache_budget(mtp=False)
    # Before the engine configures a step, the adapter's own default applies.
    assert budget == replace(_budget(sparse=sparse), prefill_step=cls.default_prefill_step)
    assert budget.as_dict()["schema"] == "gemma4-full-sliding-cache-geometry-v1"
    # The sliding cap follows the server's prefill chunk once it is known.
    adapter.execution_config(max_lanes=4, prefill_step=512)
    assert adapter.cache_budget(mtp=False).sliding_token_cap == 1024 + 512
    with pytest.raises(ValueError, match="MTP"):
        adapter.cache_budget(mtp=True)


def test_geometry_matches_the_local_configs():
    dense = _budget(sparse=False)
    assert (dense.global_layers, dense.global_kv_heads, dense.global_head_dim) == (10, 4, 512)
    assert (dense.sliding_layers, dense.sliding_kv_heads, dense.sliding_head_dim) == (50, 16, 256)
    assert dense.item_bytes == 2
    # K and V are both stored even though V derives from the key projection.
    assert dense.global_bytes_per_token == 80 * KIB
    assert dense.sliding_bytes_per_token == 800 * KIB
    sparse = _budget(sparse=True)
    assert (sparse.global_layers, sparse.sliding_layers) == (5, 25)
    assert sparse.global_bytes_per_token == 20 * KIB
    assert sparse.sliding_bytes_per_token == 200 * KIB


@pytest.mark.parametrize("sparse,global_kib,sliding_kib", [
    (False, 80, 800), (True, 20, 200),
])
def test_128k_projection_is_within_5pct_of_hand_computed_bytes(sparse, global_kib, sliding_kib):
    n, window, step, snapshots = 131072, 1024, 2048, 4
    # Global K/V grows with the context; each live sliding layer holds at
    # most the window plus one prefill chunk, and each of the lane's four
    # restore snapshots holds exactly the window.
    hand = (
        n * global_kib * KIB
        + (window - 1 + step) * sliding_kib * KIB
        + snapshots * window * sliding_kib * KIB
    )
    projected = _budget(sparse=sparse).project(n)
    assert projected >= hand
    assert projected / hand < 1.05


@pytest.mark.parametrize("sparse", [False, True])
def test_slope_above_the_window_is_global_kv_only(sparse):
    budget = _budget(sparse=sparse)
    per_token = budget.global_bytes_per_token + budget.transcript_bytes_per_token
    for low, high in ((16384, 32768), (65536, 131072), (131072, 262144)):
        assert (budget.project(high) - budget.project(low)) == (high - low) * per_token


@pytest.mark.parametrize("sparse", [False, True])
def test_sliding_term_is_flat_past_the_window(sparse):
    budget = _budget(sparse=sparse)
    per_token = budget.global_bytes_per_token + budget.transcript_bytes_per_token

    def non_global(n):
        capacity = -(-n // 256) * 256 + 256
        return budget.project(n) - capacity * per_token

    flat = non_global(8192)
    layers = budget.global_layers + budget.sliding_layers
    assert flat == (3072 + 4 * 1024) * budget.sliding_bytes_per_token + layers * 4096
    assert all(non_global(n) == flat for n in (8193, 16384, 131072, 262144))
    # Below the cap it grows, and the projection stays monotone throughout.
    assert non_global(512) < non_global(2048) < flat
    values = [budget.project(n) for n in range(0, 20000, 97)]
    assert values == sorted(values)


def test_128k_31b_request_is_no_longer_charged_the_full_attention_envelope():
    budget = _budget(sparse=False)
    envelope = C(transient_gib_per_lane=budget.transient_gib_per_lane)
    geometry = C(
        cache_estimator=budget.project,
        transient_gib_per_lane=budget.transient_gib_per_lane,
    )
    old = envelope.lane_gib(131072, 0)
    new = geometry.lane_gib(131072, 0)
    assert old > 56.5
    assert 16 < new < 17
    # 26B-A4B: about 4.0 GiB instead of 56.4.
    sparse = _budget(sparse=True)
    moe = C(cache_estimator=sparse.project, transient_gib_per_lane=sparse.transient_gib_per_lane)
    assert 3.5 < moe.lane_gib(131072, 0) < 4.5


@pytest.mark.parametrize("sparse,snapshot_gib,old_gib", [
    (False, 3.125, 9.375), (True, 0.78125, 2.34375),
])
def test_snapshot_term_charges_the_cropped_window_not_window_plus_chunk(
    sparse, snapshot_gib, old_gib
):
    """Four snapshots of exactly the window, which serving now records.

    The term used to charge each snapshot at the live cap (window + prefill
    step): 9.375 GiB on the 31B at the 2048 step, for snapshots serving never
    recorded.  ``state_checkpoint`` crops each one to the window.
    """
    budget = _budget(sparse=sparse)
    n = 16384
    term = budget.sliding_snapshots(n) * budget.sliding_snapshot_bytes(n)
    assert term / GIB == pytest.approx(snapshot_gib)
    assert budget.sliding_snapshots(n) * budget.sliding_token_cap * (
        budget.sliding_bytes_per_token
    ) / GIB == pytest.approx(old_gib)
    # Below the window a snapshot holds only the prompt so far.
    assert budget.sliding_snapshot_bytes(300) == 300 * budget.sliding_bytes_per_token
    assert budget.sliding_snapshots(300) == 1


def test_transients_are_the_measured_gemma4_constants():
    # provenance/lane-transient-gemma4.json: largest per-lane ordinary-decode
    # transient 0.5355 GiB (31B) and 0.4445 GiB (26B-A4B), both the first
    # decode after prefill.  The constant is a k=2 figure charged at 1/3 on
    # the ordinary route, sized 3 x 1.25 x the spike.
    dense, sparse = _budget(sparse=False), _budget(sparse=True)
    assert dense.transient_gib_per_lane == GEMMA4_DENSE_TRANSIENT_GIB_PER_LANE == 2.0
    assert sparse.transient_gib_per_lane == GEMMA4_MOE_TRANSIENT_GIB_PER_LANE == 1.7
    for budget, spike in ((dense, 0.5355), (sparse, 0.4445)):
        ordinary_charge = budget.transient_gib_per_lane * C.TRANSIENT_SCALE[0]
        assert 1.2 * spike <= ordinary_charge <= 1.3 * spike
        assert budget.transient_basis.startswith("measured")
        assert budget.as_dict()["workspace"].startswith("measured")
    # The families that were not measured keep the provisional placeholder.
    assert DENSE_TRANSIENT_GIB_PER_LANE == 3.1


def test_topology_is_read_from_the_config_not_hard_coded():
    text = _text(sparse=False)
    derived = dict(text, sliding_window_pattern=6)
    del derived["layer_types"]
    assert SlidingKVCacheBudget.from_gemma4_config(derived, mtp=False) == _budget(sparse=False)
    # Shared-KV layers own no cache.
    shared = _budget(sparse=False, num_kv_shared_layers=12)
    assert (shared.global_layers, shared.sliding_layers) == (8, 40)
    # Without k_eq_v the global layers use the ordinary KV head count.
    plain = _budget(sparse=False, attention_k_eq_v=False)
    assert plain.global_kv_heads == 16
    # An unknown activation dtype is charged at fp32.
    unknown = dict(text)
    del unknown["dtype"]
    assert SlidingKVCacheBudget.from_gemma4_config(unknown, mtp=False).item_bytes == 4
    for bad in (
        dict(text, layer_types=text["layer_types"][:-1]),
        dict(text, layer_types=["chunked_attention"] * 60),
        dict(text, num_kv_shared_layers=60),
    ):
        with pytest.raises(ValueError):
            SlidingKVCacheBudget.from_gemma4_config(bad, mtp=False)


@pytest.mark.parametrize("length", [5, 16, 40, 64, 97, 200])
def test_bound_covers_real_mlx2_cache_bytes_after_chunked_prefill(monkeypatch, length):
    """Drive mlx2's own caches the way chunked prefill does, on the CPU."""
    import mlx.core as mx

    from mlx2.runtime.models.cache import (
        KVCache,
        RotatingKVCache,
        record_state_checkpoints,
    )

    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "32")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "2")
    window, step, heads, dim = 16, 32, 2, 8
    text = {
        "num_hidden_layers": 3,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "num_key_value_heads": heads,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        "head_dim": dim,
        "global_head_dim": 2 * dim,
        "sliding_window": window,
        "dtype": "bfloat16",
    }
    budget = SlidingKVCacheBudget.from_gemma4_config(text, mtp=False, prefill_step=step)
    assert (budget.checkpoint_copies, budget.checkpoint_stride) == (2, 32)
    with mx.stream(mx.cpu):
        caches = [RotatingKVCache(max_size=window), KVCache(), RotatingKVCache(max_size=window)]
        peak = sliding_peak = processed = 0
        while processed < length:
            chunk = min(step, length - processed)
            for cache, kv_heads, head_dim in (
                (caches[0], heads, dim), (caches[1], 1, 2 * dim), (caches[2], heads, dim),
            ):
                keys = mx.ones((1, kv_heads, chunk, head_dim), dtype=mx.bfloat16)
                cache.update_and_fetch(keys, keys + 1)
            processed += chunk
            mx.eval([c.state for c in caches])
            record_state_checkpoints(caches, [processed])
            peak = max(peak, sum(c.nbytes for c in caches))
            sliding_peak = max(sliding_peak, caches[0].nbytes + caches[2].nbytes)
        record_state_checkpoints(caches, [processed], force=True)
        token = mx.ones((1, heads, 1, dim), dtype=mx.bfloat16)
        caches[0].update_and_fetch(token, token)
        caches[2].update_and_fetch(token, token)
        caches[1].update_and_fetch(
            mx.ones((1, 1, 1, 2 * dim), dtype=mx.bfloat16),
            mx.ones((1, 1, 1, 2 * dim), dtype=mx.bfloat16),
        )
        mx.eval([c.state for c in caches])
        peak = max(peak, sum(c.nbytes for c in caches))
        sliding_peak = max(sliding_peak, caches[0].nbytes + caches[2].nbytes)
    # Restore snapshots are real, counted in nbytes, and bounded by the budget.
    assert len(caches[0]._checkpoints) <= budget.sliding_snapshots(length)
    if length > window:
        assert caches[0]._checkpoints
    assert 0 < peak <= budget.project(length + 1)
    # The sliding term alone (live window+chunk plus each snapshot) covers the
    # sliding caches; dropping the snapshot charge would not.
    def sliding_bound(copies):
        capacity = -(-(length + 1) // 256) * 256 + 256
        return (
            min(capacity, budget.sliding_token_cap) * budget.sliding_bytes_per_token
            * (1 + min(copies, budget.sliding_snapshots(length + 1)))
        )

    assert sliding_peak <= sliding_bound(budget.checkpoint_copies)
    if length > step + window:
        assert sliding_peak > sliding_bound(0)
