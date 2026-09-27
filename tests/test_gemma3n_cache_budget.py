"""Gemma 3n cache budget (CPU only; no model load).

Fixture geometry is google/gemma-3n-E2B-it @ 5e092ebca197 (the artifact named
in qualification/runs/multimodal-processors-20260918/receipt.json): 30 layers,
the last 10 reuse earlier K/V (num_kv_shared_layers), so 20 layers own a cache
-- 4 global and 16 sliding (window 512), 2 KV heads x 256, bf16.  Gemma 3n
runs on mlx-vlm's own cache types, which keep no restore snapshots.
"""

import pytest

from mlx2.adapters.mlx_vlm import Gemma3nAdapter
from mlx2.adapters.mlx_vlm_memory import (
    DENSE_TRANSIENT_GIB_PER_LANE,
    SlidingKVCacheBudget,
)
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

KIB = 1024


def _text(**overrides):
    return {
        "model_type": "gemma3n_text",
        "num_hidden_layers": 30,
        "num_kv_shared_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_size": 2048,
        "sliding_window": 512,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
        "layer_types": [
            "full_attention" if index % 5 == 4 else "sliding_attention"
            for index in range(30)
        ],
        **overrides,
    }


def _budget(**overrides):
    return SlidingKVCacheBudget.from_gemma3n_config(_text(**overrides), mtp=False)


def test_adapter_declares_a_config_derived_cache_budget():
    assert hasattr(Gemma3nAdapter, "cache_budget")
    adapter = object.__new__(Gemma3nAdapter)
    adapter.identity = {"config": {"model_type": "gemma3n", "text_config": _text()}}
    assert adapter.cache_budget(mtp=False) == _budget()
    with pytest.raises(ValueError, match="MTP"):
        adapter.cache_budget(mtp=True)


def test_kv_sharing_leaves_twenty_cache_owning_layers():
    budget = _budget()
    assert (budget.global_layers, budget.sliding_layers) == (4, 16)
    assert budget.global_bytes_per_token == 8 * KIB
    assert budget.sliding_bytes_per_token == 32 * KIB
    assert budget.checkpoint_copies == 0
    unshared = _budget(num_kv_shared_layers=0)
    assert (unshared.global_layers, unshared.sliding_layers) == (6, 24)
    with pytest.raises(ValueError):
        _budget(layer_types=None)


@pytest.mark.parametrize("n", [32768, 131072])
def test_projection_is_within_5pct_of_hand_computed_bytes(n):
    hand = n * 8 * KIB + (512 - 1 + 2048) * 32 * KIB
    projected = _budget().project(n)
    assert hand <= projected < 1.05 * hand


def test_slope_and_flat_sliding_term_past_the_window():
    budget = _budget()
    per_token = 8 * KIB + budget.transcript_bytes_per_token
    assert budget.project(32768) - budget.project(4096) == (32768 - 4096) * per_token
    capacity = lambda n: -(-n // 256) * 256 + 256  # noqa: E731
    flat = {budget.project(n) - capacity(n) * per_token for n in (2560, 4096, 32768, 131072)}
    assert flat == {2560 * 32 * KIB + 20 * 4096}
    assert budget.project(256) < budget.project(1024) < budget.project(2560)


def test_admission_no_longer_charges_the_full_attention_envelope():
    budget = _budget()
    assert budget.transient_gib_per_lane == DENSE_TRANSIENT_GIB_PER_LANE
    assert budget.transient_basis.startswith("provisional")
    geometry = C(cache_estimator=budget.project, transient_gib_per_lane=budget.transient_gib_per_lane)
    envelope = C(transient_gib_per_lane=budget.transient_gib_per_lane)
    assert envelope.lane_gib(32768, 0) > 15
    assert geometry.lane_gib(32768, 0) < 1.5
