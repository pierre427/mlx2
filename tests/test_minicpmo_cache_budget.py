"""MiniCPM-o cache budget (CPU only; no model load).

Fixture geometry is openbmb/MiniCPM-o-2_6 @ 06849bfd36da (the artifact named
in qualification/runs/multimodal-processors-20260918/receipt.json).  Its LLM
is served as mlx-vlm's qwen2 language model: 28 layers of plain GQA, 4 KV
heads x 128 (hidden 3584 / 28 heads; the config has no head_dim), bf16.
``use_sliding_window`` is false, and mlx-vlm's qwen2 has no sliding path.
"""

import pytest

from mlx2.adapters.mlx_vlm import MiniCPMOAdapter
from mlx2.adapters.mlx_vlm_memory import (
    DENSE_TRANSIENT_GIB_PER_LANE,
    SlidingKVCacheBudget,
)
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

KIB = 1024


def _config(**overrides):
    return {
        "model_type": "minicpmo",
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "hidden_size": 3584,
        "sliding_window": 131072,
        "use_sliding_window": False,
        "max_window_layers": 28,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
        **overrides,
    }


def _budget(**overrides):
    config = _config(**overrides)
    return SlidingKVCacheBudget.from_qwen2_config(config, mtp=False, root_config=config)


def test_adapter_declares_a_config_derived_cache_budget():
    assert hasattr(MiniCPMOAdapter, "cache_budget")
    adapter = object.__new__(MiniCPMOAdapter)
    adapter.identity = {"config": _config()}
    assert adapter.cache_budget(mtp=False) == _budget()
    with pytest.raises(ValueError, match="MTP"):
        adapter.cache_budget(mtp=True)


def test_plain_gqa_geometry_with_derived_head_dim():
    budget = _budget()
    assert (budget.global_layers, budget.global_kv_heads, budget.global_head_dim) == (28, 4, 128)
    assert budget.sliding_layers == 0
    assert budget.global_bytes_per_token == 56 * KIB
    assert _budget(head_dim=64).global_head_dim == 64
    with pytest.raises(ValueError):
        _budget(hidden_size=3585)


@pytest.mark.parametrize("n", [32768, 131072])
def test_projection_is_within_5pct_of_hand_computed_bytes(n):
    hand = n * 56 * KIB
    projected = _budget().project(n)
    assert hand <= projected < 1.05 * hand


def test_cost_is_linear_with_no_sliding_term():
    budget = _budget()
    per_token = 56 * KIB + budget.transcript_bytes_per_token
    assert budget.project(32768) - budget.project(1024) == (32768 - 1024) * per_token
    assert budget.project(0) == 256 * per_token + 28 * 4096


def test_admission_no_longer_charges_the_full_attention_envelope():
    budget = _budget()
    assert budget.transient_gib_per_lane == DENSE_TRANSIENT_GIB_PER_LANE
    assert budget.transient_basis.startswith("provisional")
    geometry = C(cache_estimator=budget.project, transient_gib_per_lane=budget.transient_gib_per_lane)
    envelope = C(transient_gib_per_lane=budget.transient_gib_per_lane)
    assert envelope.lane_gib(32768, 0) > 15
    assert 2.5 < geometry.lane_gib(32768, 0) < 3.0
