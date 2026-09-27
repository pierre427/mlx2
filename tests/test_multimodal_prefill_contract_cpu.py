"""Adapter validation at the serving CPU-token handoff uses no MLX."""

import importlib.abc
import sys
from types import SimpleNamespace

import pytest


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import during CPU prefill validation")
        return None


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    assert "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [BlockMLX(), *sys.meta_path])


def test_smol_scatter_only_receives_trust_on_exact_cold_media_tokens():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter
    from mlx2.serving import validated_adapter_prefill_inputs

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter.model = SimpleNamespace(indexed_scatter=True)
    adapter._vision_feature_reuse_enabled = False
    adapter._media_proof_key = b"test-key"
    request = {"_mlx2_prompt_tokens": [7, 42, 9, 42, 11],
               "_mlx2_media_token_end": 4,
               "_mlx2_media_fingerprint": "test-media"}
    request["_mlx2_media_proof"] = adapter._media_proof(
        request["_mlx2_prompt_tokens"], "test-media", 4
    )
    supplied = {"image_token_positions": (1, 3),
                "_mlx2_smol_positions_verified": True}

    cold = validated_adapter_prefill_inputs(
        adapter, request, [7, 42, 9, 42, 11], supplied
    )
    assert cold["_mlx2_smol_positions_verified"] is True
    assert supplied == {"image_token_positions": (1, 3),
                        "_mlx2_smol_positions_verified": True}

    partial_hit = validated_adapter_prefill_inputs(
        adapter, request, [9, 42, 11], supplied
    )
    assert "_mlx2_smol_positions_verified" not in partial_hit
    wrong_positions = validated_adapter_prefill_inputs(
        adapter, request, [7, 42, 9, 42, 11],
        {"image_token_positions": (1, 2),
         "_mlx2_smol_positions_verified": True},
    )
    assert "_mlx2_smol_positions_verified" not in wrong_positions
    forged = {**request, "_mlx2_media_proof": "forged"}
    assert "_mlx2_smol_positions_verified" not in validated_adapter_prefill_inputs(
        adapter, forged, [7, 42, 9, 42, 11], supplied
    )


def test_prefill_validator_rejects_invalid_adapter_result():
    from mlx2.serving import validated_adapter_prefill_inputs

    adapter = SimpleNamespace(validate_prefill_inputs=lambda *_: None)
    with pytest.raises(TypeError, match="must return a dict"):
        validated_adapter_prefill_inputs(adapter, {}, [1], {})
    assert validated_adapter_prefill_inputs(adapter, {}, [1], None) is None


def test_qwen_feature_reuse_requires_prepared_media_proof_and_base_revision():
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter
    from mlx2.serving import validated_adapter_prefill_inputs

    adapter = object.__new__(Qwen25VLCandidateAdapter)
    adapter._vision_feature_reuse_enabled = True
    adapter._grouped_vision_enabled = False
    adapter._media_proof_key = b"test-key"
    adapter.identity = {"fingerprint": "artifact-id"}
    request = {"_mlx2_prompt_tokens": [7, 42, 11],
               "_mlx2_media_token_end": 2,
               "_mlx2_media_fingerprint": "ordered-image"}
    request["_mlx2_media_proof"] = adapter._media_proof(
        request["_mlx2_prompt_tokens"], "ordered-image", 2
    )
    supplied = {"pixel_values": object(), "_mlx2_vision_feature_verified": True}
    cold = validated_adapter_prefill_inputs(adapter, request, [7, 42, 11], supplied)
    assert cold["_mlx2_vision_feature_verified"] is True
    assert cold["_mlx2_vision_feature_certificate"].artifact_fingerprint == "artifact-id"
    assert "_mlx2_vision_feature_verified" not in validated_adapter_prefill_inputs(
        adapter, request, [42, 11], supplied
    )
    assert "_mlx2_vision_feature_verified" not in validated_adapter_prefill_inputs(
        adapter, {**request, "_mlx2_lora_fingerprint": "selected-lora"},
        [7, 42, 11], supplied
    )
    assert "_mlx2_vision_feature_verified" not in validated_adapter_prefill_inputs(
        adapter, {**request, "_mlx2_media_proof": "forged"}, [7, 42, 11], supplied
    )
    assert adapter.execution_config(max_lanes=1, prefill_step=128)[
        "vision_feature_reuse"
    ] == "candidate_v1"


def test_tower_digest_is_covered_by_private_media_proof():
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter
    from mlx2.serving import validated_adapter_prefill_inputs

    adapter = object.__new__(Qwen25VLCandidateAdapter)
    adapter._vision_feature_reuse_enabled = True
    adapter._vision_feature_reuse_scope = "tower_inputs_v1"
    adapter._grouped_vision_enabled = False
    adapter.media_feature_cache = SimpleNamespace(max_bytes=256 << 20)
    adapter._media_proof_key = b"test-key"
    adapter.identity = {"fingerprint": "artifact-id"}
    tokens = [7, 42, 11]
    request = {"_mlx2_prompt_tokens": tokens, "_mlx2_media_token_end": 2,
               "_mlx2_media_fingerprint": "ordered-image",
               "_mlx2_tower_inputs_digest": "a" * 64}
    request["_mlx2_media_proof"] = adapter._media_proof(
        tokens, "ordered-image", 2, "a" * 64
    )
    supplied = {"pixel_values": object()}
    cold = validated_adapter_prefill_inputs(adapter, request, tokens, supplied)
    assert cold["_mlx2_vision_feature_certificate"].tower_sha256 == "a" * 64
    assert adapter.execution_config(max_lanes=1, prefill_step=128)["vision_feature_reuse"] == "tower_only_candidate_v1"
    assert adapter.execution_config(max_lanes=1, prefill_step=128)["vision_feature_cache_max_bytes"] == 256 << 20
    for altered in ({**request, "_mlx2_tower_inputs_digest": "b" * 64},
                    {**request, "_mlx2_media_fingerprint": "other"}):
        assert "_mlx2_vision_feature_certificate" not in validated_adapter_prefill_inputs(
            adapter, altered, tokens, supplied
        )


def test_generic_cold_media_feature_peak_reserve_does_not_charge_resident_bytes():
    from mlx2.adapters.pinned_vlm_candidate import PinnedVisionCandidateAdapter
    from mlx2.serving import cold_media_feature_peak_increment_gib

    adapter = object.__new__(PinnedVisionCandidateAdapter)
    adapter._vision_feature_reuse_scope = "tower_inputs_v1"
    # A full LRU still needs room for the new feature until eviction. Its
    # existing 256 MiB is already reflected in measured host/device headroom.
    adapter.media_feature_cache = SimpleNamespace(max_bytes=256 << 20,
                                                   bytes=256 << 20)
    request = {"_mlx2_prefill_inputs": {"pixel_values": object()},
               "_mlx2_tower_inputs_digest": "a" * 64}
    assert cold_media_feature_peak_increment_gib(
        adapter, request, cached_tokens=0, uncached_tokens=128
    ) == 0.25
    for changed, cached, remaining in (
        (request, 127, 1),
        (request, 0, 1),
        ({**request, "_mlx2_tower_inputs_digest": None}, 0, 128),
        ({**request, "_mlx2_lora_fingerprint": "selected"}, 0, 128),
        ({"_mlx2_tower_inputs_digest": "a" * 64}, 0, 128),
    ):
        assert cold_media_feature_peak_increment_gib(
            adapter, changed, cached_tokens=cached, uncached_tokens=remaining
        ) == 0.0
    adapter._vision_feature_reuse_scope = "prompt_v1"
    assert cold_media_feature_peak_increment_gib(
        adapter, request, cached_tokens=0, uncached_tokens=128
    ) == 0.0
    assert cold_media_feature_peak_increment_gib(
        object(), request, cached_tokens=0, uncached_tokens=128
    ) == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        cold_media_feature_peak_increment_gib(
            SimpleNamespace(cold_media_feature_peak_increment_bytes=lambda: True),
            request, cached_tokens=0, uncached_tokens=128,
        )


def test_two_cold_media_lanes_cannot_spend_the_same_feature_headroom():
    """Two attached lanes see one fixed reading until either begins prefill."""
    from mlx2.adapters.pinned_vlm_candidate import PinnedVisionCandidateAdapter
    from mlx2.serving import (
        admit_lane_headroom,
        cold_media_feature_peak_increment_gib,
        unmaterialized_lane_bytes,
    )

    adapter = object.__new__(PinnedVisionCandidateAdapter)
    adapter._vision_feature_reuse_scope = "tower_inputs_v1"
    adapter.media_feature_cache = SimpleNamespace(max_bytes=256 << 20)
    request = {"_mlx2_prefill_inputs": {"pixel_values": object()},
               "_mlx2_tower_inputs_digest": "a" * 64}
    feature_gib = cold_media_feature_peak_increment_gib(
        adapter, request, cached_tokens=0, uncached_tokens=128
    )
    assert feature_gib == 0.25
    controller = SimpleNamespace(
        hard_reserve_gib=0.5,
        lane_gib=lambda _context, _depth, *, cache_gib: 0.25 + cache_gib,
    )
    fixed_headroom = int(1.1 * (1 << 30))

    def admissions(feature_increment):
        attached = []
        decisions = []
        for _ in range(2):
            granted = unmaterialized_lane_bytes(attached)
            admitted, _depth_floor, required = admit_lane_headroom(
                controller, context_tokens=128, draft_depth=0,
                cache_gib=0.0, prefill_gib=feature_increment,
                headroom=lambda: fixed_headroom - granted,
                reclaim=lambda: False, evict=lambda: False,
                evictable=lambda: 0,
            )
            decisions.append(admitted)
            if admitted:
                attached.append(SimpleNamespace(
                    admission_reserved_gib=required - controller.hard_reserve_gib
                ))
        return decisions

    # Without the newly retained 256 MiB charge both requests fit the same
    # measurement. With it, the first grant protects the second admission.
    assert admissions(0.0) == [True, True]
    assert admissions(feature_gib) == [True, False]
