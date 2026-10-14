import numpy as np
import pytest

from mlx2.runtime.streamed_load import WeightStreamRequest, load_streamed
from test_disk_weight_streaming import (
    _moe_checkpoint,
    _moe_model,
    _shards,
    _quantizer,
    _logits,
    PROMPT,
)


def test_overlap_explicit_cpu_fallback_and_state_identity(tmp_path, monkeypatch):
    root, weight_map, records = _moe_checkpoint(tmp_path, monkeypatch)
    managers = []
    try:
        outputs = []
        for enabled in (False, True):
            model = _moe_model(monkeypatch, fused=False)
            loaded = load_streamed(
                model,
                root,
                _shards(weight_map),
                request=WeightStreamRequest(
                    mode="moe_experts",
                    budget_bytes=1 << 20,
                    headroom_bytes=0,
                    expert_overlap=enabled,
                ),
                sanitize=model.sanitize,
                quantize=_quantizer(model),
                records=records,
                weight_map=weight_map,
                top_k=2,
            )
            managers.append(loaded.manager)
            outputs.append(np.asarray(_logits(model, PROMPT)))
        np.testing.assert_array_equal(*outputs)
        assert managers[0].apc_revision() != managers[1].apc_revision()
        assert managers[1].reserved_bytes() == managers[0].reserved_bytes() + (
            managers[1].plan.capacity_experts * managers[1].plan.expert_bytes
        )
        assert "overlap" not in managers[0].receipt()
        assert managers[1].receipt()["overlap"]["default"] is False
        assert managers[1].counters()["stream_overlap_attempts_total"] == 0
        assert managers[1].counters()["stream_overlap_declined_total"] > 0
    finally:
        for manager in managers:
            manager.close()


def test_overlap_request_fails_closed_for_other_modes():
    with pytest.raises(ValueError, match="MoE-only"):
        WeightStreamRequest(mode="dense_mlp", budget_bytes=1 << 20, expert_overlap=True)
    with pytest.raises(ValueError, match="boolean"):
        WeightStreamRequest(mode="moe_experts", budget_bytes=1 << 20, expert_overlap=1)
