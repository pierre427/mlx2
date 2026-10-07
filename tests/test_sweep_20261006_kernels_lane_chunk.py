"""KR-04 (2026-10-06 sweep): with chunk_above_max, a grouped lane-matmul
member ran its OWN LaneWeights past max_rows while calls up to max_rows ran
the group's stacked weights.  Split-K depends on N, so one row got different
bits from a wide call than from a narrow one."""

import mlx.core as mx
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer, policy
from mlx2.runtime.lane.matmul import split_k


def _q(k, n, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    m = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
    return m


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _q(128, 64, 1)
        self.k_proj = _q(128, 64, 2)


def test_members_split_k_differs_from_the_group():
    """The premise: stacking changes split-K (gate/up of a 12288 MLP)."""
    assert split_k(24576, 4096, 64, 4) != split_k(12288, 4096, 64, 4)


def test_wide_grouped_calls_chunk_the_group_launch(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    seen = []

    def fake(x, lw):
        seen.append((int(x.shape[0]), lw.n))
        # Column j carries j, so a member's slice shows which columns it got.
        return mx.broadcast_to(mx.arange(lw.n, dtype=x.dtype), (*x.shape[:-1], lw.n))

    monkeypatch.setattr(installer, "lane_matmul", fake)
    model = _Attn()
    resolved = policy.resolve(
        policy.detect(model),
        overrides={"min_rows": {"q4": 1}, "max_rows": 16, "chunk_above_max": True},
    )
    receipt = lane.apply_policy(model, resolved)
    installer.STATS.clear()
    try:
        assert receipt["groups"], receipt
        assert "+chunk-above-16-stacked" in receipt["law_id"]
        for rows in (8, 40):
            seen.clear()
            x = mx.zeros((rows, 128), dtype=mx.bfloat16)
            q, k = model.q_proj(x), model.k_proj(x)
            # Every launch is the stacked group (N = 128), wide or narrow.
            assert {n for _rows, n in seen} == {128}, seen
            assert sum(r for r, _n in seen) == rows  # one launch serves both
            assert mx.array_equal(q[0], mx.arange(64, dtype=q.dtype)).item()
            assert mx.array_equal(k[0], mx.arange(64, 128, dtype=k.dtype)).item()
        counts = lane.stats()
        assert counts["lane_chunked_calls"] == 2
        assert counts["lane_chunked_launches"] == 3
        assert counts["group_reuses"] == 2
    finally:
        installer.STATS.clear()
        lane.uninstall(model)
