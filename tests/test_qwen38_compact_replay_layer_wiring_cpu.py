"""The production fused GDN layer stages compact replay on the served view (CPU).

The real ``qwen38_27b.GatedDeltaNet`` at Qwen3.8 geometry, compact replay
enabled with the real ``QWEN38_STATE_PLANE_OPERATIONS``, runs
``_try_fused_decode`` on a speculative block over a one-row
``SegmentedBatchArraysCache`` (the served native-MTP shape).  Only the Metal
entry points are replaced: the runtime/served-silu gates, the replay-verify
kernel (a CPU fake with the kernel's output geometry) and the reconstruction
kernel (a recording fake).  Everything between them -- admission,
authorization, plan construction, the view's rollback record, the split into
the row, trim and re-join -- is production code.  No numerical claim is made
here; exactness lives in test_qwen38_compact_replay_segmented_cpu.py.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen38_27b
from mlx2.runtime.models import qwen38_fused_gdn as route
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen3_5 import TextModelArgs
from mlx2.runtime.models.qwen38_state_plane import QWEN38_STATE_PLANE_OPERATIONS
from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache


def _layer():
    args = TextModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    )
    obj = qwen38_27b.GatedDeltaNet(args)
    obj.set_dtype(mx.bfloat16)
    obj.eval()
    obj.set_fused_gdn_enabled(True)
    obj.set_fused_gdn_compact_replay_enabled(True, QWEN38_STATE_PLANE_OPERATIONS)
    return obj


def _operands(obj, width):
    return [mx.zeros((1, width, n), mx.bfloat16)
            for n in (obj.conv_dim, obj.value_dim, obj.num_v_heads, obj.num_v_heads)]


def _row(obj, fill=0.0):
    row = ArraysCache(size=2)
    row[0] = mx.full((1, 3, obj.conv_dim), fill, mx.bfloat16)
    row[1] = mx.full((1, 48, 128, 128), fill, mx.float32)
    row.start_speculation()
    return row


@pytest.fixture
def fake_metal(monkeypatch):
    calls = {"verify": 0, "reconstruct": []}
    monkeypatch.setattr(route.kernels, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(
        route.verify_kernels, "probe_qwen4_fused_gdn_replay_verify", lambda *a, **k: 8
    )

    def replay_verify(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, state, *rest, **kw):
        calls["verify"] += 1
        steps = int(qkv.shape[1])
        return (
            mx.zeros((1, steps, z.shape[-1]), qkv.dtype),
            mx.full(conv_state.shape, 7.0, conv_state.dtype),
            mx.full(state.shape, 7.0, state.dtype),
            mx.zeros((1, steps - 1, 16, 128), qkv.dtype),
            mx.zeros((1, steps - 1, 48, 128), mx.float32),
            mx.zeros((1, steps - 1, 48), mx.float32),
        )

    def reconstruct(state, keys, corrections, decay, accepted, *, threadgroup_y):
        calls["reconstruct"].append((int(accepted), threadgroup_y))
        return state + float(accepted)

    monkeypatch.setattr(route.verify_kernels, "qwen4_fused_gdn_replay_verify", replay_verify)
    monkeypatch.setattr(route.verify_kernels, "qwen4_fused_gdn_reconstruct", reconstruct)
    return calls


def test_one_row_segmented_view_engages_compact_and_rewinds_through_the_row(fake_metal):
    obj = _layer()
    row = _row(obj)
    view = SegmentedBatchArraysCache([row])
    steps = 4
    view.prepare(lengths=[steps])
    out = obj._try_fused_decode(*_operands(obj, steps), None, view)
    view.finalize()
    assert out is not None, obj.fused_gdn_counters["last_fallback"]
    counts = obj.fused_gdn_counters
    assert counts["compact_verify_calls"] == 1 and counts["verify_calls"] == 1
    assert counts["compact_tape_bytes"] > 0 and counts["fallbacks"] == 0
    assert fake_metal["verify"] == 1
    # The final pair was split into the row; the record lives on the row.
    assert float(row[1][0, 0, 0, 0]) == 7.0 and len(row._rollbacks) == 1
    # Accept 2 of 4: the row replays the plan's accepted prefix from the
    # pinned checkpoint (zeros) through the reconstruction kernel.
    view.trim(2)
    assert fake_metal["reconstruct"] == [(2, 8)]
    assert counts["compact_rollback_calls"] == 1
    assert float(row[1][0, 0, 0, 0]) == 2.0 and float(view[1][0, 0, 0, 0]) == 2.0


def test_two_row_view_never_reaches_compact_and_reason_is_explicit(fake_metal):
    obj = _layer()
    rows = [_row(obj), _row(obj)]
    view = SegmentedBatchArraysCache(rows)
    view.prepare(lengths=[4, 4])
    old = [list(row.cache) for row in rows]
    # Multi-row blocks go to the batched verify, which declines compact mode
    # before dispatch and keeps the fallback.
    assert obj._try_fused_decode(*[mx.zeros((2,) + x.shape[1:], x.dtype)
                                   for x in _operands(obj, 4)], None, view) is None
    view.finalize()
    assert fake_metal["verify"] == 0
    assert all(a is b for row, before in zip(rows, old) for a, b in zip(row.cache, before))


def test_refused_authorization_records_the_message(fake_metal):
    class Other(ArraysCache):
        pass

    obj = _layer()
    other = Other(size=2)
    other[0] = mx.zeros((1, 3, obj.conv_dim), mx.bfloat16)
    other[1] = mx.zeros((1, 48, 128, 128), mx.float32)
    other.start_speculation()
    assert obj._try_fused_decode(*_operands(obj, 4), None, other) is None
    reason = obj.fused_gdn_counters["last_fallback"]
    assert reason.startswith(
        "compact replay state contract refused: StatePlaneOperationRefused: "
    ), reason
    assert "got Other" in reason
    assert fake_metal["verify"] == 0
