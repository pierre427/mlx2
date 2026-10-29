"""Synthetic Source AST route ownership and source-generation only; no GPU."""
from contextlib import contextmanager
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import source_ast_m1_expert as gu
from mlx2.adapters import deepseek_v41_ast_moe as ast


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")))


@pytest.mark.parametrize("kind", ["original", "q1"])
def test_all16_occurrences_keep_the_same_m1_gu_source(kind):
    from mlx2.runtime.gguf.source_gate_up_fusion import kernel_spec
    assert gu.middle_spec(kind) == kernel_spec(kind, selected_rows=1)
    for rows in (1, 2, 7, 12, 13, 16):
        p = gu.plan(rows, kind)
        assert p["dot_rows"] == 1 and p["grid"][1] == rows
        assert p["output_shape"] == (rows, 2304)


def test_logical16row_moe_maps_scores_and_combines_ascending_experts(monkeypatch):
    from mlx2.runtime.gguf import source_ast_down_rows as down
    from mlx2.runtime.models import deepseek_v41_quant as quant
    from mlx2.runtime.gguf import source_middle_fusion as mf
    from mlx2.adapters import deepseek_v41_source_math as combine
    monkeypatch.setattr(quant, "fused_quantize", lambda x, kind: x)
    monkeypatch.setattr(mf, "exp_spelling", lambda: "metal::exp")
    calls = []
    def middle(x, rows, scores, expert, **kwargs):
        calls.append((expert, rows.size))
        return mx.broadcast_to((x[rows, 0] + scores + expert*10)[:, None], (rows.size, 2304))
    monkeypatch.setattr(gu, "middle", middle)
    monkeypatch.setattr(down, "down", lambda x, expert: mx.broadcast_to(x[:, :1]*(expert+1), (x.shape[0], 5120)))
    def ordered(parts, values, order):
        packed = mx.concatenate(parts)
        offsets = combine.ordered_offsets(values, order)
        result = mx.zeros((16, 5120))
        for slot in range(6):
            result = result + packed[mx.array(offsets[:, slot])]
        return result
    monkeypatch.setattr(combine, "ordered_combine", ordered)
    class Stream:
        _active = False
        @contextmanager
        def window(self):
            self._active = True
            try:
                yield
            finally:
                self._active = False
        def acquire(self, prefix, expert):
            assert self._active
            return expert
        def observe_routes(self, prefix, values):
            assert values.shape == (16, 6)
    weights = SimpleNamespace(_ready=lambda: None, expert_stream=Stream(),
        expert_math_policy="routed_fused_fp32", _adaptive_expert_plan=lambda *a: None,
        scheduled_expert_calls={})
    values = np.array([np.roll(np.arange(6), row % 6) for row in range(16)], np.uint32)
    scores = np.arange(96, dtype=np.float32).reshape(16, 6)
    x = np.broadcast_to(np.arange(16, dtype=np.float32)[:, None], (16, 5120)).copy()
    out = ast.moe(weights, mx.array(x), mx.array(values), mx.array(scores), "blk.14")
    expected = np.zeros((16, 5120), np.float32)
    for row in range(16):
        total = np.float32(0)
        for expert in range(6):
            slot = int(np.flatnonzero(values[row] == expert)[0])
            total = np.float32(total + np.float32((row+scores[row, slot]+expert*10)*(expert+1)))
        expected[row] = total
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert calls == [(expert, 16) for expert in range(6)]
    assert weights.scheduled_expert_calls["source_ast_m1_logical_m16_moe"] == 1
    assert weights.scheduled_expert_calls["source_ast_m1_gpu_occurrences"] == 96
    assert weights.scheduled_expert_calls["source_ast_m1_python_token_forward_calls"] == 0
    assert weights.source_route_host_materializations == 1
