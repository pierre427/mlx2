"""Lane installer follow-ups from the Rapid-MLX port (2026-09-30).

* Below the crossover, a group's stock calls run as one stacked stock launch
  where a probe (bf16 and fp16, every row count below the crossover) proves
  it bitwise equal to separate calls.
* The simd backend refuses non-bf16 checkpoints (its kernels read bf16
  activations only).
* A narrower reinstall restores projections it no longer covers to stock.

CPU: MLX's quantized_matmul runs there too, so the probe and the stacked path
execute for real; only the device check is patched.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer as inst
from mlx2.runtime.lane import matmul as lm


def _quantized(k, n, bits, gs, seed=0):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    module = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    module.weight, module.scales, module.biases = mx.quantize(w, group_size=gs, bits=bits)
    return module


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _quantized(256, 64, 4, 64, seed=1)
        self.k_proj = _quantized(256, 32, 4, 64, seed=2)
        self.v_proj = _quantized(256, 32, 4, 64, seed=3)
        self.o_proj = _quantized(64, 256, 4, 64, seed=4)

    def __call__(self, x):
        return self.o_proj(self.q_proj(x) + mx.concatenate([self.k_proj(x), self.v_proj(x)], -1))


@pytest.fixture
def live(monkeypatch):
    """A device that 'has' a lane backend; lane launches themselves are not reached."""
    monkeypatch.setattr(inst, "available", lambda: True)
    monkeypatch.setattr(inst, "backend", lambda: "mpp")
    monkeypatch.setattr(lm, "backend", lambda: "mpp")
    inst.STATS.clear()
    yield
    inst.set_enabled(True, grouping=True)


def test_sub_crossover_calls_run_one_proven_stacked_stock_launch(live):
    model = _Attention()
    x = mx.random.normal((1, 3, 256), key=mx.random.key(6)).astype(mx.bfloat16)
    before = model(x)
    receipt = lane.install(model, min_rows=8)
    assert receipt["stock_stacked"] == {"groups": 1, "unproven": 0}
    assert receipt["law_id"] == "lane-matmul-v1+stock-below-8+grouped"   # law unchanged
    group = inst._group(model.q_proj)
    assert group.stock_stacked and group.call == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert mx.array_equal(model(x), before).item()
    stats = lane.stats()
    assert stats["stock_stacked_calls"] == 3 and stats["stock_below_min_rows"] == 4
    assert group.stock_last is None and group.stock_taken == 3   # released once all took it
    # The paired-A/B switch keeps stock arms on separate stock calls.
    lane.set_enabled(True, grouping=False)
    assert mx.array_equal(model(x), before).item()
    assert lane.stats()["stock_stacked_calls"] == 3
    lane.uninstall(model)


def test_the_probe_covers_bf16_and_fp16_and_every_sub_crossover_row(live, monkeypatch):
    real = inst._stock_matmul
    seen = []
    monkeypatch.setattr(inst, "_stock_matmul", lambda group, x: seen.append(
        (x.dtype, int(x.shape[0]))) or real(group, x))
    lane.install(_Attention(), min_rows=4)
    assert seen == [(dt, rows) for dt in (mx.bfloat16, mx.float16) for rows in (1, 2, 3)]


def test_a_stack_that_changes_stock_bits_keeps_separate_calls(live, monkeypatch):
    real = inst._stock_matmul
    monkeypatch.setattr(inst, "_stock_matmul", lambda group, x: real(group, x) + 1)
    model = _Attention()
    x = mx.random.normal((1, 3, 256), key=mx.random.key(7)).astype(mx.bfloat16)
    before = model(x)
    receipt = lane.install(model, min_rows=8)
    assert receipt["stock_stacked"] == {"groups": 0, "unproven": 1}
    assert not inst._group(model.q_proj).stock_stacked
    assert mx.array_equal(model(x), before).item()
    assert "stock_stacked_calls" not in lane.stats()
    lane.uninstall(model)


def test_exact_mode_and_off_device_installs_probe_nothing(monkeypatch):
    assert "stock_stacked" not in lane.install(_Attention(), min_rows=4)   # CPU: no backend
    monkeypatch.setattr(inst, "available", lambda: True)
    assert "stock_stacked" not in lane.install(_Attention(), min_rows=1)


def test_a_narrower_reinstall_is_refused_and_keeps_the_installed_layout(live):
    # Review r2 (2026-10-08): an installed model's coverage is fixed; a
    # reinstall that would cover other projections raises before changing any.
    model = _Attention()
    lane.install(model, min_rows=1)
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    before = [(type(model[n]), inst._group(model[n]), inst._prepared(model[n])) for n in names]
    assert before[0][0] is inst.LaneQuantizedLinear
    with pytest.raises(ValueError, match="coverage cannot change"):
        lane.install(model, min_rows=1, min_rows_by_format={"q8": 8})
    for name, (kind, group, prepared) in zip(names, before, strict=True):
        assert type(model[name]) is kind and model[name]._lane_min_rows == 1
        assert inst._group(model[name]) is group and inst._prepared(model[name]) is prepared


def test_simd_refuses_non_bf16_checkpoints():
    module = _quantized(256, 64, 4, 64)
    module.scales = module.scales.astype(mx.float16)
    module.biases = module.biases.astype(mx.float16)
    with pytest.raises(lane.LaneUnsupported, match="bf16 checkpoint"):
        lm.prepare(module, "simd")
    assert lm.prepare(module, "mpp").backend == "mpp"     # the M5 kernels take fp16


def test_a_group_result_is_held_only_until_every_member_took_it(live, monkeypatch):
    monkeypatch.setattr(inst, "lane_matmul", lambda x, lw: mx.zeros((*x.shape[:-1], lw.n), x.dtype))
    model = _Attention()
    lane.install(model, min_rows=1)
    group = inst._group(model.q_proj)
    x = mx.zeros((1, 2, 256), dtype=mx.bfloat16)
    model.q_proj(x)
    model.k_proj(x)
    assert group.last is not None and group.taken == 2
    model.v_proj(x)
    assert group.last is None and group.taken == 3
    lane.uninstall(model)
