"""CPU tests for the 2026-09-25 omlx Flash-Next ports (#3912 routed decode).

The #3912 kernels are Metal-only. Here they are replaced by a reference built
from the composed MLX ops, so the tests pin the plumbing: admission, shapes,
which down path runs, counters, and quiet declines. The kernels' numerics are
checked on the GPU by ``scripts/check_fn_routed_decode.py`` and the Metal-gated
test at the bottom (``MLX2_METAL_TESTS=1``).
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen4_routed_decode as RD

E, H, I = 16, 512, 64  # hidden % 512 == 0, intermediate % 512 != 0, % 64 == 0


def _switch(seed=0, dtype=mx.bfloat16, bits=4, group_size=64):
    mx.random.seed(seed)
    sw = QN.FusedGateUpSwitchGLU(H, I, E)
    sw.gate_up_proj.weight = (mx.random.normal((E, 2 * I, H)) * 0.05).astype(dtype)
    sw.down_proj.weight = (mx.random.normal((E, H, I)) * 0.05).astype(dtype)
    nn.quantize(sw, group_size=group_size, bits=bits)
    sw.set_dtype(dtype)
    QN._enable_routed_decode(sw, "off")
    sw.eval()
    return sw


def _route(seed, rows=1, dtype=mx.bfloat16):
    key = mx.random.key(seed)
    gates = mx.softmax(mx.random.normal((1, rows, E), key=key), axis=-1).astype(dtype)
    inds = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    return inds, scores / scores.sum(axis=-1, keepdims=True)


def _x(seed, rows=1, dtype=mx.bfloat16):
    return mx.random.normal((1, rows, H), key=mx.random.key(100 + seed)).astype(dtype)


@pytest.fixture
def reference_kernels(monkeypatch):
    """Swap the Metal kernels for the composed ops they stand in for."""
    calls = {"gate_up": 0, "down": 0}

    def gate_up_swiglu(x, indices, gate_up):
        calls["gate_up"] += 1
        inter = gate_up["weight"].shape[1] // 2
        gu = gate_up(x, indices)
        return QN.SwiGLU()(gu[..., inter:], gu[..., :inter]).reshape(10, inter)

    def down_combine(h, indices, scores, down):
        calls["down"] += 1
        y = down(h.reshape(indices.shape + (1, h.shape[-1])), indices).squeeze(-2)
        return (y * scores[..., None]).sum(axis=-2).reshape(-1)

    monkeypatch.setattr(RD, "gate_up_swiglu", gate_up_swiglu)
    monkeypatch.setattr(RD, "down_combine", down_combine)
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    return calls


def test_mode_parsing(monkeypatch):
    for raw, mode in (("", "off"), ("0", "off"), ("1", "gate_up"), ("two_launch", "two_launch")):
        monkeypatch.setenv(RD.ROUTED_DECODE_ENV, raw)
        assert RD.mode_from_env() == mode
    monkeypatch.delenv(RD.ROUTED_DECODE_ENV)
    assert RD.mode_from_env() == "off"
    monkeypatch.setenv(RD.ROUTED_DECODE_ENV, "fast")
    with pytest.raises(ValueError):
        RD.mode_from_env()


def test_admission_accepts_flash_next_geometry_and_refuses_the_rest():
    sw = _switch()
    inds, scores = _route(0)
    x = mx.expand_dims(_x(0), (-2, -3))
    ok = RD.admit_routed_decode(x, inds, scores, sw.gate_up_proj, sw.down_proj)
    assert ok.accepted, ok.reason
    two_rows = mx.expand_dims(_x(0, rows=2), (-2, -3))
    inds2, scores2 = _route(0, rows=2)
    assert RD.admit_routed_decode(two_rows, inds2, scores2, sw.gate_up_proj, sw.down_proj).reason == "not one token"
    half = x.astype(mx.float16)
    assert not RD.admit_routed_decode(half, inds, scores, sw.gate_up_proj, sw.down_proj).accepted
    assert not RD.admit_routed_decode(x, inds, scores.astype(mx.float32), sw.gate_up_proj, sw.down_proj).accepted
    assert not RD.admit_routed_decode(x, inds[..., :8], scores[..., :8], sw.gate_up_proj, sw.down_proj).accepted
    q8 = _switch(bits=8)
    assert "bits" in RD.admit_routed_decode(x, inds, scores, q8.gate_up_proj, q8.down_proj).reason
    g32 = _switch(group_size=32)
    assert "group size" in RD.admit_routed_decode(x, inds, scores, g32.gate_up_proj, g32.down_proj).reason


@pytest.mark.parametrize("variant", ["stock", "scalar"])
@pytest.mark.parametrize("mode", ["gate_up", "two_launch"])
def test_routed_modes_match_the_composed_body(reference_kernels, mode, variant):
    """With reference kernels, each mode reproduces the composed body exactly."""
    sw = _switch()
    for seed in range(4):
        x = _x(seed)
        inds, scores = _route(seed)
        sw.routed_decode_mode = "off"
        want = sw(x, inds, scores=scores, variant="stock")
        sw.routed_decode_mode = mode
        got = sw(x, inds, scores=scores, variant=variant)
        assert got.shape == want.shape == (1, 1, H)
        assert got.dtype == want.dtype
        if variant == "stock" or mode == "two_launch":
            assert mx.array_equal(got, want).item()
    assert sw.routed_decode_calls == 4
    assert sw.routed_decode_fallbacks == 0
    assert reference_kernels["gate_up"] == 4
    assert reference_kernels["down"] == (4 if mode == "two_launch" else 0)
    if mode == "two_launch":
        assert sw._last_fused_variant == "routed_two_launch"


def test_two_launch_without_scores_degrades_to_gate_up(reference_kernels):
    sw = _switch()
    x = _x(1)
    inds, _ = _route(1)
    sw.routed_decode_mode = "off"
    want = sw(x, inds)
    sw.routed_decode_mode = "two_launch"
    got = sw(x, inds)
    assert mx.array_equal(got, want).item()
    assert reference_kernels == {"gate_up": 1, "down": 0}


def test_multi_token_forwards_decline_quietly(reference_kernels):
    sw = _switch()
    sw.routed_decode_mode = "two_launch"
    x = _x(2, rows=3)
    inds, scores = _route(2, rows=3)
    sw(x, inds, scores=scores, variant="stock")
    assert sw.routed_decode_calls == 0 and sw.routed_decode_fallbacks == 0
    assert reference_kernels == {"gate_up": 0, "down": 0}


def test_ineligible_one_token_is_counted_and_runs_the_composed_body(reference_kernels):
    sw = _switch(bits=8)
    sw.routed_decode_mode = "gate_up"
    x = _x(3)
    inds, scores = _route(3)
    got = sw(x, inds, scores=scores, variant="stock")
    sw.routed_decode_mode = "off"
    assert mx.array_equal(got, sw(x, inds, scores=scores, variant="stock")).item()
    assert sw.routed_decode_fallbacks == 1
    assert "bits" in sw.routed_decode_last_fallback
    assert reference_kernels["gate_up"] == 0


def test_without_metal_the_route_falls_back(monkeypatch):
    monkeypatch.setattr(RD, "runtime_supported", lambda: False)
    sw = _switch()
    sw.routed_decode_mode = "gate_up"
    x = _x(4)
    inds, scores = _route(4)
    sw(x, inds, scores=scores, variant="stock")
    assert sw.routed_decode_last_fallback == "Metal runtime unavailable"


def test_block_setter_validates():
    from types import SimpleNamespace

    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=10,
    )
    block = QN.Qwen3NextSparseMoeBlock(args)
    assert block.switch_mlp.routed_decode_mode == "off"
    assert block.set_moe_routed_decode_mode("two_launch") == "two_launch"
    assert block.switch_mlp.routed_decode_mode == "two_launch"
    with pytest.raises(ValueError):
        block.set_moe_routed_decode_mode("on")


def test_flash_next_policy_switch_is_opt_in_and_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert "moe_routed_decode" not in default.as_dict()
    assert RD.ROUTED_DECODE_ENV not in default.environment()
    for mode in ("gate_up", "two_launch"):
        chosen = FlashNextPolicy.from_mapping({"moe_routed_decode": mode})
        assert chosen.environment()[RD.ROUTED_DECODE_ENV] == mode
        assert chosen.as_dict()["moe_routed_decode"] == mode
    with pytest.raises(ValueError, match="moe_routed_decode"):
        FlashNextPolicy.from_mapping({"moe_routed_decode": "on"})


@pytest.mark.skipif(
    os.environ.get("MLX2_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="Metal oracle test; set MLX2_METAL_TESTS=1 on an idle GPU",
)
def test_metal_kernels_track_the_composed_body():
    """Kernel vs composed body at the tolerance the GPU gate measured."""
    with mx.stream(mx.gpu):
        sw = _switch()
        for seed in range(8):
            x = _x(seed)
            inds, scores = _route(seed)
            sw.routed_decode_mode = "off"
            want = sw(x, inds, scores=scores, variant="stock").astype(mx.float32)
            for mode in ("gate_up", "two_launch"):
                sw.routed_decode_mode = mode
                got = sw(x, inds, scores=scores, variant="stock").astype(mx.float32)
                rel = (mx.linalg.norm(got - want) / mx.linalg.norm(want)).item()
                assert rel < 0.03, (mode, seed, rel)
