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
    assert sw.routed_decode_down_calls == (4 if mode == "two_launch" else 0)
    assert sw.routed_decode_degraded == 0
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
    # Counted as a gate+up launch, and reported as a degraded two_launch.
    assert sw.routed_decode_calls == 1
    assert sw.routed_decode_down_calls == 0
    assert sw.routed_decode_degraded == 1


def test_multi_token_forwards_decline_quietly(reference_kernels):
    sw = _switch()
    sw.routed_decode_mode = "two_launch"
    x = _x(2, rows=3)
    inds, scores = _route(2, rows=3)
    sw(x, inds, scores=scores, variant="stock")
    assert sw.routed_decode_calls == 0 and sw.routed_decode_fallbacks == 0
    assert sw.routed_decode_down_calls == 0 and sw.routed_decode_degraded == 0
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


def _block(monkeypatch, bits=4):
    """A Flash-Next-shaped MoE block whose expert tables admit routed decode."""
    from types import SimpleNamespace

    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", True)
    monkeypatch.setattr(QN, "_MOE_SHARED_IN_GATHER", False)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=10,
    )
    mx.random.seed(21)
    block = QN.Qwen3NextSparseMoeBlock(args)
    switch = block.switch_mlp
    switch.gate_up_proj.weight = mx.random.normal((E, 2 * I, H)) * 0.05
    switch.down_proj.weight = mx.random.normal((E, H, I)) * 0.05
    nn.quantize(switch, group_size=64, bits=bits)
    block.set_dtype(mx.bfloat16)
    block.eval()
    return block


def test_block_two_launch_under_the_stock_expert_kernel_runs_the_routed_down(
    monkeypatch, reference_kernels
):
    """The stock expert branch called the switch without scores, so a selected
    ``two_launch`` silently ran gate+up only while its call counter rose."""
    block = _block(monkeypatch)
    block.set_fused_expert_kernel_mode("stock")
    x = _x(7)
    want = block(x)
    block.set_moe_routed_decode_mode("two_launch")
    got = block(x)
    switch = block.switch_mlp
    assert reference_kernels == {"gate_up": 1, "down": 1}
    assert switch.routed_decode_calls == switch.routed_decode_down_calls == 1
    assert switch.routed_decode_degraded == switch.routed_decode_fallbacks == 0
    assert got.dtype == want.dtype and mx.array_equal(got, want).item()
    # A multi-token forward keeps the stock branch and counts nothing.
    block(_x(8, rows=3))
    assert switch.routed_decode_calls == 1
    assert reference_kernels == {"gate_up": 1, "down": 1}


def test_block_pass_through_is_bit_identical_when_admission_declines(monkeypatch):
    """8-bit experts are refused; the scores handed to the switch then take
    the stock ops, so the output equals the routed-off block bit for bit."""
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    block = _block(monkeypatch, bits=8)
    block.set_fused_expert_kernel_mode("stock")
    x = _x(9)
    want = block(x)
    block.set_moe_routed_decode_mode("two_launch")
    got = block(x)
    switch = block.switch_mlp
    assert mx.array_equal(got, want).item()
    assert switch.routed_decode_fallbacks == 1 and "bits" in switch.routed_decode_last_fallback
    assert switch.routed_decode_calls == switch.routed_decode_down_calls == 0


def test_flash_next_policy_switch_defaults_to_the_fold_and_off_is_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert default.as_dict()["moe_routed_decode"] == "gate_up_down_shared"
    assert default.environment()[RD.ROUTED_DECODE_ENV] == "gate_up_down_shared"
    off = FlashNextPolicy.from_mapping({"moe_routed_decode": "off"})
    assert "moe_routed_decode" not in off.as_dict()
    assert RD.ROUTED_DECODE_ENV not in off.environment()
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


# --------------------------------------------------------------------------
# Split gate/up tables (the served Flash-Next layout, MLX_QWEN4_MOE_FUSED_GATE_UP=0)
# --------------------------------------------------------------------------


def _split_switch(seed=0, bits=4, group_size=64):
    mx.random.seed(seed)
    sw = QN.FusedDownSwitchGLU(H, I, E)
    for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
        getattr(sw, proj).weight = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
    nn.quantize(sw, group_size=group_size, bits=bits)
    sw.set_dtype(mx.bfloat16)
    QN._enable_routed_decode(sw, "off")
    sw.eval()
    return sw


@pytest.fixture
def split_reference_kernels(monkeypatch):
    """Composed stand-ins for the split gate+up and served-down kernels."""
    calls = {"split_gate_up": 0, "served_down": 0, "down": 0}

    def split_gate_up_swiglu(x, indices, gate, up):
        calls["split_gate_up"] += 1
        return QN.SwiGLU()(up(x, indices), gate(x, indices)).reshape(10, I)

    def composed_down(h, indices, scores, down):
        y = down(h.reshape(indices.shape + (1, h.shape[-1])), indices).squeeze(-2)
        return (y * scores[..., None]).sum(axis=-2).reshape(-1)

    def served_down(h, indices, scores, down, rows=None):
        calls["served_down"] += 1
        return composed_down(h, indices, scores, down)

    def down_combine(h, indices, scores, down):
        calls["down"] += 1
        return composed_down(h, indices, scores, down)

    monkeypatch.setattr(RD, "split_gate_up_swiglu", split_gate_up_swiglu)
    monkeypatch.setattr(RD, "served_down", served_down)
    monkeypatch.setattr(RD, "down_combine", down_combine)
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    return calls


def test_split_admission_accepts_served_layout_and_refuses_the_rest():
    sw = _split_switch()
    inds, scores = _route(0)
    x = mx.expand_dims(_x(0), (-2, -3))
    ok = RD.admit_split_routed_decode(x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj)
    assert ok.accepted, ok.reason
    q8 = _split_switch(bits=8)
    assert "bits" in RD.admit_split_routed_decode(x, inds, scores, q8.gate_proj, q8.up_proj, q8.down_proj).reason
    # gate and up tables must match each other and the down table
    other = _split_switch(seed=1)
    other.up_proj.weight = other.up_proj.weight[:, : I // 2]
    other.up_proj.scales = other.up_proj.scales[:, : I // 2]
    other.up_proj.biases = other.up_proj.biases[:, : I // 2]
    assert "do not match" in RD.admit_split_routed_decode(
        x, inds, scores, other.gate_proj, other.up_proj, other.down_proj).reason

    class Streamed(type(sw.gate_proj)):
        pass

    sub = Streamed.__new__(Streamed)
    nn.Module.__init__(sub)
    sub.weight, sub.scales, sub.biases = (sw.gate_proj[k] for k in ("weight", "scales", "biases"))
    sub.bits, sub.group_size, sub.mode = 4, 64, "affine"
    assert "resident" in RD.admit_split_routed_decode(x, inds, scores, sub, sw.up_proj, sw.down_proj).reason


@pytest.mark.parametrize("mode", ["gate_up", "two_launch"])
def test_split_tables_route_through_the_kernels(split_reference_kernels, mode):
    sw = _split_switch()
    for seed in range(3):
        x = _x(seed)
        inds, scores = _route(seed)
        sw.routed_decode_mode = "off"
        want = sw(x, inds, scores=scores, variant="stock")
        sw.routed_decode_mode = mode
        got = sw(x, inds, scores=scores, variant="stock")
        assert got.shape == want.shape == (1, 1, H)
        assert mx.array_equal(got, want).item()
    assert sw.routed_decode_calls == 3 and sw.routed_decode_fallbacks == 0
    assert split_reference_kernels["split_gate_up"] == 3
    assert split_reference_kernels["down"] == (3 if mode == "two_launch" else 0)
    assert sw.routed_down_calls == (3 if mode == "two_launch" else 0)


def test_gate_up_down_runs_served_down_where_tile4_would(split_reference_kernels, monkeypatch):
    seen = []

    def refusal(switch_mlp, inter, dtype, indices, scores, variant):
        seen.append(variant)
        return None

    monkeypatch.setattr(QN, "_served_down_refusal", refusal)
    sw = _split_switch()
    x = _x(5)
    inds, scores = _route(5)
    sw.routed_decode_mode = "off"
    want = sw(x, inds, scores=scores, variant="stock")
    sw.routed_decode_mode = "gate_up_down"
    got = sw(x, inds, scores=scores, variant="auto")
    assert mx.array_equal(got, want).item()
    assert seen == ["auto"]
    assert sw.routed_decode_calls == 1 and sw.routed_down_calls == 1
    assert sw._last_fused_variant == "routed_gate_up_down"
    assert split_reference_kernels == {"split_gate_up": 1, "served_down": 1, "down": 0}


def test_gate_up_down_declines_the_down_half_off_tile4(split_reference_kernels):
    """variant="stock" (or a CPU/odd-shape tile4 refusal) keeps the block's
    own down after the gate+up kernel, and the decline is counted."""
    sw = _split_switch()
    x = _x(6)
    inds, scores = _route(6)
    sw.routed_decode_mode = "off"
    want = sw(x, inds, scores=scores, variant="stock")
    sw.routed_decode_mode = "gate_up_down"
    got = sw(x, inds, scores=scores, variant="stock")
    assert mx.array_equal(got, want).item()
    assert sw.routed_decode_calls == 1
    assert sw.routed_down_calls == 0 and sw.routed_down_fallbacks == 1
    assert "not tile4" in sw.routed_down_last_fallback
    # tile4 selected, but the tile4 admission refuses this geometry / CPU
    sw(x, inds, scores=scores, variant="tile4")
    assert sw.routed_down_fallbacks == 2
    assert sw.routed_down_last_fallback.startswith("tile4 admission")
    assert split_reference_kernels["served_down"] == 0


def test_gate_up_down_without_scores_is_gate_up(split_reference_kernels):
    sw = _split_switch()
    x = _x(7)
    inds, _ = _route(7)
    sw.routed_decode_mode = "off"
    want = sw(x, inds)
    sw.routed_decode_mode = "gate_up_down"
    assert mx.array_equal(sw(x, inds), want).item()
    assert sw.routed_down_calls == 0 and sw.routed_down_fallbacks == 0


def test_served_block_layout_accepts_routed_modes(monkeypatch):
    """The served profile (split tables) used to force routed decode off and
    refuse the setter; it now carries the selected mode."""
    from types import SimpleNamespace

    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    monkeypatch.setattr(QN, "_MOE_ROUTED_DECODE", "gate_up_down")
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=10,
    )
    block = QN.Qwen3NextSparseMoeBlock(args)
    assert isinstance(block.switch_mlp, QN.FusedDownSwitchGLU)
    assert block.switch_mlp.routed_decode_mode == "gate_up_down"
    for mode in RD.MODES:
        assert block.set_moe_routed_decode_mode(mode) == mode


def test_policy_accepts_gate_up_down():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    chosen = FlashNextPolicy.from_mapping({"moe_routed_decode": "gate_up_down"})
    assert chosen.environment()[RD.ROUTED_DECODE_ENV] == "gate_up_down"
    assert chosen.as_dict()["moe_routed_decode"] == "gate_up_down"


def test_scheduling_env_parsing(monkeypatch):
    monkeypatch.delenv(RD.VIEWS_ENV, raising=False)
    assert RD.views_from_env() is True
    monkeypatch.setenv(RD.VIEWS_ENV, "0")
    assert RD.views_from_env() is False
    monkeypatch.delenv(RD.DOWN_ROWS_ENV, raising=False)
    assert RD.down_rows_from_env() == 2
    monkeypatch.setenv(RD.DOWN_ROWS_ENV, "4")
    assert RD.down_rows_from_env() == 4
    monkeypatch.setenv(RD.DOWN_ROWS_ENV, "3")
    with pytest.raises(ValueError):
        RD.down_rows_from_env()
    with pytest.raises(ValueError):
        RD.set_served_down_rows(8)


def test_expert_views_alias_the_table_and_rebuild_on_replace():
    sw = _split_switch()
    layer = sw.gate_proj
    prev = RD.expert_views_enabled()
    try:
        RD.set_expert_views(True)
        views = RD.expert_operands(layer)
        assert [v.shape[0] for v in views] == [1, 1, 1]
        for whole, view in zip((layer["weight"], layer["scales"], layer["biases"]), views):
            assert RD._address(view) == RD._address(whole)
        assert RD.expert_view_count(layer) == 3
        assert RD.expert_operands(layer) is views  # cached
        layer.weight = mx.array(layer.weight)  # replaced array -> rebuilt
        again = RD.expert_operands(layer)
        assert again is not views and RD._address(again[0]) == RD._address(layer.weight)
        RD.set_expert_views(False)
        whole = RD.expert_operands(layer)
        assert whole[0] is layer["weight"] and whole[0].shape[0] == E
    finally:
        RD.set_expert_views(prev)


def test_expert_view_keeps_a_non_contiguous_table_whole():
    t = mx.zeros((E, 2 * I, H // 8), mx.uint32)
    half = t[:, I:, :]  # strided rows: a [:1] view would not index experts
    mx.eval(half)
    assert RD._expert_view(half) is half


# --------------------------------------------------------------------------
# Shared-expert fold (moe_routed_decode = "gate_up_down_shared")
# --------------------------------------------------------------------------


def _served_block(monkeypatch, mode="gate_up_down_shared", seed=0):
    from types import SimpleNamespace

    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    monkeypatch.setattr(QN, "_MOE_ROUTED_DECODE", mode)
    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=10,
    )
    block = QN.Qwen3NextSparseMoeBlock(args)
    for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
        getattr(block.switch_mlp, proj).weight = (mx.random.normal(shape) * 0.05)
    nn.quantize(block, group_size=64, bits=4)
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(
        nn.Linear(H, 1, bias=False), group_size=64, bits=8)
    block.set_dtype(mx.bfloat16)
    block.set_fused_expert_kernel_mode("tile4")  # tile4 declines on CPU -> stock down
    block.eval()
    return block


@pytest.fixture
def fold_reference(monkeypatch):
    """Composed stand-in for the folded launches; served_down admission forced."""
    calls = {"fold": 0}

    def shared_fold_decode(x, indices, scores, gate, up, down, shared, gate_logit, rows=None):
        calls["fold"] += 1
        xe = mx.expand_dims(x, (-2, -3))
        h = QN.SwiGLU()(up(xe, indices), gate(xe, indices))
        y = (down(h, indices).squeeze(-2) * scores[..., None]).sum(axis=-2)
        return (y + mx.sigmoid(gate_logit) * shared(x)).reshape(-1)

    monkeypatch.setattr(RD, "shared_fold_decode", shared_fold_decode)
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(QN, "_served_down_refusal", lambda *a: None)
    return calls


def test_shared_fold_admission():
    shared = QN.Qwen3NextMLP(H, I)
    nn.quantize(shared, group_size=64, bits=4)
    shared.set_dtype(mx.bfloat16)
    assert RD.admit_shared_fold(shared, H, I).accepted
    assert "no shared expert" in RD.admit_shared_fold(None, H, I).reason
    shared.down_proj.__dict__["_lane_prepared"] = object()
    assert "lane matmul" in RD.admit_shared_fold(shared, H, I).reason
    del shared.down_proj.__dict__["_lane_prepared"]
    q8 = QN.Qwen3NextMLP(H, I)
    nn.quantize(q8, group_size=64, bits=8)
    q8.set_dtype(mx.bfloat16)
    assert "b8g64 != b4g64" in RD.admit_shared_fold(q8, H, I).reason
    object.__setattr__(shared, "_prefill_counts", {})
    assert "tensorfold" in RD.admit_shared_fold(shared, H, I).reason


def test_shared_fold_replaces_the_block_tail(monkeypatch, fold_reference):
    block = _served_block(monkeypatch)
    for seed in range(3):
        x = _x(seed)
        block.set_moe_routed_decode_mode("off")
        want = block(x)
        block.set_moe_routed_decode_mode("gate_up_down_shared")
        got = block(x)
        assert got.shape == want.shape == (1, 1, H)
        assert mx.array_equal(got, want).item()
    assert block.shared_fold_calls == 3 and block.shared_fold_fallbacks == 0
    assert block.switch_mlp.routed_decode_calls == 3 and block.switch_mlp.routed_down_calls == 3
    assert fold_reference["fold"] == 3


def test_shared_fold_declines_with_reasons(monkeypatch, fold_reference):
    block = _served_block(monkeypatch)
    x = _x(4)
    # multi-token forwards decline quietly
    block(_x(4, rows=3))
    assert block.shared_fold_calls == 0 and block.shared_fold_fallbacks == 0
    monkeypatch.setattr(QN, "_COMPILE_GLUE", True)
    block(x)
    assert block.shared_fold_fallbacks == 1
    assert "glue" in block.shared_fold_last_fallback
    monkeypatch.setattr(QN, "_COMPILE_GLUE", False)
    shared = QN.Qwen3NextMLP(H, I)
    nn.quantize(shared, group_size=64, bits=8)
    shared.set_dtype(mx.bfloat16)
    block.shared_expert = shared
    block(x)
    assert block.shared_fold_fallbacks == 2
    assert "shared gate_proj" in block.shared_fold_last_fallback
    assert fold_reference["fold"] == 0


def test_policy_accepts_gate_up_down_shared():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    chosen = FlashNextPolicy.from_mapping({"moe_routed_decode": "gate_up_down_shared"})
    assert chosen.environment()[RD.ROUTED_DECODE_ENV] == "gate_up_down_shared"


def test_shared_fold_admission_sees_through_the_row_exact_subclass():
    """The row-exact verify route swaps its own QuantizedLinear subclass onto
    the trunk; its one-row call is the plain one, so the fold still admits."""
    from mlx2.runtime.models import qwen4_row_exact as RX

    shared = QN.Qwen3NextMLP(H, I)
    nn.quantize(shared, group_size=64, bits=4)
    shared.set_dtype(mx.bfloat16)
    for linear in (shared.gate_proj, shared.up_proj, shared.down_proj):
        linear.__class__ = RX._subclass(RX._RowExactQuantizedLinear, type(linear))
    assert type(shared.gate_proj) is not nn.QuantizedLinear
    assert RD.admit_shared_fold(shared, H, I).accepted
