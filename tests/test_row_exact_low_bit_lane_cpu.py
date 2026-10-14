"""Row-exact verify: kernel-declined (2/3-bit) layers on the lane law (default off).

Pierre 2026-10-14 asked why the 2/3-bit layers of a row-exact verify cannot
go through the TensorFold-derived lane matmul.  The row-exact contract is
that each verify row equals the same layer's one-token decode call; the lane
law is row-invariant (1..128 rows a call, and per chunk past that), so a
declined layer may run the lane law instead of one stock call per verify
row, provided its one-token decode calls run it too.  These CPU tests cover
the plumbing (dispatch, receipts, policy, identity binding); the lane
kernels are Metal-only, so the lane product is faked here with the stock
product and its calls are counted.  Bit-exactness and speed are GPU work.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from test_qwen4_row_exact_verify import _fake_gdn_engaged, _low_bit_tiny_model

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime import lane as LANE
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.lane import matmul as LM
from mlx2.runtime.models import row_exact_qmv as REQ
from mlx2.runtime.models import qwen4_row_exact as RX


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def fake_lane(monkeypatch):
    """The M5 backend, with lane_matmul replaced by a counted stock product."""
    calls = []

    def lane_matmul(x, lw):
        calls.append(int(x.size // x.shape[-1]))
        w = mx.dequantize(lw.weight, lw.scale_bias[..., 0].T, lw.scale_bias[..., 1].T,
                          group_size=lw.group_size, bits=lw.bits)
        y = x @ w.T.astype(x.dtype)
        return y if lw.bias is None else y + lw.bias

    monkeypatch.setattr(LM, "backend", lambda: "mpp")
    monkeypatch.setattr(LANE, "lane_matmul", lane_matmul)
    return calls


def _bf16_low_bit_model():
    """fp32 activations (CPU gather_mm needs them) with bf16 scales on the
    3-bit layers, which the lane preparation requires."""
    model = _low_bit_tiny_model()
    for m in _low_bit_layers(model):
        m.scales = m.scales.astype(mx.bfloat16)
        m.biases = m.biases.astype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _low_bit_layers(model):
    return [m for _, m in model.language_model.named_modules()
            if isinstance(m, nn.QuantizedLinear) and m.bits == 3]


def test_declined_layers_run_the_lane_law_for_decode_and_verify(cpu, fake_lane):
    model = _bf16_low_bit_model()
    low = _low_bit_layers(model)
    handle = RX.install(model, low_bit_lane=True)
    status = handle.status()
    lane = status["low_bit_lane"]
    assert lane["count"] == len(low) and lane["by_format"] == {"affine-q3-g32": len(low)}
    assert lane["law_id"] == "lane-matmul-v1+row-exact-low-bit"
    assert lane["intended"] == len(low)
    assert lane["qualified"] is False and lane["refused"] == {}
    # Served by the lane now, so no longer labelled as the per-row slow path.
    assert status["per_row_layers"]["quantized"]["count"] == 0
    assert status["per_row_layers"]["slow_path"] is False

    layer = low[0]
    k = layer.scales.shape[-1] * layer.group_size
    # One-token decode outside a window: the lane law.
    fake_lane.clear()
    mx.eval(layer(mx.zeros((2, 1, k))))
    assert fake_lane == [2]
    # A long prefill call outside a window runs the lane too, in chunks.
    fake_lane.clear()
    mx.eval(layer(mx.zeros((1, 200, k))))
    assert fake_lane == [128, 72]
    # Inside a window, the direct row_exact_qmv entry point takes the lane too.
    record = REV.Window(3)
    with REV.window(record):
        (_, route) = REQ.quantized_linear(layer, mx.zeros((1, 3, k)))
        mx.eval(layer(mx.zeros((1, 3, k))))
    assert route == "lane_exact" and record.stages["projections"] == {"lane_exact": 1}

    handle.enable(True)
    _fake_gdn_engaged(handle)
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 7, 3, 9, 2, 8]]), cache=cache))
    start = handle.snapshot()
    (hidden, _) = handle.verify_backbone(mx.array([[4, 6, 5]]), cache)
    mx.eval(handle.verify_logits(hidden))
    stages = handle.status()["stages"]["projections"]
    assert stages.get("lane_exact", 0) >= len(low)
    receipt = handle.receipt(start)
    assert receipt["low_bit_lane"]["law_id"] == lane["law_id"]
    assert "per_row_layers" not in receipt
    handle.remove()
    assert all(REQ.lane_preparation(m) is None for m in low)


def test_an_explicit_option_that_cannot_prepare_every_declined_layer_refuses_startup(cpu, monkeypatch):
    """Codex 2026-10-14 MEDIUM 4: off the GPU (or with refused layers) the
    option silently left every layer per row and its receipt omitted the
    label, so a qualification could pass with zero lane engagement.  An
    explicit selection that cannot engage on every declined layer refuses."""
    monkeypatch.setattr(LM, "backend", lambda: None)
    model = _bf16_low_bit_model()
    low = _low_bit_layers(model)
    with pytest.raises(ValueError, match="lane_unavailable"):
        RX.install(model, low_bit_lane=True)
    assert all(REQ.lane_preparation(m) is None for m in low)
    assert getattr(model, "_mlx2_row_exact_verify", None) is None


def test_an_option_with_no_declined_layer_refuses_startup(cpu, fake_lane):
    from test_qwen4_row_exact_verify import _quantized_tiny_model

    with pytest.raises(ValueError, match="no layer"):
        RX.install(_quantized_tiny_model(), low_bit_lane=True)


def test_default_install_prepares_nothing(cpu, fake_lane):
    model = _bf16_low_bit_model()
    handle = RX.install(model)
    assert handle.status()["low_bit_lane"] is None
    assert all(REQ.lane_preparation(m) is None for m in _low_bit_layers(model))
    with pytest.raises(ValueError, match="low_bit_lane"):
        RX.install(model, low_bit_lane=True)


def test_policy_field_is_default_off_and_needs_the_route():
    assert "row_exact_low_bit_lane" not in FlashNextPolicy().as_dict()
    with pytest.raises(ValueError, match="requires row_exact_verify"):
        FlashNextPolicy(row_exact_low_bit_lane=True)
    on = FlashNextPolicy(row_exact_verify=True, row_exact_low_bit_lane=True)
    assert on.as_dict()["row_exact_low_bit_lane"] is True
    with pytest.raises(ValueError, match="boolean"):
        FlashNextPolicy(row_exact_verify=True, row_exact_low_bit_lane="yes")


def test_the_lane_law_enters_the_numerics_contract_only_when_used():
    from mlx2.adapters.flash_next import FlashNextAdapter

    block = SimpleNamespace(fused_expert_kernel_mode="off", switch_mlp=SimpleNamespace(),
                            moe_window_consumers=())
    adapter = SimpleNamespace(policy=FlashNextPolicy(), _diagnostic_modules=[block],
                              row_exact_verify=None)
    plain = FlashNextAdapter.execution_numerics_contract(adapter)
    assert "row_exact_low_bit_lane_v1" not in plain
    adapter.row_exact_verify = SimpleNamespace(lane_layers={
        "count": 2, "law_id": "lane-matmul-v1+row-exact-low-bit",
        "by_format": {"affine-q3-g32": 2}})
    bound = FlashNextAdapter.execution_numerics_contract(adapter)
    assert bound["row_exact_low_bit_lane_v1"]["law_id"].startswith("lane-matmul-v1")
    assert bound["flash_next_expert_arithmetic_v1"] == plain["flash_next_expert_arithmetic_v1"]


def test_an_input_the_lane_refuses_falls_back_to_the_stock_law_at_every_width(cpu, fake_lane, monkeypatch):
    """The lane refuses non-bf16/fp16 activations at every width, so decode
    and verify both take the stock law (per-row in the window): still equal."""
    model = _bf16_low_bit_model()
    handle = RX.install(model, low_bit_lane=True)
    layer = _low_bit_layers(model)[0]
    k = layer.scales.shape[-1] * layer.group_size

    def refuse(x, lw):
        raise LANE.LaneUnsupported("activations must be bf16 or fp16")

    monkeypatch.setattr(LANE, "lane_matmul", refuse)
    x = mx.random.normal((1, 3, k))
    serial = mx.concatenate([layer(x[:, r : r + 1]) for r in range(3)], axis=1)
    record = REV.Window(3)
    with REV.window(record):
        (y, route) = REQ.quantized_linear(layer, x)
    assert route == "per_row"
    assert mx.array_equal(y, serial).item()
    handle.remove()


def test_the_lane_law_does_not_depend_on_prefill_segmentation(cpu, monkeypatch):
    """Codex 2026-10-14 HIGH 2: with stock above 128 rows, a cold 200-row
    prompt chunk ran stock while an APCv2 hit or a rolling checkpoint that
    left a <=128-row tail ran the lane: one prompt, two states, one
    namespace.  Prepared layers now take the lane at every width (chunked
    past the lane's per-call limit, which keeps each row's bits), so cold
    and split prefills compute every row the same way.  The fake lane here
    is row-wise by construction; the real kernels' row invariance is the
    GPU gate (scripts/lane_matmul_gate.py)."""
    calls = []

    def lane_matmul(x, lw):
        calls.append(int(x.size // x.shape[-1]))
        w = mx.dequantize(lw.weight, lw.scale_bias[..., 0].T, lw.scale_bias[..., 1].T,
                          group_size=lw.group_size, bits=lw.bits)
        flat = x.reshape(-1, x.shape[-1])
        rows = [flat[r : r + 1] @ w.T.astype(x.dtype) for r in range(flat.shape[0])]
        return mx.concatenate(rows, axis=0).reshape(*x.shape[:-1], -1)

    monkeypatch.setattr(LM, "backend", lambda: "mpp")
    monkeypatch.setattr(LANE, "lane_matmul", lane_matmul)
    model = _bf16_low_bit_model()
    handle = RX.install(model, low_bit_lane=True)
    layer = _low_bit_layers(model)[0]

    def stock(self, x):
        raise AssertionError("a prepared layer ran the stock product")

    monkeypatch.setattr(nn.QuantizedLinear, "__call__", stock)
    k = layer.scales.shape[-1] * layer.group_size
    x = mx.random.normal((1, 200, k), key=mx.random.key(7))
    cold = layer(x)                                          # one 200-row chunk
    warm = mx.concatenate([layer(x[:, :150]), layer(x[:, 150:])], axis=1)   # hit at 150
    tail = mx.concatenate([layer(x[:, :72]), layer(x[:, 72:])], axis=1)     # hit at 72
    assert mx.array_equal(cold, warm).item() and mx.array_equal(cold, tail).item()
    assert sorted(calls) == sorted([128, 72, 128, 22, 50, 72, 128])
    handle.remove()


def test_fused_projection_tables_exclude_prepared_layers(cpu):
    """A fused q/k/v/index or GDN in-projection table reads its members'
    weights with one stock quantized_matmul (attn_fused_rows is default on).
    A prepared member must keep its own call, or one-token decode would run
    stock while its verify rows run the lane."""
    from test_qwen4_row_exact_verify import _quantized_tiny_model
    from mlx2.runtime.models import qwen4_exp as Q

    model = _quantized_tiny_model()
    attention = next(m for _, m in model.language_model.named_modules() if isinstance(m, Q.Attention))
    gdn = next(m for _, m in model.language_model.named_modules() if isinstance(m, Q.GatedDeltaNet))
    assert attention._fused_projection_table() is not None
    assert gdn._fused_inproj_table() is not None
    object.__setattr__(attention.k_proj, REQ.LANE_ATTR, object())
    object.__setattr__(gdn.in_proj_z, REQ.LANE_ATTR, object())
    assert attention._fused_projection_table() is None
    assert gdn._fused_inproj_table() is None


def test_tensorfold_prefill_cannot_be_combined_with_the_option():
    with pytest.raises(ValueError, match="tensorfold_prefill"):
        FlashNextPolicy(row_exact_verify=True, row_exact_low_bit_lane=True, tensorfold_prefill=True)
