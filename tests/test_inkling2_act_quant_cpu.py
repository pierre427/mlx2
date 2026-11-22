"""Inkling 2 W4A4 fake activation quantization (``inkling2_moe_act_quant``), CPU.

The quantizers are checked against an independent NumPy reference of the
checkpoint contract (NVFP4 four-over-six MAE per token, MXFP8 block 32 with
round-up UE8M0 scales): E4M3 and E2M1 rounding come from enumerated grids
with ties to the even code, not from MLX.  The option is checked default
off, strict, bitwise inert when off, and engaged identically in the
reference and fused MoE paths.
"""

import numpy as np
import mlx.core as mx
import pytest

from test_inkling2_cpu import _run, _tiny_model
from test_inkling2_moe_cpu import _plan_case, _quantize_experts

from mlx2.runtime.models.inkling2 import act_quant as aq

E2M1_GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


# ----- independent NumPy reference -----


def _e4m3_grid():
    values = [m / 8 * 2.0 ** -6 for m in range(8)]  # subnormals (code 0..7)
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN
            values.append((1 + m / 8) * 2.0 ** (e - 7))
    return np.array(values, dtype=np.float64)


E4M3_GRID = _e4m3_grid()


def _round_to_grid(v, grid):
    """Nearest grid value of |v| (ties to the even code), saturating, signed."""
    a = np.minimum(np.abs(v.astype(np.float64)), grid[-1])
    hi = np.clip(np.searchsorted(grid, a, side="left"), 0, len(grid) - 1)
    lo = np.clip(hi - 1, 0, len(grid) - 1)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    pick = np.where(d_lo < d_hi, lo, np.where(d_hi < d_lo, hi, np.where(lo % 2 == 0, lo, hi)))
    return (np.sign(v) * grid[pick]).astype(np.float32)


def ref_nvfp4(x):
    """``(out, q, scale, err4, err6)`` per the four-over-six MAE contract."""
    f32 = np.float32
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    amax = np.abs(blocks).max(axis=(-2, -1), keepdims=True)
    bmax = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        encode = f32(1536) / amax
        decode = f32(1) / encode
        hp6 = (bmax / f32(6)) * encode
        hp6 = np.where(np.isnan(hp6), f32(0), hp6).astype(np.float32)
        cands = []
        for hp in (hp6, (hp6 * f32(1.5)).astype(np.float32)):
            s = _round_to_grid(hp, E4M3_GRID)
            scaled = np.where(s != 0, blocks * (f32(1) / (decode * s)), f32(0)).astype(np.float32)
            q = _round_to_grid(scaled, E2M1_GRID)
            d = (((q * s) * amax) / f32(1536)).astype(np.float32)
            err = np.abs(d.astype(np.float64) - blocks.astype(np.float64)).sum(-1, keepdims=True)
            cands.append((q, s, d, err))
    (q6, s6, d6, e6), (q4, s4, d4, e4) = cands
    pick4 = e4 < e6
    out = np.where(pick4, d4, d6).reshape(x.shape)
    return (out, np.where(pick4, q4, q6), np.where(pick4, s4, s6), e4, e6)


def ref_mxfp8(x):
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    hp = (np.abs(blocks).max(-1, keepdims=True) / np.float32(448)).astype(np.float32)
    bits = hp.view(np.uint32)
    exponent = (bits >> 23) & 0xFF
    exponent = np.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1).astype(np.uint32)
    scale = (exponent << 23).view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(exponent != 0, blocks / scale, np.float32(0)).astype(np.float32)
    return (_round_to_grid(scaled, E4M3_GRID) * scale).reshape(x.shape)


# ----- inputs -----


def _random(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * np.exp(rng.standard_normal(shape))).astype(np.float32)


def _edge_rows():
    rows = []
    rows.append(np.zeros(64, np.float32))  # amax = 0 token
    r = _random(64, 1) * 0.01
    r[37] = 80.0  # one large outlier in block 2
    rows.append(r)
    rows.append(-np.abs(_random(64, 2)))  # all negative
    # Ties at E2M1 midpoints: amax = 6 and block max 6 make s6 = 256 and the
    # scaled values the inputs themselves.
    rows.append(np.array([6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75,
                          -2.5, -3.5, -5, 0] * 4, np.float32))
    # Subnormal E4M3 block scales: tiny blocks next to a large one (values
    # can then scale past 6 and must saturate).
    r = _random(64, 3) * 1e-4
    r[:16] = _random(16, 4) * 1e3
    rows.append(r)
    r = np.zeros(64, np.float32)
    r[5] = -3.0  # one nonzero value in the row
    rows.append(r)
    return np.stack(rows)


def test_nvfp4_tiny_amax_token_quantizes_to_zero_not_nan():
    # 0 < amax < 1536 / FLT_MAX overflows the encode scale.
    x = np.zeros((2, 32), np.float32)
    x[0, 3] = 1e-37
    x[1, 3] = 1e-30  # small but representable: quantized normally
    got = np.array(aq.fake_quant_nvfp4_act(mx.array(x)))
    assert np.isfinite(got).all()
    assert (got[0] == 0).all()
    assert np.array_equal(got[1:], ref_nvfp4(x[1:])[0]) and got[1, 3] != 0


def _assert_nvfp4_matches(x):
    got = np.array(aq.fake_quant_nvfp4_act(mx.array(x)))
    out, q, s, e4, e6 = ref_nvfp4(x)
    # The block error sums may differ in the last bit by summation order;
    # only a block whose two candidate errors are that close may differ, and
    # then it must equal one of the candidates.
    close = (np.abs(e4 - e6) <= 1e-6 * np.maximum(e4, e6)).squeeze(-1)
    blocks_got = got.reshape(*got.shape[:-1], -1, 16)
    blocks_ref = out.reshape(blocks_got.shape)
    same = (blocks_got == blocks_ref).all(-1)
    assert (same | close).all()
    return got, out, q, s


# ----- quantizer tests -----


@pytest.mark.parametrize("shape,seed", [((3, 5, 128), 0), ((7, 2304), 1), ((2, 8, 1, 1152), 2)])
def test_nvfp4_matches_numpy_reference_random(shape, seed):
    x = _random(shape, seed)
    got, out, q, s = _assert_nvfp4_matches(x)
    assert np.array_equal(got, out)
    assert np.isin(np.abs(q), E2M1_GRID).all()
    assert np.isin(s, E4M3_GRID).all() and s.max() <= 384


def test_nvfp4_edge_cases_match_reference():
    x = _edge_rows()
    got, out, q, s = _assert_nvfp4_matches(x)
    assert np.array_equal(got, out)
    assert (got[0] == 0).all()  # amax = 0 token
    assert np.isfinite(got).all()
    assert (np.sign(got[2]) <= 0).all()  # negatives stay non-positive
    # The single value of the last row is reproduced exactly (6 * s / 6).
    assert got[5, 5] == -3.0 and np.count_nonzero(got[5]) == 1
    # Subnormal-scale blocks saturate instead of leaving the E2M1 range.
    assert np.abs(q).max() <= 6


def test_e2m1_rounds_midpoints_to_even_and_saturates():
    v = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 7.0, 100.0, -0.75, -5.0, 0.0])
    assert aq._e2m1(v).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0,
                                    -1.0, -4.0, 0.0]


def test_midpoint_row_scale6_candidate_is_the_grid_rounding():
    x = mx.array(_edge_rows()[3:4])
    _, d6, _, _ = aq._nvfp4_candidates(x)
    expected = _round_to_grid(np.array(x), E2M1_GRID)
    assert np.array_equal(np.array(d6).reshape(expected.shape), expected)


def test_four_over_six_picks_lower_mae_candidate():
    x = _random((4, 9, 256), 7)
    blocks, d6, d4, pick4 = (np.array(a) for a in aq._nvfp4_candidates(mx.array(x)))
    e6 = np.abs(d6.astype(np.float64) - blocks).sum(-1, keepdims=True)
    e4 = np.abs(d4.astype(np.float64) - blocks).sum(-1, keepdims=True)
    chosen = np.where(pick4, e4, e6)
    assert (chosen <= np.minimum(e4, e6) * (1 + 1e-6) + 1e-30).all()
    assert 0.02 < pick4.mean() < 0.98  # both candidates are used on random data
    got = np.array(aq.fake_quant_nvfp4_act(mx.array(x))).reshape(blocks.shape)
    assert np.array_equal(got, np.where(pick4, d4, d6))


def test_four_over_six_known_blocks():
    # Block 0: [6, 5 x 15]: scale 6 rounds 5 -> 4 (error 1 each); scale 4
    # gives 6 -> 4 * 1.5 and 5 -> 3.33 -> 3 * 1.5 = 4.5 (error 0.5): 4 wins.
    # Block 1: [6, 3 x 15]: scale 6 is exact: 6 is kept.
    x = np.array([[6.0] + [5.0] * 15 + [6.0] + [3.0] * 15], np.float32)
    _, _, _, pick4 = aq._nvfp4_candidates(mx.array(x))
    assert np.array(pick4).reshape(-1).tolist() == [True, False]
    got = np.array(aq.fake_quant_nvfp4_act(mx.array(x)))[0]
    assert got[:16].tolist() == [6.0] + [4.5] * 15
    assert got[16:].tolist() == [6.0] + [3.0] * 15


def test_nvfp4_is_per_token():
    x = _random((2, 64), 9)
    both = np.array(aq.fake_quant_nvfp4_act(mx.array(x)))
    for row in range(2):
        alone = np.array(aq.fake_quant_nvfp4_act(mx.array(x[row:row + 1])))
        assert np.array_equal(both[row:row + 1], alone)
    scaled = x.copy()
    scaled[1] *= 1000  # another token's amax must not change this one
    assert np.array_equal(np.array(aq.fake_quant_nvfp4_act(mx.array(scaled)))[0], both[0])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_output_dtype_follows_input(dtype):
    x = mx.array(_random((3, 2304), 4)).astype(dtype)
    assert aq.fake_quant_nvfp4_act(x).dtype == dtype
    assert aq.fake_quant_mxfp8_act(x).dtype == dtype
    if dtype == mx.bfloat16:
        ref = ref_nvfp4(np.array(x.astype(mx.float32)))[0]
        got = aq.fake_quant_nvfp4_act(x)
        assert mx.array_equal(got, mx.array(ref).astype(mx.bfloat16))


@pytest.mark.parametrize("seed", [0, 1])
def test_mxfp8_matches_numpy_reference_and_mx_quantize(seed):
    x = np.concatenate([_random((5, 256), seed), _edge_rows()[:, :64].repeat(4, axis=1)])
    got = np.array(aq.fake_quant_mxfp8_act(mx.array(x)))
    assert np.array_equal(got, ref_mxfp8(x))
    assert (got[5] == 0).all()
    w, s = mx.quantize(mx.array(x), group_size=32, bits=8, mode="mxfp8")
    stock = np.array(mx.dequantize(w, s, group_size=32, bits=8, mode="mxfp8").astype(mx.float32))
    assert np.array_equal(got, stock)
    # E4M3 times a power of two is exact in BF16.
    xb = mx.array(x).astype(mx.bfloat16)
    yb = aq.fake_quant_mxfp8_act(xb)
    assert mx.array_equal(yb.astype(mx.float32), aq.fake_quant_mxfp8_act(xb.astype(mx.float32)))


def test_quantizers_refuse_partial_blocks():
    with pytest.raises(ValueError, match="multiple of block 16"):
        aq.fake_quant_nvfp4_act(mx.zeros((2, 24)))
    with pytest.raises(ValueError, match="multiple of block 32"):
        aq.fake_quant_mxfp8_act(mx.zeros((2, 48)))


# ----- the option -----


def test_policy_default_off_and_strict():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, validate_policy

    assert KERNEL_OPTIONS["inkling2_moe_act_quant"] == "moe_act_quant"
    assert validate_policy(None)[0]["moe_act_quant"] is False
    assert validate_policy({"inkling2_moe_act_quant": True})[0]["moe_act_quant"] is True
    for bad in (1, None, "true", 0.0):
        with pytest.raises(TypeError, match="inkling2_moe_act_quant must be a boolean"):
            validate_policy({"inkling2_moe_act_quant": bad})


def test_feature_grid_lists_the_option():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "scripts/build_model_feature_grid.py").read_text()
    assert '"inkling2_moe_act_quant"' in text


def test_numerics_contract_identity():
    from mlx2.adapters.inkling2 import NUMERICS, NUMERICS_W4A4_FAKE, Inkling2Adapter, validate_policy

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.kernel_policy = validate_policy(None)[0]
    off = adapter.execution_numerics_contract()
    assert off["operation"] == NUMERICS == "inkling2-native-w4a16-v1"
    assert "inkling2/act_quant.py" not in off["source_sha256"]
    adapter.kernel_policy = validate_policy({"inkling2_moe_act_quant": True})[0]
    on = adapter.execution_numerics_contract()
    assert on["operation"] == NUMERICS_W4A4_FAKE == "inkling2-native-w4a4-fake-v1"
    assert len(on["source_sha256"]["inkling2/act_quant.py"]) == 64
    assert on["kernel_policy"]["moe_act_quant"] is True


def test_refuses_experts_without_a_declared_format():
    model = _tiny_model()  # dense experts
    model.configure_kernels()
    before = dict(model.kernel_policy)
    with pytest.raises(ValueError, match="needs NVFP4 or MXFP8 experts"):
        model.configure_kernels(moe_act_quant=True)
    assert model.kernel_policy == before
    assert all(layer.mlp.__dict__.get("_act_quant") is None for layer in model.model.layers)


def _quantized_model(dtype):
    model = _tiny_model()
    _quantize_experts(model)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    return model


# 48 visits per forward: layers 0-19 (NVFP4) are visited 6 + 12 * 3 + 2 = 44
# times, layers 20-23 (MXFP8) 4 times.
def _expected_calls(path, forwards):
    return {f"{path}:mxfp8:w13": 4 * forwards, f"{path}:mxfp8:w2": 4 * forwards,
            f"{path}:nvfp4:w13": 44 * forwards, f"{path}:nvfp4:w2": 44 * forwards}


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_off_is_bitwise_reference_and_on_engages_both_paths_identically(dtype):
    model = _quantized_model(dtype)
    tokens, plan = _plan_case()
    model.configure_kernels()
    base = _run(model, tokens, plan)
    receipt = model.kernel_receipt()["moe_act_quant"]
    assert receipt == {"calls": {}, "layers": {"nvfp4": 0, "mxfp8": 0}}

    model.configure_kernels(moe_act_quant=True)
    assert model.kernel_policy["moe_act_quant"] is True
    reference = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    assert receipt["moe_act_quant"]["layers"] == {"nvfp4": 20, "mxfp8": 4}
    assert receipt["moe_act_quant"]["calls"] == _expected_calls("reference", len(plan))
    assert receipt["moe_fused_calls"] == 0
    assert not mx.array_equal(reference, base)  # it changes the numerics

    model.configure_kernels(moe_fused=True, moe_act_quant=True)
    fused = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    assert receipt["moe_fused_calls"] == 48 * len(plan)
    calls = receipt["moe_act_quant"]["calls"]
    assert {k: v for k, v in calls.items() if k.startswith("fused:")} == _expected_calls(
        "fused", len(plan))
    assert mx.array_equal(fused, reference)

    # Off again (fused and reference): bitwise the untouched reference.
    model.configure_kernels(moe_fused=True)
    assert mx.array_equal(_run(model, tokens, plan), base)
    model.configure_kernels()
    assert mx.array_equal(_run(model, tokens, plan), base)
    assert all(layer.mlp.switch_mlp.activation.__dict__.get("_act_quant") is None
               for layer in model.model.layers)


@pytest.mark.parametrize("layer_index,fmt", [(0, "nvfp4"), (21, "mxfp8")])
def test_router_reads_unquantized_input_and_experts_the_quantized_one(layer_index, fmt):
    from mlx2.runtime.models.inkling2.language import _combine_routes, _fp32, _route

    model = _quantized_model(mx.float32)
    model.configure_kernels(moe_act_quant=True)
    moe = model.model.layers[layer_index].mlp
    x = mx.array(_random((2, 5, 128), 11))
    got = moe(x)
    indices, weights = _route(x, _fp32(moe, "gate_weight"), moe.e_score_correction_bias,
                              moe.top_k, moe.route_scale)
    switch = moe.switch_mlp
    xq = mx.expand_dims(aq.apply(fmt, x), (-2, -3))
    up = switch.up_proj(xq, indices)
    gate = switch.gate_proj(xq, indices)
    from mlx2.runtime.models.inkling2.language import _softcapped_swiglu

    hidden = aq.apply(fmt, _softcapped_swiglu(up, gate, switch.activation.gate_cap,
                                              switch.activation.up_cap))
    expected = _combine_routes(switch.down_proj(hidden, indices).squeeze(-2), weights)
    assert mx.array_equal(got, expected)
