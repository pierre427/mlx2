"""Q8 in-place W8A8 prefill (port of jundot/omlx #4350) inside int8 prefill.

CPU tests (the suite default) never launch a Metal kernel: policy and
revision binding, CLI composition, eligibility, byte accounting, receipts,
Prometheus, fail-closed install, and routing with the two kernels replaced by
CPU mirrors (which also proves a Q8 module never reaches the requant path).
GPU tests need an M5-class GPU and ``MLX2_TEST_INT8_NAX=1`` (or
``MLX2_TEST_OPTIONAL_METAL=1``); they check Stage A and the GEMM against an
independent float64 reference on random packed bytes.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime import int8_prefill as ip

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _qlinear(k, n, *, bits=8, group_size=64, bias=False, dtype=mx.bfloat16):
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.float32)
    if bias:
        layer.bias = (mx.random.normal((n,)) * 0.02).astype(dtype)
    q = nn.QuantizedLinear.from_linear(layer, group_size=group_size, bits=bits)
    q.scales = q.scales.astype(dtype)
    q.biases = q.biases.astype(dtype)
    return q


class _MLP(nn.Module):
    def __init__(self, hidden, inter, bits=8, group_size=64, dtype=mx.bfloat16):
        super().__init__()
        kw = dict(bits=bits, group_size=group_size, dtype=dtype)
        self.gate_proj = _qlinear(hidden, inter, **kw)
        self.up_proj = _qlinear(hidden, inter, **kw)
        self.down_proj = _qlinear(inter, hidden, **kw)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, hidden, bits):
        super().__init__()
        self.q_proj = _qlinear(hidden, hidden, bits=bits)
        self.o_proj = _qlinear(hidden, hidden, bits=bits)


class _Layer(nn.Module):
    def __init__(self, hidden, inter, mlp_bits=8, attn_bits=4):
        super().__init__()
        self.self_attn = _Attn(hidden, attn_bits)
        self.mlp = _MLP(hidden, inter, bits=mlp_bits)


class _Model(nn.Module):
    def __init__(self, hidden=256, inter=512, mlp_bits=8, attn_bits=4, layers=2):
        super().__init__()
        self.layers = [_Layer(hidden, inter, mlp_bits, attn_bits) for _ in range(layers)]


@pytest.fixture
def fake_device(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")


Q8 = {"enabled": True, "scope": "mlp", "q8_inplace": True, "act_scale": "group64"}


# --------------------------------------------------------------------------
# CPU mirrors of the two kernels (same math, plain MLX ops)
# --------------------------------------------------------------------------


def _mirror_stage_a(x, act_scale="group64", m_dim=None):
    m, k = x.shape
    g = k // 64
    xf = x.astype(mx.float32)
    if act_scale == "per_row":
        amax = mx.max(mx.abs(xf), axis=1)
        inv = mx.where(amax > 0, 127.0 / amax, 0.0)
        q = mx.clip(mx.round(xf * inv[:, None]), -127, 127)
        sa = mx.where(amax > 0, amax / 127.0, 0.0)
    else:
        xg = xf.reshape(m, g, 64)
        amax = mx.max(mx.abs(xg), axis=2)
        inv = mx.where(amax > 0, 127.0 / amax, 0.0)
        q = mx.clip(mx.round(xg * inv[..., None]), -127, 127).reshape(m, k)
        sa = mx.where(amax > 0, amax / 127.0, 0.0).T
    ra = q.reshape(m, g, 64).sum(axis=2).astype(mx.int16).T
    return q.astype(mx.int8), sa, ra


def _mirror_gemm(qa, sa, ra, weight, st, bt, *, act_scale, out_dtype=None, bias=None,
                 tile=None, m_dim=None):
    m, k = qa.shape
    n = weight.shape[0]
    g = k // 64
    codes = mx.view(weight, mx.uint8).reshape(n, g, 64).astype(mx.float32) - 128.0
    a = qa.astype(mx.float32).reshape(m, g, 64)
    acc = mx.einsum("mgk,ngk->mgn", a, codes)
    r = ra.T.astype(mx.float32)[..., None]  # [m, g, 1]
    part = st.astype(mx.float32)[None] * (acc + 128.0 * r) + bt.astype(mx.float32)[None] * r
    if act_scale == "per_row":
        y = part.sum(axis=1) * sa[:, None]
    else:
        y = (part * sa.T[..., None]).sum(axis=1)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(out_dtype or st.dtype)


@pytest.fixture
def mirrored(monkeypatch, fake_device):
    calls = {"stage_a": 0, "gemm": 0}

    def stage_a(*args, **kwargs):
        calls["stage_a"] += 1
        return _mirror_stage_a(*args, **kwargs)

    def gemm(*args, **kwargs):
        calls["gemm"] += 1
        return _mirror_gemm(*args, **kwargs)

    def no_requant(*args, **kwargs):
        raise AssertionError("a Q8 in-place module reached the requant path")

    monkeypatch.setattr(ip, "q8_stage_a", stage_a)
    monkeypatch.setattr(ip, "q8_gemm", gemm)
    monkeypatch.setattr(ip, "_requant_packed", no_requant)
    return calls


# --------------------------------------------------------------------------
# policy / revision / CLI
# --------------------------------------------------------------------------


def test_policy_defaults_and_parsing():
    off = ip.Int8PrefillPolicy()
    assert off.q8_inplace is False and off.act_scale == "group64"
    q8 = ip.Int8PrefillPolicy.from_value(Q8)
    assert (q8.enabled, q8.q8_inplace, q8.act_scale) == (True, True, "group64")
    row = ip.Int8PrefillPolicy.from_value({**Q8, "act_scale": "per_row"})
    assert row.act_scale == "per_row"


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "q8_inplace": 1},
        {"enabled": True, "q8_inplace": True, "act_scale": "per_tensor"},
        {"enabled": False, "q8_inplace": True},
        {"q8_inplace": True},
    ],
)
def test_policy_rejects_invalid_q8_values(value):
    with pytest.raises(ValueError):
        ip.Int8PrefillPolicy.from_value(value)


def test_off_mode_records_and_revision_are_unchanged():
    base = ip.Int8PrefillPolicy.from_value("mlp")
    assert set(base.as_dict()) == {"enabled", "scope", "row_threshold", "cache", "ttl_s"}
    # act_scale has no effect without q8_inplace, so it is not numerics.
    assert ip.Int8PrefillPolicy(enabled=True, act_scale="per_row").revision == base.revision
    q8 = ip.Int8PrefillPolicy.from_value(Q8)
    assert q8.as_dict()["q8_inplace"] is True and q8.as_dict()["act_scale"] == "group64"


def test_q8_mode_and_scaling_bind_revision_and_apc_namespace():
    base = ip.Int8PrefillPolicy.from_value("mlp")
    g64 = ip.Int8PrefillPolicy.from_value(Q8)
    row = ip.Int8PrefillPolicy.from_value({**Q8, "act_scale": "per_row"})
    revisions = {base.revision, g64.revision, row.revision}
    assert len(revisions) == 3
    namespaces = {
        ip.apc_semantic_fingerprint("text-token-v1", p) for p in (None, base, g64, row)
    }
    assert len(namespaces) == 4
    # Memory lifetime is still not numerics under q8.
    assert ip.Int8PrefillPolicy.from_value({**Q8, "cache": "ttl"}).revision == g64.revision


def test_cli_value_composes_scope_and_q8_mode():
    assert ip.cli_value("off") == "off"
    assert ip.cli_value("mlp", "off") == "mlp"
    assert ip.cli_value("all", "per_row") == {
        "enabled": True, "scope": "all", "q8_inplace": True, "act_scale": "per_row",
    }
    with pytest.raises(ValueError, match="requires --int8-prefill"):
        ip.cli_value("off", "group64")
    with pytest.raises(ValueError):
        ip.cli_value("mlp", "per_tensor")


def test_server_flag_defaults_off_and_feeds_engine_kwargs():
    from mlx2.server import build_parser

    parser = build_parser()
    assert parser.parse_args(["--model", "m"]).int8_prefill_q8 == "off"
    args = parser.parse_args(
        ["--model", "m", "--int8-prefill", "mlp", "--int8-prefill-q8", "group64"]
    )
    value = ip.cli_value(args.int8_prefill, args.int8_prefill_q8)
    assert ip.Int8PrefillPolicy.from_value(value) == ip.Int8PrefillPolicy.from_value(Q8)
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", "--int8-prefill-q8", "on"])


# --------------------------------------------------------------------------
# eligibility and install
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bits,group_size,dtype,ok",
    [
        (8, 64, mx.bfloat16, True),
        (8, 64, mx.float16, True),
        (8, 128, mx.bfloat16, False),
        (8, 32, mx.bfloat16, False),
        (6, 64, mx.bfloat16, False),
        (4, 64, mx.bfloat16, False),
        (8, 64, mx.float32, False),
    ],
)
def test_q8_eligibility_matrix(bits, group_size, dtype, ok):
    layer = _qlinear(512, 256, bits=bits, group_size=group_size, dtype=dtype)
    spec, reason = ip.module_spec("mlp.up_proj", layer)
    assert spec is not None, reason
    assert ip.q8_inplace_eligible(spec, layer)[0] is ok


def test_q8_eligibility_refuses_mixed_metadata_dtypes():
    layer = _qlinear(512, 256)
    layer.biases = layer.biases.astype(mx.float16)
    spec, _ = ip.module_spec("mlp.up_proj", layer)
    assert ip.q8_inplace_eligible(spec, layer) == (False, "q8 scales and biases differ in dtype")


def test_q8_install_binds_only_q8_modules_and_accounts_bytes(fake_device):
    model = _Model(mlp_bits=8, attn_bits=4)
    handle = ip.apply(model, {**Q8, "scope": "all"})
    try:
        status = handle.status()
        assert status["module_kinds"] == {"q8_inplace": 6, "q4": 4}
        assert handle.q8_module_count() == 6
        q8_bounds = [b for b in handle._bound.values() if b.q8]
        assert {b.spec.bits for b in q8_bounds} == {8}
        # Requant copies only for the 4-bit modules (auto = per call).
        q4 = [b for b in handle._bound.values() if not b.q8]
        assert handle.transient_weight_bytes_max() == max(b.spec.n * (b.spec.k + 4) for b in q4)
        assert handle.resident_estimate_bytes() == 0
        expected_meta = sum(2 * b.module["scales"].size * 2 for b in q8_bounds)
        assert handle.metadata_copy_bytes() == expected_meta
        summary = status["q8_inplace"]
        assert summary["modules"] == 6 and summary["weight_copy_bytes"] == 0
        assert summary["metadata_copy_bytes"] == expected_meta
        assert summary["act_scale"] == "group64"
        assert summary["kernel_revision"] == ip.Q8_KERNEL_REVISION
        assert summary["declined"] == 4
        receipt = handle.receipt()
        assert receipt["revision"] == ip.Int8PrefillPolicy.from_value({**Q8, "scope": "all"}).revision
        assert receipt["q8_inplace"]["modules"] == 6
        for key in ("q8_calls", "q8_rows", "q8_stage_a", "q8_stage_a_reuse", "q8_meta_builds"):
            assert status["counts"][key] == 0
    finally:
        ip.remove(handle)


def test_all_q8_model_has_zero_weight_copy_bytes(fake_device):
    model = _Model(mlp_bits=8, attn_bits=8)
    handle = ip.apply(model, {**Q8, "scope": "all"})
    try:
        assert handle.weight_copy_bytes() == 0
        assert handle.metadata_copy_bytes() > 0
        assert handle.receipt()["weight_copy_bytes"] == 0
    finally:
        ip.remove(handle)


def test_q8_cache_none_reports_transient_metadata(fake_device):
    model = _Model(mlp_bits=8, attn_bits=8)
    handle = ip.apply(model, {**Q8, "cache": "none"})
    try:
        largest = max(b.meta_size() for b in handle._bound.values())
        assert handle.metadata_copy_bytes() == largest
        assert handle.warmup() == 0
    finally:
        ip.remove(handle)


def test_q8_inplace_without_any_q8_module_fails_closed(fake_device):
    model = _Model(mlp_bits=6, attn_bits=4)
    with pytest.raises(ip.Int8PrefillError, match="q8_inplace"):
        ip.apply(model, Q8)
    # Nothing was installed.
    assert not any(ip._STATE_ATTR in m.__dict__ for _, m in model.named_modules())


def test_off_mode_status_has_no_q8_surface(fake_device):
    model = _Model(mlp_bits=8, attn_bits=4)
    handle = ip.apply(model, "mlp")
    try:
        status = handle.status()
        assert "q8_inplace" not in status and "q8_calls" not in status["counts"]
        assert status["module_kinds"] == {"q8": 6}
        assert "q8_inplace" not in handle.receipt()
        assert not any(b.q8 for b in handle._bound.values())
    finally:
        ip.remove(handle)


# --------------------------------------------------------------------------
# routing (CPU mirrors in place of the kernels)
# --------------------------------------------------------------------------


def test_q8_routing_counts_shares_stage_a_and_never_requantizes(mirrored):
    model = _Model(mlp_bits=8, attn_bits=4, layers=1)
    mlp = model.layers[0].mlp
    x = (mx.random.normal((2, 300, 256)) * 0.5).astype(mx.bfloat16)  # 600 rows
    reference = mlp(x)
    handle = ip.apply(model, Q8)
    try:
        got = mlp(x)
        mx.eval(got, reference)
        assert got.shape == reference.shape and got.dtype == mx.bfloat16
        rel = (mx.linalg.norm((got - reference).astype(mx.float32))
               / mx.linalg.norm(reference.astype(mx.float32))).item()
        # Routing sanity only (three W8A8 projections through silu * up);
        # the GPU tests check the numerics against an exact reference.
        assert rel < 0.08, rel
        counts = handle.counts
        assert counts["q8_calls"] == 3 and counts["engaged_calls"] == 3
        assert counts["q8_rows"] == 1800 and counts["engaged_rows"] == 1800
        # gate/up share x; down quantizes h.
        assert counts["q8_stage_a"] == 2 and counts["q8_stage_a_reuse"] == 1
        assert mirrored == {"stage_a": 2, "gemm": 3}
        assert counts["weight_builds"] == 0 and handle.weight_bytes() == 0
        # auto cache keeps the metadata resident, built once per module.
        assert counts["q8_meta_builds"] == 3
        mx.eval(mlp(x * 1))
        assert counts["q8_meta_builds"] == 3 and counts["q8_calls"] == 6
        assert handle.metadata_bytes() == handle.metadata_copy_bytes()
        assert handle.evict_weights() == 3 and handle.metadata_bytes() == 0
    finally:
        ip.remove(handle)


def test_q8_routing_per_row_scaling_uses_its_stage_a(mirrored, monkeypatch):
    seen = []
    real = ip.q8_stage_a

    def spy(x, act_scale="group64", m_dim=None):
        seen.append(act_scale)
        return real(x, act_scale, m_dim=m_dim)

    monkeypatch.setattr(ip, "q8_stage_a", spy)
    model = _Model(mlp_bits=8, attn_bits=4, layers=1)
    handle = ip.apply(model, {**Q8, "act_scale": "per_row"})
    try:
        mx.eval(model.layers[0].mlp(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        assert seen == ["per_row", "per_row"]
        assert handle.counts["q8_calls"] == 3
    finally:
        ip.remove(handle)


def test_q8_sub_threshold_and_dtype_mismatch_stay_stock(mirrored):
    model = _Model(mlp_bits=8, attn_bits=4, layers=1)
    mlp = model.layers[0].mlp
    small = mx.random.normal((16, 256)).astype(mx.bfloat16)
    wide32 = mx.random.normal((600, 256)).astype(mx.float32)
    ref_small, ref32 = mlp(small), mlp(wide32)
    handle = ip.apply(model, Q8)
    try:
        assert mx.array_equal(mlp(small), ref_small).item()
        assert mx.array_equal(mlp(wide32), ref32).item()
        counts = handle.counts
        assert counts["q8_calls"] == 0 and mirrored["gemm"] == 0
        assert counts["fallback_rows"] == 3 and counts["fallback_dtype"] == 3
    finally:
        ip.remove(handle)


def test_prometheus_exports_q8_counters(mirrored):
    from mlx2.prometheus import PrometheusBuilder, _add_int8_prefill

    class Engine:
        int8_prefill_policy = ip.Int8PrefillPolicy.from_value(Q8)
        int8_prefill_handle = None

    builder = PrometheusBuilder()
    _add_int8_prefill(builder, Engine())
    assert "mlx2_int8_prefill_q8_inplace_calls_total 0" in builder.render()
    engine = Engine()
    model = _Model(mlp_bits=8, attn_bits=4, layers=1)
    engine.int8_prefill_handle = ip.apply(model, Engine.int8_prefill_policy)
    try:
        mx.eval(model.layers[0].mlp(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        builder = PrometheusBuilder()
        _add_int8_prefill(builder, engine)
        text = builder.render()
        assert "mlx2_int8_prefill_q8_inplace_calls_total 3" in text
        assert "mlx2_int8_prefill_q8_inplace_rows_total 1800" in text
        assert "mlx2_int8_prefill_q8_inplace_modules 3" in text
        assert 'mlx2_int8_prefill_calls_total{outcome="engaged"} 3' in text
    finally:
        ip.remove(engine.int8_prefill_handle)


def test_bind_for_serving_records_q8_settings(fake_device):
    class Adapter:
        model = _Model(mlp_bits=8, attn_bits=4)

        @staticmethod
        def int8_prefill_supported():
            return ("mlp",)

    handle, settings = ip.bind_for_serving(
        Adapter(), Q8, max_lanes=4, config={}, speculation="ordinary"
    )
    try:
        assert settings["q8_inplace"] is True and settings["act_scale"] == "group64"
        assert settings["q8_modules"] == 6
        assert settings["weight_copy_bytes"] == 0
        assert settings["metadata_copy_bytes"] == handle.metadata_copy_bytes() > 0
        assert settings["revision"] == ip.Int8PrefillPolicy.from_value(Q8).revision
        # warmup built the resident metadata
        assert handle.metadata_bytes() == settings["metadata_copy_bytes"]
    finally:
        ip.remove(handle)


def test_qwen38_adapter_declares_int8_scopes_but_moe_family_does_not():
    from mlx2.adapters.qwen35_122b import Qwen35122BA10BAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert ip.adapter_scopes(Qwen3827BAdapter) == {"mlp", "all"}
    assert ip.adapter_scopes(Qwen3635BA3BAdapter) == frozenset()
    assert ip.adapter_scopes(Qwen35122BA10BAdapter) == frozenset()


def test_serving_engine_records_q8_route(monkeypatch, fake_device):
    from test_int8_prefill import _serving_engine, _serving_model

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model = _serving_model(hidden=256, inter=512)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0,
    )
    mx.eval(model.parameters())  # the worker thread has no CPU stream
    engine = _serving_engine(model, ("mlp",), int8_prefill=ip.cli_value("mlp", "group64"))
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        recorded = status["settings"]["int8_prefill"]
        assert recorded["q8_inplace"] is True and recorded["act_scale"] == "group64"
        assert recorded["q8_modules"] > 0 and recorded["weight_copy_bytes"] == 0
        assert recorded["revision"] == ip.Int8PrefillPolicy.from_value(Q8).revision
        assert status["int8_prefill"]["q8_inplace"]["modules"] == recorded["q8_modules"]
    finally:
        engine.close()


# --------------------------------------------------------------------------
# GPU numerics (M5-class, explicit opt-in)
# --------------------------------------------------------------------------

_GPU_OPT_IN = (
    os.environ.get("MLX2_TEST_INT8_NAX") == "1"
    or os.environ.get("MLX2_TEST_OPTIONAL_METAL") == "1"
)


@pytest.fixture
def gpu():
    if not _GPU_OPT_IN:
        pytest.skip("set MLX2_TEST_INT8_NAX=1 to run M5 NAX numerics")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        ok, reason = ip.device_support()
        if not ok:
            pytest.skip(f"int8 NAX unsupported here: {reason}")
        yield
    finally:
        mx.set_default_device(previous)


def _random_q8(n, k, dtype, seed):
    """Packed Q8 weight from random bytes (edge codes forced in every row)
    plus random metadata; also returns the float64 dequantized reference."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    codes[:, :4] = np.array([0, 127, 128, 255], dtype=np.uint8)
    codes[0, :] = 255
    codes[1, :] = 0
    g = k // 64
    scales = (rng.standard_normal((n, g)) * 0.01).astype(np.float32)
    biases = (rng.standard_normal((n, g)) * 0.5).astype(np.float32)
    s = mx.array(scales).astype(dtype)
    b = mx.array(biases).astype(dtype)
    packed = mx.array(codes.view(np.uint32).reshape(n, k // 4))
    s64 = np.repeat(np.array(s.astype(mx.float32)).astype(np.float64), 64, axis=1)
    b64 = np.repeat(np.array(b.astype(mx.float32)).astype(np.float64), 64, axis=1)
    w64 = codes.astype(np.float64) * s64 + b64
    return packed, s, b, w64


def _dequant_activation(qa, sa, act_scale):
    q = np.array(qa).astype(np.float64)
    s = np.array(sa).astype(np.float64)
    if act_scale == "per_row":
        return q * s[:, None]
    return q * np.repeat(s.T, 64, axis=1)


def test_gpu_independent_unpack_matches_mlx_dequantize(gpu):
    packed, s, b, w64 = _random_q8(256, 512, mx.float16, 0)
    mx_w = mx.dequantize(packed, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=8)
    assert np.array_equal(np.array(mx_w).astype(np.float64), w64.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("m", [1, 7, 130])
def test_gpu_stage_a_codes_sums_and_scales(gpu, act_scale, dtype, m):
    k = 640
    rng = np.random.default_rng(m)
    xn = (rng.standard_normal((m, k)) * 2.0).astype(np.float32)
    xn[0, :64] = 0.0  # an all-zero group
    if m > 1:
        xn[1, :] = 0.0  # an all-zero row
        xn[-1, 5] = 40.0  # an outlier
    x = mx.array(xn).astype(dtype)
    qa, sa, ra = ip.q8_stage_a(x, act_scale)
    mx.eval(qa, sa, ra)
    xf = np.array(x.astype(mx.float32)).astype(np.float32)
    g = k // 64
    if act_scale == "per_row":
        amax = np.abs(xf).max(axis=1)
        inv = np.where(amax > 0, np.float32(127.0) / np.maximum(amax, 1e-30), 0).astype(np.float32)
        ref = np.clip(np.rint(xf * inv[:, None]), -127, 127)
        ref_s = np.where(amax > 0, amax / np.float32(127.0), 0)
        assert np.allclose(np.array(sa), ref_s, rtol=1e-6, atol=0)
        assert sa.shape == (m,)
    else:
        xg = xf.reshape(m, g, 64)
        amax = np.abs(xg).max(axis=2)
        inv = np.where(amax > 0, np.float32(127.0) / np.maximum(amax, 1e-30), 0).astype(np.float32)
        ref = np.clip(np.rint(xg * inv[..., None]), -127, 127).reshape(m, k)
        ref_s = np.where(amax > 0, amax / np.float32(127.0), 0)
        assert sa.shape == (g, m)
        assert np.allclose(np.array(sa).T, ref_s, rtol=1e-6, atol=0)
    got = np.array(qa).astype(np.int32)
    diff = np.abs(got - ref)
    # Codes agree with an independent numpy quantizer; an fp32 tie at a
    # rounding boundary may differ by one code.
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, (diff.max(), (diff > 0).mean())
    # Ra is the exact sum of the emitted codes, group-major.
    assert ra.shape == (g, m) and ra.dtype == mx.int16
    assert np.array_equal(np.array(ra).T, got.reshape(m, g, 64).sum(axis=2))
    if m > 1:
        assert not got[1].any() and not np.array(ra)[:, 1].any()


_GEMM_CASES = [
    # (m, k, n): M = 1, M below / above / not a multiple of every BM
    (1, 512, 256),
    (33, 1024, 256),
    (100, 256, 512),
    (257, 640, 256),
]


@pytest.mark.parametrize("tile", ip.Q8_TILES)
@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("m,k,n", _GEMM_CASES)
def test_gpu_q8_gemm_matches_independent_reference(gpu, tile, act_scale, dtype, m, k, n):
    packed, s, b, w64 = _random_q8(n, k, dtype, m + k + n)
    x = (mx.random.normal((m, k)) * 1.5).astype(dtype)
    qa, sa, ra = ip.q8_stage_a(x, act_scale)
    st, bt = ip.q8_metadata(s, b)
    y = ip.q8_gemm(qa, sa, ra, packed, st, bt, act_scale=act_scale,
                   out_dtype=mx.float32, tile=tile)
    yb = ip.q8_gemm(qa, sa, ra, packed, st, bt, act_scale=act_scale, tile=tile)
    mx.eval(y, yb)
    a64 = _dequant_activation(qa, sa, act_scale)
    ref = a64 @ w64.T
    scale = np.abs(a64) @ np.abs(w64).T
    err = np.abs(np.array(y).astype(np.float64) - ref)
    # fp32 accumulation over K/64 group partials: a few fp32 ulps of the
    # absolute dot product.
    assert (err / np.maximum(scale, 1e-30)).max() < 2e-6, (err / scale).max()
    # Metadata-dtype output is the fp32 result rounded once.
    assert yb.dtype == dtype
    assert mx.array_equal(yb, y.astype(dtype)).item()


@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
def test_gpu_q8_tiles_are_bit_identical(gpu, act_scale):
    m, k, n = 300, 1024, 512
    packed, s, b, _ = _random_q8(n, k, mx.bfloat16, 3)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    qa, sa, ra = ip.q8_stage_a(x, act_scale)
    st, bt = ip.q8_metadata(s, b)
    outs = [ip.q8_gemm(qa, sa, ra, packed, st, bt, act_scale=act_scale,
                       out_dtype=mx.float32, tile=t) for t in ip.Q8_TILES]
    mx.eval(*outs)
    for out in outs[1:]:
        assert mx.array_equal(out, outs[0]).item()


def test_gpu_q8_bias_epilogue(gpu):
    m, k, n = 64, 512, 256
    packed, s, b, w64 = _random_q8(n, k, mx.bfloat16, 9)
    bias = (mx.random.normal((n,)) * 0.1).astype(mx.bfloat16)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    qa, sa, ra = ip.q8_stage_a(x, "group64")
    st, bt = ip.q8_metadata(s, b)
    y0 = ip.q8_gemm(qa, sa, ra, packed, st, bt, out_dtype=mx.float32)
    y1 = ip.q8_gemm(qa, sa, ra, packed, st, bt, out_dtype=mx.float32, bias=bias)
    mx.eval(y0, y1)
    assert mx.array_equal(y1, y0 + bias.astype(mx.float32)).item()


class _One(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.mlp = layer

    def __call__(self, x):
        return self.mlp(x)


def _rel(y, ref):
    y, ref = y.astype(mx.float32), ref.astype(mx.float32)
    return (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()


@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_gpu_q8_module_route_vs_stock_and_counters(gpu, act_scale, dtype):
    mx.random.seed(5)
    model = _One(_qlinear(5120, 1024, dtype=dtype))
    x = mx.random.normal((2, 1024, 5120)).astype(dtype)
    decode = x[:, :8]
    stock, stock_decode = model(x), model(decode)
    mx.eval(stock, stock_decode)
    handle = ip.apply(model, {**Q8, "act_scale": act_scale})
    try:
        got, got_decode = model(x), model(decode)
        mx.eval(got, got_decode)
        rel = _rel(got, stock)
        print(f"q8 {act_scale} {dtype}: rel L2 vs stock quantized_matmul {rel:.5f}")
        assert got.dtype == stock.dtype and got.shape == stock.shape
        assert rel < 0.02, rel
        assert mx.array_equal(got_decode, stock_decode).item()
        counts = handle.counts
        assert counts["q8_calls"] == 1 and counts["q8_rows"] == 2048
        assert counts["weight_builds"] == 0 and handle.weight_bytes() == 0
        assert handle.metadata_bytes() == 2 * 1024 * 80 * 2
    finally:
        ip.remove(handle)


def test_gpu_q8_mlp_shares_stage_a(gpu):
    model = _Model(hidden=512, inter=1024, mlp_bits=8, attn_bits=4, layers=1)
    mlp = model.layers[0].mlp
    x = mx.random.normal((1024, 512)).astype(mx.bfloat16)
    stock = mlp(x)
    handle = ip.apply(model, Q8)
    try:
        got = mlp(x)
        mx.eval(got, stock)
        assert _rel(got, stock) < 0.03
        assert handle.counts["q8_stage_a"] == 2 and handle.counts["q8_stage_a_reuse"] == 1
        assert handle.counts["q8_calls"] == 3
    finally:
        ip.remove(handle)
