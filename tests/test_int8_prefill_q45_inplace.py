"""Q4/Q5 in-place W4A8/W5A8 prefill (port of the omlx Q4/Q5 A8 path).

CPU tests (the suite default) never launch a Metal kernel: policy and
revision binding, eligibility, byte accounting, receipts, Prometheus,
fail-closed install, ``inplace_only`` binding, and routing with the two
kernels replaced by CPU mirrors (which also proves a Q4/Q5 module never
reaches the requant path or the Q8 kernel).  GPU tests need an M5-class GPU
and ``MLX2_TEST_INT8_NAX=1`` (or ``MLX2_TEST_OPTIONAL_METAL=1``); they check
Stage A v8 and the GEMM against an independent float64 reference on random
packed codes in MLX's own bitstream layout.
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


def _qlinear(k, n, *, bits=4, group_size=64, bias=False, dtype=mx.bfloat16):
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.float32)
    if bias:
        layer.bias = (mx.random.normal((n,)) * 0.02).astype(dtype)
    q = nn.QuantizedLinear.from_linear(layer, group_size=group_size, bits=bits)
    q.scales = q.scales.astype(dtype)
    q.biases = q.biases.astype(dtype)
    return q


class _MLP(nn.Module):
    def __init__(self, hidden, inter, gate_bits=4, down_bits=5, dtype=mx.bfloat16):
        super().__init__()
        self.gate_proj = _qlinear(hidden, inter, bits=gate_bits, dtype=dtype)
        self.up_proj = _qlinear(hidden, inter, bits=gate_bits, dtype=dtype)
        self.down_proj = _qlinear(inter, hidden, bits=down_bits, dtype=dtype)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, hidden, q_bits, o_bits):
        super().__init__()
        self.q_proj = _qlinear(hidden, hidden, bits=q_bits)
        self.o_proj = _qlinear(hidden, hidden, bits=o_bits)


class _Layer(nn.Module):
    def __init__(self, hidden, inter, gate_bits, down_bits, q_bits, o_bits):
        super().__init__()
        self.self_attn = _Attn(hidden, q_bits, o_bits)
        self.mlp = _MLP(hidden, inter, gate_bits, down_bits)


class _Model(nn.Module):
    def __init__(self, hidden=256, inter=512, gate_bits=4, down_bits=5,
                 q_bits=8, o_bits=6, layers=2):
        super().__init__()
        self.layers = [
            _Layer(hidden, inter, gate_bits, down_bits, q_bits, o_bits)
            for _ in range(layers)
        ]


@pytest.fixture
def fake_device(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")


Q45 = {"enabled": True, "scope": "mlp", "q45_inplace": True, "act_scale": "group64"}


def _perm_v8(q):
    """Stage A v8 K order (omlx fast.qwen35_oq_a8_stage_a_v8 formula)."""
    m, k = q.shape
    return q.reshape(m, k // 64, 4, 2, 4, 2).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)


def _unperm_v8(q):
    m, k = q.shape
    return q.reshape(m, k // 64, 4, 2, 2, 4).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)


# --------------------------------------------------------------------------
# CPU mirrors of the two kernels (same math, plain MLX ops)
# --------------------------------------------------------------------------


def _mirror_natural_stage_a(x, act_scale):
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
    return q, sa, ra


def _mirror_stage_a(x, act_scale="group64", m_dim=None):
    q, sa, ra = _mirror_natural_stage_a(x, act_scale)
    return _perm_v8(q).astype(mx.int8), sa, ra


def _mirror_gemm(qa, sa, ra, weight, st, bt, *, bits, act_scale, out_dtype=None,
                 bias=None, tile=None, m_dim=None):
    m, k = qa.shape
    n = weight.shape[0]
    g = k // 64
    ones = mx.ones((n, g), dtype=mx.float32)
    codes = mx.dequantize(weight, ones, mx.zeros_like(ones), group_size=64, bits=bits)
    codes = codes.reshape(n, g, 64)
    a = _unperm_v8(qa.astype(mx.float32)).reshape(m, g, 64)
    acc = mx.einsum("mgk,ngk->mgn", a, codes)
    r = ra.T.astype(mx.float32)[..., None]
    part = st.astype(mx.float32)[None] * acc + bt.astype(mx.float32)[None] * r
    if act_scale == "per_row":
        y = part.sum(axis=1) * sa[:, None]
    else:
        y = (part * sa.T[..., None]).sum(axis=1)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(out_dtype or st.dtype)


@pytest.fixture
def mirrored(monkeypatch, fake_device):
    calls = {"stage_a": 0, "gemm": 0, "bits": []}

    def stage_a(*args, **kwargs):
        calls["stage_a"] += 1
        return _mirror_stage_a(*args, **kwargs)

    def gemm(*args, **kwargs):
        calls["gemm"] += 1
        calls["bits"].append(kwargs["bits"])
        return _mirror_gemm(*args, **kwargs)

    def no_requant(*args, **kwargs):
        raise AssertionError("an in-place module reached the requant path")

    def no_q8(*args, **kwargs):
        raise AssertionError("a Q4/Q5 module reached the Q8 kernel")

    monkeypatch.setattr(ip, "q45_stage_a", stage_a)
    monkeypatch.setattr(ip, "q45_gemm", gemm)
    monkeypatch.setattr(ip, "_requant_packed", no_requant)
    monkeypatch.setattr(ip, "q8_gemm", no_q8)
    monkeypatch.setattr(ip, "q8_stage_a", no_q8)
    return calls


# --------------------------------------------------------------------------
# policy / revision
# --------------------------------------------------------------------------


def test_policy_defaults_and_parsing():
    off = ip.Int8PrefillPolicy()
    assert off.q45_inplace is False and off.inplace_only is False
    p = ip.Int8PrefillPolicy.from_value({**Q45, "inplace_only": True})
    assert (p.enabled, p.q45_inplace, p.inplace_only, p.act_scale) == (
        True, True, True, "group64",
    )
    both = ip.Int8PrefillPolicy.from_value({**Q45, "q8_inplace": True, "inplace_only": True})
    assert both.q8_inplace and both.q45_inplace


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "q45_inplace": 1},
        {"enabled": False, "q45_inplace": True},
        {"q45_inplace": True},
        {"enabled": True, "inplace_only": True},
        {"enabled": True, "q8_inplace": True, "inplace_only": True},
        {"enabled": True, "q45_inplace": True, "inplace_only": 1},
        {"enabled": True, "q45_inplace": True, "q8_inplace": True, "q8_only": True},
        {"enabled": True, "q45_inplace": True, "act_scale": "per_tensor"},
    ],
)
def test_policy_rejects_invalid_q45_values(value):
    with pytest.raises(ValueError):
        ip.Int8PrefillPolicy.from_value(value)


def test_existing_policies_do_not_see_the_q45_mode(monkeypatch):
    policies = [
        ip.Int8PrefillPolicy(),
        ip.Int8PrefillPolicy.from_value("mlp"),
        ip.Int8PrefillPolicy.from_value("all"),
        ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True),
        ip.Int8PrefillPolicy(enabled=True, q8_inplace=True, act_scale="per_row"),
        ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True, q8_only=True),
    ]
    dicts = [p.as_dict() for p in policies]
    revisions = [p.revision for p in policies]
    for d in dicts:
        assert "q45_inplace" not in d and "inplace_only" not in d
    # The Q4/Q5 kernel source is not part of any existing revision.
    monkeypatch.setattr(ip, "Q45_KERNEL_REVISION", "0" * 16)
    assert [p.revision for p in policies] == revisions
    assert [p.as_dict() for p in policies] == dicts


def test_q45_revision_binds_its_kernel_source(monkeypatch):
    p = ip.Int8PrefillPolicy.from_value(Q45)
    before = p.revision
    monkeypatch.setattr(ip, "Q45_KERNEL_REVISION", "0" * 16)
    assert p.revision != before
    # ...and not the Q8 kernel source unless Q8 is on too.
    monkeypatch.undo()
    monkeypatch.setattr(ip, "Q8_KERNEL_REVISION", "0" * 16)
    assert p.revision == before


def test_q45_records_and_revisions_are_distinct_per_numerics():
    base = ip.Int8PrefillPolicy.from_value("mlp")
    g64 = ip.Int8PrefillPolicy.from_value(Q45)
    row = ip.Int8PrefillPolicy.from_value({**Q45, "act_scale": "per_row"})
    only = ip.Int8PrefillPolicy.from_value({**Q45, "inplace_only": True})
    both = ip.Int8PrefillPolicy.from_value({**Q45, "q8_inplace": True})
    q8 = ip.Int8PrefillPolicy.from_value({**Q45, "q45_inplace": False, "q8_inplace": True})
    policies = (base, g64, row, only, both, q8)
    assert len({p.revision for p in policies}) == len(policies)
    namespaces = {ip.apc_semantic_fingerprint("text-token-v1", p) for p in policies}
    assert len(namespaces) == len(policies)
    assert g64.as_dict()["q45_inplace"] is True and g64.as_dict()["act_scale"] == "group64"
    assert "inplace_only" not in g64.as_dict() and only.as_dict()["inplace_only"] is True
    assert ip.Int8PrefillPolicy.from_value({**Q45, "cache": "ttl"}).revision == g64.revision


def test_auto_policy_does_not_cover_q45():
    class Adapter:
        @staticmethod
        def int8_prefill_supported():
            return ("mlp", "all")

    policy = ip.auto_policy(Adapter)
    assert policy.q45_inplace is False and policy.inplace_only is False


# --------------------------------------------------------------------------
# eligibility and install
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bits,group_size,dtype,ok",
    [
        (4, 64, mx.bfloat16, True),
        (4, 64, mx.float16, True),
        (5, 64, mx.bfloat16, True),
        (5, 64, mx.float16, True),
        (4, 32, mx.bfloat16, False),
        (4, 128, mx.bfloat16, False),
        (5, 32, mx.bfloat16, False),
        (5, 128, mx.bfloat16, False),
        (8, 64, mx.bfloat16, False),
        (6, 64, mx.bfloat16, False),
        (3, 64, mx.bfloat16, False),
        (2, 64, mx.bfloat16, False),
        (4, 64, mx.float32, False),
        (5, 64, mx.float32, False),
    ],
)
def test_q45_eligibility_matrix(bits, group_size, dtype, ok):
    layer = _qlinear(512, 256, bits=bits, group_size=group_size, dtype=dtype)
    spec, reason = ip.module_spec("mlp.up_proj", layer)
    assert spec is not None, reason
    assert ip.q45_inplace_eligible(spec, layer)[0] is ok
    # The two in-place kernels never claim the same module.
    assert not (ok and ip.q8_inplace_eligible(spec, layer)[0])


def test_q45_eligibility_refuses_mixed_metadata_dtypes():
    layer = _qlinear(512, 256, bits=5)
    layer.biases = layer.biases.astype(mx.float16)
    spec, _ = ip.module_spec("mlp.up_proj", layer)
    assert ip.q45_inplace_eligible(spec, layer) == (False, "q5 scales and biases differ in dtype")


def test_q45_install_binds_only_q45_modules_and_accounts_bytes(fake_device):
    model = _Model(gate_bits=4, down_bits=5, q_bits=8, o_bits=6)
    policy = {**Q45, "scope": "all"}
    handle = ip.apply(model, policy)
    try:
        status = handle.status()
        assert status["module_kinds"] == {
            "q4_inplace": 4, "q5_inplace": 2, "q8": 2, "q6": 2,
        }
        assert handle.q45_module_count() == 6 and handle.q8_module_count() == 0
        assert handle.q45_module_count(4) == 4 and handle.q45_module_count(5) == 2
        others = [b for b in handle._bound.values() if not b.inplace]
        assert {b.spec.bits for b in others} == {8, 6}
        # Requant copies only for the Q8/Q6 modules (auto = per call).
        assert handle.transient_weight_bytes_max() == max(
            b.spec.n * (b.spec.k + 4) for b in others
        )
        assert handle.resident_estimate_bytes() == 0
        inplace = [b for b in handle._bound.values() if b.q45]
        expected_meta = sum(2 * b.module["scales"].size * 2 for b in inplace)
        assert handle.metadata_copy_bytes() == expected_meta
        assert handle.metadata_copy_bytes("q45") == expected_meta
        assert handle.metadata_copy_bytes("q8") == 0
        summary = status["q45_inplace"]
        assert summary["modules"] == 6 and summary["modules_by_bits"] == {"q4": 4, "q5": 2}
        assert summary["weight_copy_bytes"] == 0
        assert summary["metadata_copy_bytes"] == expected_meta
        assert summary["kernel_revision"] == ip.Q45_KERNEL_REVISION
        assert summary["tiles"] == {"q4": [1, 2], "q5": [2, 2]}
        assert summary["declined"] == 4 and summary["inplace_only"] is False
        assert "q8_inplace" not in status
        receipt = handle.receipt()
        assert receipt["revision"] == ip.Int8PrefillPolicy.from_value(policy).revision
        assert receipt["q45_inplace"]["modules"] == 6
        for key in ("q45_calls", "q45_rows", "q4_calls", "q5_calls", "q45_stage_a",
                    "q45_stage_a_reuse", "q45_meta_builds"):
            assert status["counts"][key] == 0
        assert "q8_calls" not in status["counts"]
    finally:
        ip.remove(handle)


def test_inplace_only_leaves_other_projections_on_stock(mirrored):
    model = _Model(gate_bits=4, down_bits=5, q_bits=8, o_bits=6, layers=1)
    attn = model.layers[0].self_attn
    x = mx.random.normal((600, 256)).astype(mx.bfloat16)
    ref_q, ref_o = attn.q_proj(x), attn.o_proj(x)
    handle = ip.apply(model, {**Q45, "scope": "all", "inplace_only": True})
    try:
        assert handle.status()["module_kinds"] == {"q4_inplace": 2, "q5_inplace": 1}
        assert handle.weight_copy_bytes() == 0
        skipped = handle._skipped
        assert set(skipped) == {"layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj"}
        assert all(reason.startswith("inplace_only:") for reason in skipped.values())
        # Unbound projections are the stock classes, bit-exact.
        assert type(attn.q_proj) is nn.QuantizedLinear
        assert mx.array_equal(attn.q_proj(x), ref_q).item()
        assert mx.array_equal(attn.o_proj(x), ref_o).item()
        assert handle.counts["engaged_calls"] == 0
    finally:
        ip.remove(handle)


def test_q8_and_q45_together_split_by_bits(fake_device):
    model = _Model(gate_bits=4, down_bits=5, q_bits=8, o_bits=6)
    policy = {**Q45, "scope": "all", "q8_inplace": True, "inplace_only": True}
    handle = ip.apply(model, policy)
    try:
        status = handle.status()
        assert status["module_kinds"] == {"q4_inplace": 4, "q5_inplace": 2, "q8_inplace": 2}
        # o_proj (6-bit) is the only module neither kernel takes.
        assert set(handle._skipped) == {"layers.0.self_attn.o_proj", "layers.1.self_attn.o_proj"}
        assert status["q8_inplace"]["declined"] == 2
        assert status["q45_inplace"]["declined"] == 2
        assert status["q8_inplace"]["metadata_copy_bytes"] == handle.metadata_copy_bytes("q8")
        assert (handle.metadata_copy_bytes("q8") + handle.metadata_copy_bytes("q45")
                == handle.metadata_copy_bytes())
        assert handle.weight_copy_bytes() == 0
    finally:
        ip.remove(handle)


def test_q45_inplace_without_any_q45_module_fails_closed(fake_device):
    model = _Model(gate_bits=8, down_bits=6, q_bits=8, o_bits=6)
    with pytest.raises(ip.Int8PrefillError, match="q45_inplace"):
        ip.apply(model, {**Q45, "scope": "all"})
    assert not any(ip._STATE_ATTR in m.__dict__ for _, m in model.named_modules())


def test_q45_cache_none_reports_transient_metadata(fake_device):
    model = _Model(gate_bits=4, down_bits=5)
    handle = ip.apply(model, {**Q45, "cache": "none"})
    try:
        largest = max(b.meta_size() for b in handle._bound.values())
        assert handle.metadata_copy_bytes() == largest
        assert handle.warmup() == 0
    finally:
        ip.remove(handle)


# --------------------------------------------------------------------------
# routing (CPU mirrors in place of the kernels)
# --------------------------------------------------------------------------


def test_q45_routing_counts_shares_stage_a_and_never_requantizes(mirrored):
    model = _Model(gate_bits=4, down_bits=5, layers=1)
    mlp = model.layers[0].mlp
    x = (mx.random.normal((2, 300, 256)) * 0.5).astype(mx.bfloat16)  # 600 rows
    reference = mlp(x)
    handle = ip.apply(model, Q45)
    try:
        got = mlp(x)
        mx.eval(got, reference)
        assert got.shape == reference.shape and got.dtype == mx.bfloat16
        rel = (mx.linalg.norm((got - reference).astype(mx.float32))
               / mx.linalg.norm(reference.astype(mx.float32))).item()
        assert rel < 0.08, rel
        counts = handle.counts
        assert counts["q45_calls"] == 3 and counts["engaged_calls"] == 3
        assert counts["q4_calls"] == 2 and counts["q5_calls"] == 1
        assert counts["q45_rows"] == 1800 and counts["engaged_rows"] == 1800
        # gate/up share x; down quantizes h.
        assert counts["q45_stage_a"] == 2 and counts["q45_stage_a_reuse"] == 1
        assert mirrored["stage_a"] == 2 and mirrored["bits"] == [4, 4, 5]
        assert counts["weight_builds"] == 0 and handle.weight_bytes() == 0
        assert counts["q45_meta_builds"] == 3
        mx.eval(mlp(x * 1))
        assert counts["q45_meta_builds"] == 3 and counts["q45_calls"] == 6
        assert handle.metadata_bytes() == handle.metadata_copy_bytes()
        assert handle.evict_weights() == 3 and handle.metadata_bytes() == 0
    finally:
        ip.remove(handle)


class _GDN(nn.Module):
    """qkv (Q4) and z (Q5) share one input, as in Qwen3.8 linear attention."""

    def __init__(self, hidden):
        super().__init__()
        self.in_proj_qkv = _qlinear(hidden, 512, bits=4)
        self.in_proj_z = _qlinear(hidden, 256, bits=5)

    def __call__(self, x):
        return self.in_proj_qkv(x), self.in_proj_z(x)


class _GDNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _GDN(256)


def test_q4_and_q5_projections_share_one_stage_a(mirrored):
    model = _GDNModel()
    handle = ip.apply(model, {**Q45, "scope": "all"})
    try:
        mx.eval(*model.linear_attn(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        counts = handle.counts
        assert counts["q4_calls"] == 1 and counts["q5_calls"] == 1
        assert counts["q45_stage_a"] == 1 and counts["q45_stage_a_reuse"] == 1
    finally:
        ip.remove(handle)


def test_q8_and_q45_on_one_input_use_their_own_stage_a(monkeypatch, fake_device):
    import test_int8_prefill_q8_inplace as q8t

    monkeypatch.setattr(ip, "q45_stage_a", _mirror_stage_a)
    monkeypatch.setattr(ip, "q45_gemm", _mirror_gemm)
    monkeypatch.setattr(ip, "q8_stage_a", q8t._mirror_stage_a)
    monkeypatch.setattr(ip, "q8_gemm", q8t._mirror_gemm)
    model = _GDNModel()
    model.linear_attn.in_proj_z = _qlinear(256, 256, bits=8)
    handle = ip.apply(model, {**Q45, "scope": "all", "q8_inplace": True})
    try:
        mx.eval(*model.linear_attn(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        counts = handle.counts
        assert counts["q4_calls"] == 1 and counts["q8_calls"] == 1
        assert counts["q45_stage_a"] == 1 and counts["q8_stage_a"] == 1
        assert counts["q45_stage_a_reuse"] == 0 and counts["q8_stage_a_reuse"] == 0
    finally:
        ip.remove(handle)


def test_q45_per_row_scaling_uses_its_stage_a(mirrored, monkeypatch):
    seen = []
    real = ip.q45_stage_a

    def spy(x, act_scale="group64", m_dim=None):
        seen.append(act_scale)
        return real(x, act_scale, m_dim=m_dim)

    monkeypatch.setattr(ip, "q45_stage_a", spy)
    model = _Model(gate_bits=4, down_bits=5, layers=1)
    handle = ip.apply(model, {**Q45, "act_scale": "per_row"})
    try:
        mx.eval(model.layers[0].mlp(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        assert seen == ["per_row", "per_row"]
        assert handle.counts["q45_calls"] == 3
    finally:
        ip.remove(handle)


def test_q45_sub_threshold_and_dtype_mismatch_stay_stock(mirrored):
    model = _Model(gate_bits=4, down_bits=5, layers=1)
    mlp = model.layers[0].mlp
    small = mx.random.normal((16, 256)).astype(mx.bfloat16)
    wide16 = mx.random.normal((600, 256)).astype(mx.float16)
    ref_small, ref16 = mlp(small), mlp(wide16)
    handle = ip.apply(model, Q45)
    try:
        assert mx.array_equal(mlp(small), ref_small).item()
        assert mx.array_equal(mlp(wide16), ref16).item()
        counts = handle.counts
        assert counts["q45_calls"] == 0 and mirrored["gemm"] == 0
        assert counts["fallback_rows"] == 3 and counts["fallback_dtype"] == 3
    finally:
        ip.remove(handle)


def test_prometheus_exports_q45_series_only_when_on(mirrored):
    from mlx2.prometheus import PrometheusBuilder, _add_int8_prefill

    class Engine:
        int8_prefill_policy = ip.Int8PrefillPolicy.from_value(Q45)
        int8_prefill_handle = None

    off = type("Off", (), {"int8_prefill_policy": ip.Int8PrefillPolicy.from_value("mlp"),
                           "int8_prefill_handle": None})()
    builder = PrometheusBuilder()
    _add_int8_prefill(builder, off)
    assert "q45" not in builder.render()
    engine = Engine()
    model = _Model(gate_bits=4, down_bits=5, layers=1)
    engine.int8_prefill_handle = ip.apply(model, Engine.int8_prefill_policy)
    try:
        mx.eval(model.layers[0].mlp(mx.random.normal((600, 256)).astype(mx.bfloat16)))
        builder = PrometheusBuilder()
        _add_int8_prefill(builder, engine)
        text = builder.render()
        assert 'mlx2_int8_prefill_q45_inplace_calls_total{bits="4"} 2' in text
        assert 'mlx2_int8_prefill_q45_inplace_calls_total{bits="5"} 1' in text
        assert "mlx2_int8_prefill_q45_inplace_rows_total 1800" in text
        assert 'mlx2_int8_prefill_q45_inplace_modules{bits="4"} 2' in text
        assert 'mlx2_int8_prefill_calls_total{outcome="engaged"} 3' in text
    finally:
        ip.remove(engine.int8_prefill_handle)


def test_bind_for_serving_records_q45_settings(fake_device):
    class Adapter:
        model = _Model(gate_bits=4, down_bits=5)

        @staticmethod
        def int8_prefill_supported():
            return ("mlp",)

    policy = {**Q45, "inplace_only": True}
    handle, settings = ip.bind_for_serving(
        Adapter(), policy, max_lanes=4, config={}, speculation="ordinary"
    )
    try:
        assert settings["q45_inplace"] is True and settings["inplace_only"] is True
        assert settings["q45_modules"] == {"q4": 4, "q5": 2}
        assert settings["weight_copy_bytes"] == 0
        assert settings["metadata_copy_bytes"] == handle.metadata_copy_bytes() > 0
        assert settings["revision"] == ip.Int8PrefillPolicy.from_value(policy).revision
        assert "q8_modules" not in settings
        assert handle.metadata_bytes() == settings["metadata_copy_bytes"]
    finally:
        ip.remove(handle)


def test_serving_engine_records_q45_route(monkeypatch, fake_device):
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
        bits=4,
        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0,
    )
    mx.eval(model.parameters())
    policy = {**Q45, "inplace_only": True}
    engine = _serving_engine(model, ("mlp",), int8_prefill=policy)
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        recorded = status["settings"]["int8_prefill"]
        assert recorded["q45_inplace"] is True and recorded["inplace_only"] is True
        assert recorded["q45_modules"]["q4"] > 0 and recorded["weight_copy_bytes"] == 0
        assert recorded["revision"] == ip.Int8PrefillPolicy.from_value(policy).revision
        assert status["int8_prefill"]["q45_inplace"]["modules"] == recorded["q45_modules"]["q4"]
        assert "-q45a8-group64" in status["profile"]
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


def _pack_bitstream(codes, bits):
    """MLX affine layout, built independently: code i of a row occupies bits
    [i * bits, (i + 1) * bits) of a little-endian uint32 stream."""
    n, k = codes.shape
    stream = (codes[..., None].astype(np.uint64) >> np.arange(bits, dtype=np.uint64)) & 1
    stream = stream.reshape(n, k * bits // 32, 32)
    return (stream << np.arange(32, dtype=np.uint64)).sum(axis=-1).astype(np.uint32)


def _straddling(bits):
    """In-group code indices whose field crosses a 32-bit word boundary."""
    return [i for i in range(64) if (i * bits) % 32 + bits > 32]


def _random_q45(n, k, bits, dtype, seed):
    """Packed Q4/Q5 weight from random codes with edge cases forced, random
    metadata, and the float64 dequantized reference."""
    rng = np.random.default_rng(seed)
    top = (1 << bits) - 1
    codes = rng.integers(0, top + 1, size=(n, k)).astype(np.uint32)
    codes[:, :2] = np.array([0, top], dtype=np.uint32)
    codes[0, :] = top
    codes[1, :] = 0
    straddle = np.array(_straddling(bits))
    if straddle.size:
        g = k // 64
        cols = (np.arange(g)[:, None] * 64 + straddle[None]).ravel()
        codes[2, :] = 0
        codes[2, cols] = top  # max codes exactly on the word seams
        codes[3, :] = top
        codes[3, cols] = 0
    g = k // 64
    scales = (rng.standard_normal((n, g)) * 0.01).astype(np.float32)
    biases = (rng.standard_normal((n, g)) * 0.1).astype(np.float32)
    s = mx.array(scales).astype(dtype)
    b = mx.array(biases).astype(dtype)
    packed = mx.array(_pack_bitstream(codes, bits))
    s64 = np.repeat(np.array(s.astype(mx.float32)).astype(np.float64), 64, axis=1)
    b64 = np.repeat(np.array(b.astype(mx.float32)).astype(np.float64), 64, axis=1)
    w64 = codes.astype(np.float64) * s64 + b64
    return packed, s, b, w64


def _dequant_activation(qa, sa, act_scale):
    q = _unperm_v8(np.array(qa).astype(np.float64))
    s = np.array(sa).astype(np.float64)
    if act_scale == "per_row":
        return q * s[:, None]
    return q * np.repeat(s.T, 64, axis=1)


def test_straddling_fields_exist_for_q5_only():
    assert _straddling(4) == []
    assert _straddling(5) == [6, 12, 19, 25, 38, 44, 51, 57]


@pytest.mark.parametrize("bits", ip.Q45_BITS)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_gpu_independent_pack_matches_mlx_dequantize(gpu, bits, dtype):
    packed, s, b, w64 = _random_q45(256, 512, bits, dtype, bits)
    mx_w = mx.dequantize(
        packed, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits
    )
    assert np.array_equal(
        np.array(mx_w).astype(np.float64), w64.astype(np.float32).astype(np.float64)
    )


@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("m", [1, 7, 130])
def test_gpu_stage_a_v8_codes_sums_and_scales(gpu, act_scale, dtype, m):
    k = 640
    rng = np.random.default_rng(m)
    xn = (rng.standard_normal((m, k)) * 2.0).astype(np.float32)
    xn[0, :64] = 0.0
    if m > 1:
        xn[1, :] = 0.0
        xn[-1, 5] = 40.0
    x = mx.array(xn).astype(dtype)
    qa, sa, ra = ip.q45_stage_a(x, act_scale)
    q8a, q8s, q8r = ip.q8_stage_a(x, act_scale)
    mx.eval(qa, sa, ra, q8a, q8s, q8r)
    xf = np.array(x.astype(mx.float32)).astype(np.float32)
    g = k // 64
    if act_scale == "per_row":
        amax = np.abs(xf).max(axis=1)
        inv = np.where(amax > 0, np.float32(127.0) / np.maximum(amax, 1e-30), 0).astype(np.float32)
        ref = np.clip(np.rint(xf * inv[:, None]), -127, 127)
        assert sa.shape == (m,)
    else:
        xg = xf.reshape(m, g, 64)
        amax = np.abs(xg).max(axis=2)
        inv = np.where(amax > 0, np.float32(127.0) / np.maximum(amax, 1e-30), 0).astype(np.float32)
        ref = np.clip(np.rint(xg * inv[..., None]), -127, 127).reshape(m, k)
        assert sa.shape == (g, m)
    got = np.array(qa).astype(np.int32)
    natural = _unperm_v8(got)
    diff = np.abs(natural - ref)
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, (diff.max(), (diff > 0).mean())
    # Exactly the Q8 Stage A codes in omlx's v8 order; scales and sums equal.
    assert np.array_equal(got, _perm_v8(np.array(q8a).astype(np.int32)))
    assert np.array_equal(np.array(sa), np.array(q8s))
    assert ra.shape == (g, m) and ra.dtype == mx.int16
    assert np.array_equal(np.array(ra).T, got.reshape(m, g, 64).sum(axis=2))


_GEMM_CASES = [
    (1, 512, 256),
    (33, 1024, 256),
    (100, 256, 512),
    (257, 640, 256),
]


@pytest.mark.parametrize("tile", ip.Q45_TILES)
@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", ip.Q45_BITS)
@pytest.mark.parametrize("m,k,n", _GEMM_CASES)
def test_gpu_q45_gemm_matches_independent_reference(gpu, tile, act_scale, dtype, bits, m, k, n):
    packed, s, b, w64 = _random_q45(n, k, bits, dtype, m + k + n + bits)
    x = (mx.random.normal((m, k)) * 1.5).astype(dtype)
    qa, sa, ra = ip.q45_stage_a(x, act_scale)
    st, bt = ip.q8_metadata(s, b)
    y = ip.q45_gemm(qa, sa, ra, packed, st, bt, bits=bits, act_scale=act_scale,
                    out_dtype=mx.float32, tile=tile)
    yb = ip.q45_gemm(qa, sa, ra, packed, st, bt, bits=bits, act_scale=act_scale, tile=tile)
    mx.eval(y, yb)
    a64 = _dequant_activation(qa, sa, act_scale)
    ref = a64 @ w64.T
    scale = np.abs(a64) @ np.abs(w64).T
    err = np.abs(np.array(y).astype(np.float64) - ref)
    assert (err / np.maximum(scale, 1e-30)).max() < 2e-6, (err / scale).max()
    assert yb.dtype == dtype
    assert mx.array_equal(yb, y.astype(dtype)).item()


@pytest.mark.parametrize("bits", ip.Q45_BITS)
@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
def test_gpu_q45_tiles_are_bit_identical(gpu, bits, act_scale):
    m, k, n = 300, 1024, 512
    packed, s, b, _ = _random_q45(n, k, bits, mx.bfloat16, 3)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    qa, sa, ra = ip.q45_stage_a(x, act_scale)
    st, bt = ip.q8_metadata(s, b)
    outs = [ip.q45_gemm(qa, sa, ra, packed, st, bt, bits=bits, act_scale=act_scale,
                        out_dtype=mx.float32, tile=t) for t in ip.Q45_TILES]
    mx.eval(*outs)
    for out in outs[1:]:
        assert mx.array_equal(out, outs[0]).item()


@pytest.mark.parametrize("bits", ip.Q45_BITS)
def test_gpu_q45_bias_epilogue(gpu, bits):
    m, k, n = 64, 512, 256
    packed, s, b, _ = _random_q45(n, k, bits, mx.bfloat16, 9)
    bias = (mx.random.normal((n,)) * 0.1).astype(mx.bfloat16)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    qa, sa, ra = ip.q45_stage_a(x, "group64")
    st, bt = ip.q8_metadata(s, b)
    y0 = ip.q45_gemm(qa, sa, ra, packed, st, bt, bits=bits, out_dtype=mx.float32)
    y1 = ip.q45_gemm(qa, sa, ra, packed, st, bt, bits=bits, out_dtype=mx.float32, bias=bias)
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


@pytest.mark.parametrize("bits", ip.Q45_BITS)
@pytest.mark.parametrize("act_scale", ip.Q8_ACT_SCALES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_gpu_q45_module_route_vs_stock_and_counters(gpu, bits, act_scale, dtype):
    mx.random.seed(5)
    model = _One(_qlinear(5120, 1024, bits=bits, dtype=dtype))
    x = mx.random.normal((2, 1024, 5120)).astype(dtype)
    decode = x[:, :8]
    stock, stock_decode = model(x), model(decode)
    mx.eval(stock, stock_decode)
    handle = ip.apply(model, {**Q45, "act_scale": act_scale})
    try:
        got, got_decode = model(x), model(decode)
        mx.eval(got, got_decode)
        rel = _rel(got, stock)
        print(f"q{bits} {act_scale} {dtype}: rel L2 vs stock quantized_matmul {rel:.5f}")
        assert got.dtype == stock.dtype and got.shape == stock.shape
        assert rel < 0.02, rel
        assert mx.array_equal(got_decode, stock_decode).item()
        counts = handle.counts
        assert counts["q45_calls"] == 1 and counts["q45_rows"] == 2048
        assert counts[f"q{bits}_calls"] == 1
        assert counts["weight_builds"] == 0 and handle.weight_bytes() == 0
        assert handle.metadata_bytes() == 2 * 1024 * 80 * 2
    finally:
        ip.remove(handle)


def test_gpu_mixed_q4_q5_q8_model_routes_each_kernel(gpu):
    model = _Model(hidden=512, inter=1024, gate_bits=4, down_bits=5, q_bits=8, o_bits=4,
                   layers=1)
    layer = model.layers[0]
    x = mx.random.normal((1024, 512)).astype(mx.bfloat16)
    stock = (layer.mlp(x), layer.self_attn.q_proj(x), layer.self_attn.o_proj(x))
    mx.eval(*stock)
    policy = {**Q45, "scope": "all", "q8_inplace": True, "inplace_only": True}
    handle = ip.apply(model, policy)
    try:
        got = (layer.mlp(x), layer.self_attn.q_proj(x), layer.self_attn.o_proj(x))
        mx.eval(*got)
        for g, s in zip(got, stock):
            assert _rel(g, s) < 0.03
        counts = handle.counts
        assert counts["q4_calls"] == 3 and counts["q5_calls"] == 1 and counts["q8_calls"] == 1
        # gate/up share one Stage A v8 of x, down quantizes h, and o_proj
        # re-quantizes x (the single-entry cache holds h by then); q_proj
        # (Q8) has its own natural-order Stage A.
        assert counts["q45_stage_a"] == 3 and counts["q45_stage_a_reuse"] == 1
        assert counts["q8_stage_a"] == 1
    finally:
        ip.remove(handle)
