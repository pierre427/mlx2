"""Source HC metadata/ownership/oracle checks; no Metal or checkpoint run."""
import hashlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v41_source_hc as hc
from mlx2.runtime.models.deepseek_v4_hc import reference_split
from mlx2.runtime.models.deepseek_v4 import _sinkhorn_mix_fp32


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    verify, kernel = hc._verify_installed, hc._kernel
    verify.cache_clear()
    kernel.cache_clear()
    def forbidden(*args, **kwargs): raise AssertionError("CPU oracle reached Metal")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    verify.cache_clear()
    kernel.cache_clear()
    mx.set_default_device(old)


def inputs(rows=7, dtype=mx.float32, batched=True):
    return (mx.zeros((1, rows, 24) if batched else (rows, 24), dtype),
            mx.ones((3,), dtype), mx.zeros((24,), dtype))


@pytest.mark.parametrize("rows", [1, 7, 99, 1000])
@pytest.mark.parametrize("batched", [True, False])
def test_bound_geometry(rows, batched):
    assert hc.plan(*inputs(rows, batched=batched)) == rows


@pytest.mark.parametrize("bad", ["zero", "too_many", "batch2", "rank1", "width", "scale", "base", "half"])
def test_bad_geometry_refused_before_device_or_source_checks(monkeypatch, bad):
    z, s, b = inputs()
    if bad == "zero": z = mx.zeros((1, 0, 24))
    elif bad == "too_many": z = mx.zeros((1, 1001, 24))
    elif bad == "batch2": z = mx.zeros((2, 7, 24))
    elif bad == "rank1": z = mx.zeros((24,))
    elif bad == "width": z = mx.zeros((1, 7, 25))
    elif bad == "scale": s = mx.ones((4,))
    elif bad == "base": b = mx.zeros((23,))
    else: z = z.astype(mx.float16)
    def forbidden(): raise AssertionError("invalid geometry reached GPU")
    monkeypatch.setattr(hc, "_device", forbidden)
    with pytest.raises(ValueError): hc.split(z, s, b)


@pytest.mark.parametrize("iterations", [0, 101, True, 2.0])
def test_bad_iterations(iterations):
    with pytest.raises(ValueError, match="iterations"):
        hc.plan(*inputs(), iterations=iterations)


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True, "epsilon"])
def test_bad_epsilon(eps):
    with pytest.raises(ValueError, match="epsilon"):
        hc.plan(*inputs(), eps=eps)


@pytest.mark.parametrize("key,value", [("reduction", "automatic"), ("softmax_exp", "guess")])
def test_unknown_candidate_refuses(key, value):
    with pytest.raises(ValueError, match="candidate"):
        hc.plan(*inputs(), **{key: value})


def test_cpu_invocation_refuses_no_fallback():
    with pytest.raises(ValueError, match="owned Metal"):
        hc.split(*inputs())


def test_installed_pins_are_current_and_do_not_execute_gpu():
    receipt = hc.installed_receipt()
    assert receipt["version"] == "0.32.4"
    assert set(receipt["files_sha256"]) == set(hc.INSTALLED_PINS)
    assert receipt["files_sha256"] == hc.INSTALLED_PINS


@pytest.mark.parametrize("change", ["version", "missing", "content"])
def test_source_drift_refuses(tmp_path, monkeypatch, change):
    source = tmp_path/"source.h"
    source.write_bytes(b"source")
    pins = {"source.h": hashlib.sha256(b"source").hexdigest()}
    monkeypatch.setattr(hc, "INSTALLED_PINS", pins)
    monkeypatch.setattr(hc, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(hc.importlib.metadata, "version", lambda name: "wrong" if change == "version" else hc.INSTALLED_VERSION)
    if change == "missing": source.unlink()
    elif change == "content": source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="version|missing|changed"):
        hc.installed_receipt()


@pytest.mark.parametrize("reduction", hc.REDUCTIONS)
def test_shader_retains_stores_and_math_boundaries(reduction):
    header, source = hc.compilation_source(reduction=reduction)
    assert "#pragma clang fp contract(off)" in header
    assert "metal::precise::exp(metal::abs(x))" in header
    assert "x<0.0f ? z : 1.0f-z" in header
    assert "metal::fast::exp(c[r*4+s]-maximum)" in source
    assert "const float inverse=1.0f/total" in source
    assert "values[s]*inverse+epsilon" in source
    assert "ITERS" in source and "iteration=1" in source
    assert "ROW_SUM" not in source and "COL_SUM" not in source
    assert "bfloat" not in source+header and "half" not in source+header
    assert "rms" not in source and "projection" not in source


def test_compile_spelling_choices_are_explicit_and_do_not_rewrite_sigmoid():
    for spelling in hc.SOFTMAX_EXPS:
        h, s = hc.compilation_source(softmax_exp=spelling)
        assert "metal::precise::exp(metal::abs(x))" in h
        assert spelling+"(c[r*4+s]-maximum)" in s
    with pytest.raises(ValueError): hc.compilation_source(sigmoid_exp="fast::exp")


def test_served_sigmoid_gate_refuses_unknown_and_returns_known(monkeypatch):
    monkeypatch.setattr(hc, "_device", lambda: mx)
    monkeypatch.setattr(hc, "_sigmoid_gate", lambda: SimpleNamespace(spelling=lambda dtype: None))
    with pytest.raises(ValueError, match="cannot reproduce"):
        hc.sigmoid_spelling()
    monkeypatch.setattr(hc, "_sigmoid_gate", lambda: SimpleNamespace(spelling=lambda dtype: "metal::precise::exp"))
    assert hc.sigmoid_spelling() == "metal::precise::exp"


def test_dispatch_binds_exact_inputs_and_preserves_raw_f32_outputs(monkeypatch):
    created, submitted = [], []
    def factory(**kwargs):
        created.append(kwargs)
        def launch(**args):
            submitted.append(args)
            return [mx.arange(7*24, dtype=mx.float32).reshape(1, 7, 24)]
        return launch
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    monkeypatch.setattr(hc, "_device", lambda: mx)
    monkeypatch.setattr(hc, "_verify_installed", lambda: ())
    monkeypatch.setattr(hc, "sigmoid_spelling", lambda: "metal::precise::exp")
    z, s, b = inputs()
    before = sum(hc._SCHEDULED.values())
    pre, post, comb = hc.split(z, s, b)
    assert pre.shape == post.shape == (1, 7, 4) and comb.shape == (1, 7, 4, 4)
    assert pre.dtype == post.dtype == comb.dtype == mx.float32
    assert submitted[0]["inputs"][:3] == [z, s, b]
    assert submitted[0]["template"] == [("ROWS", 7), ("ITERS", 20)]
    assert submitted[0]["threadgroup"] == (7, 1, 1)
    assert created[0]["compile_options"] == {"math_mode": "safe"}
    np.testing.assert_array_equal(np.array(pre), np.arange(7*24).reshape(1, 7, 24)[..., :4])
    assert sum(hc._SCHEDULED.values()) == before+1


@pytest.mark.parametrize("rows", [1, 7, 99])
def test_independent_high_precision_coefficient_oracle(rows):
    rng = np.random.default_rng(40+rows)
    z = rng.normal(0, 2, (1, rows, 24)).astype(np.float32)
    s = np.array([.1, .2, .3], np.float32)
    b = rng.normal(0, .1, 24).astype(np.float32)
    expected = reference_split(z, s, b, iterations=20, eps=1e-6)
    actual = _sinkhorn_mix_fp32(mx.array(z), mx.array(s), mx.array(b), 4, 20, 1e-6)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.array(a), e, rtol=2e-6, atol=3e-7)
    # This validates the CPU ordinary oracle, not the Metal candidate.
    assert np.isfinite(np.array(actual[2])).all() and (np.array(actual[2]) > 0).all()
    np.testing.assert_allclose(np.array(actual[2]).sum(-2), np.ones((1, rows, 4)), rtol=0, atol=2e-6)


def _source_block(rows=7, count=4, policy="source_fused_fp32"):
    from mlx2.runtime.models.deepseek_v41 import Block
    rng = np.random.default_rng(472)
    dim = 8
    h = mx.array(rng.normal(0, .25, (1, rows, count, dim)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    width = 2*count+count*count
    fn = mx.array(rng.normal(0, .125, (width, count*dim)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    scale = mx.array([.1, .2, .3], mx.float32)
    base = mx.array(rng.normal(0, .1, width).astype(np.float32))
    block = object.__new__(Block)
    block.c = SimpleNamespace(hc_mult=count, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    block.model = SimpleNamespace(hc_policy="reference", hc_coeff_policy=policy, scheduled_hc_calls={})
    block.hc = {"attn": (fn, scale, base)}
    return block, h


@pytest.mark.parametrize("rows", [1, 7, 99, 1000])
def test_source_callback_retains_unrounded_projection_and_rms(rows, monkeypatch):
    block, h = _source_block(rows)
    calls = []
    def source(mixes, scale, base, *, iterations, eps):
        calls.append((mixes, scale, base, iterations, eps))
        return _sinkhorn_mix_fp32(mixes, scale, base, 4, iterations, eps)
    monkeypatch.setattr(hc, "split", source)
    actual = block.mixes(h, "attn")
    fn, scale, base = block.hc["attn"]
    flat = h.reshape(1, rows, -1).astype(mx.float32)
    projection = flat @ fn.T
    expected_mixes = projection*mx.rsqrt(mx.mean(mx.square(flat), -1, keepdims=True)+block.c.rms_norm_eps)
    np.testing.assert_array_equal(np.array(calls[0][0]).view(np.uint32), np.array(expected_mixes).view(np.uint32))
    assert np.any(np.array(projection).view(np.uint32) != np.array(projection.astype(mx.bfloat16).astype(mx.float32)).view(np.uint32))
    for a, e in zip(actual, _sinkhorn_mix_fp32(expected_mixes, scale, base, 4, 20, 1e-6)):
        np.testing.assert_array_equal(np.array(a).view(np.uint32), np.array(e).view(np.uint32))
    assert calls[0][1] is scale and calls[0][2] is base and calls[0][3:] == (20, 1e-6)
    assert block.model.hc_policy == "reference"
    assert block.model.scheduled_hc_calls == {"source_fused_fp32": 1}


@pytest.mark.parametrize("rows,count", [(1001, 4), (7, 3)])
def test_unsupported_source_geometry_uses_ordinary_coefficients(rows, count, monkeypatch):
    block, h = _source_block(rows, count)
    def forbidden(*args, **kwargs): raise AssertionError("fallback invoked Source or legacy GPU kernel")
    monkeypatch.setattr(hc, "split", forbidden)
    monkeypatch.setattr(__import__('mlx2.runtime.models.deepseek_v4_hc', fromlist=['split_sinkhorn']), "split_sinkhorn", forbidden)
    actual = block.mixes(h, "attn")
    assert all(np.isfinite(np.array(v)).all() for v in actual)
    assert block.model.scheduled_hc_calls == {"reference_fallback": 1}


def test_source_kernel_refusal_is_not_a_silent_numerical_fallback(monkeypatch):
    block, h = _source_block()
    def refused(*args, **kwargs): raise ValueError("source gate refused")
    monkeypatch.setattr(hc, "split", refused)
    with pytest.raises(ValueError, match="source gate refused"):
        block.mixes(h, "attn")
    assert block.model.scheduled_hc_calls == {}


def test_reference_coefficient_policy_never_calls_source(monkeypatch):
    block, h = _source_block(policy="reference")
    def forbidden(*args, **kwargs): raise AssertionError("reference called Source")
    monkeypatch.setattr(hc, "split", forbidden)
    assert len(block.mixes(h, "attn")) == 3
    assert not block.model.scheduled_hc_calls


@pytest.mark.parametrize("kwargs", [{"hc_coeff_policy": "unknown"},
    {"hc_coeff_policy": "source_fused_fp32", "hc_policy": "fused_fp32"}])
def test_source_coefficient_policy_refuses_unknown_or_ambiguous(kwargs):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    from test_deepseek_v41_cpu import config, Weights
    c = config()
    with pytest.raises(ValueError, match="Source HC|legacy fused"):
        DeepSeekV41Model(c, Weights(c), identity="synthetic@source-hc", **kwargs)


def test_source_coefficient_identity_rejects_cross_profile_cache(monkeypatch):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    from test_deepseek_v41_cpu import config, Weights
    c = config()
    reference = DeepSeekV41Model(c, Weights(c), identity="synthetic@source-hc")
    source = DeepSeekV41Model(c, Weights(c), identity="synthetic@source-hc", hc_coeff_policy="source_fused_fp32")
    assert reference.identity != source.identity and source.hc_policy == "reference"
    with pytest.raises(ValueError, match="identity"):
        source(mx.array([[1]], mx.int32), reference.make_cache())
    calls = []
    def stock(mixes, scale, base, *, iterations, eps):
        calls.append((mixes.shape, iterations, eps))
        return _sinkhorn_mix_fp32(mixes, scale, base, 4, iterations, eps)
    monkeypatch.setattr(hc, "split", stock)
    a = reference(mx.array([[1, 2, 3]], mx.int32), reference.make_cache())
    b = source(mx.array([[1, 2, 3]], mx.int32), source.make_cache())
    np.testing.assert_array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))
    assert len(calls) == 2*c.num_hidden_layers
    assert source.scheduled_hc_calls == {"source_fused_fp32": 2*c.num_hidden_layers}


def test_source_coefficient_profile_snapshot_replay_preserves_control_state(monkeypatch):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    from test_deepseek_v41_cpu import config, Weights
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@source-hc-replay", hc_coeff_policy="source_fused_fp32")
    def stock(mixes, scale, base, *, iterations, eps):
        return _sinkhorn_mix_fp32(mixes, scale, base, 4, iterations, eps)
    monkeypatch.setattr(hc, "split", stock)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]], mx.int32), cache)
    snapshots = [s.snapshot() for s in cache]
    first = model(mx.array([[4]], mx.int32), cache)
    for state, snapshot in zip(cache, snapshots): state.restore(snapshot)
    replay = model(mx.array([[4]], mx.int32), cache)
    np.testing.assert_array_equal(np.array(first).view(np.uint32), np.array(replay).view(np.uint32))
    assert all(not s.pending and s.offset == 4 for s in cache)
    assert model.scheduled_hc_calls == {"source_fused_fp32": 6*c.num_hidden_layers}


def test_source_gate_error_poisoned_frontier_cannot_be_published(monkeypatch):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    from test_deepseek_v41_cpu import config, Weights
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@source-hc-refusal", hc_coeff_policy="source_fused_fp32")
    cache = model.make_cache()
    def refused(*args, **kwargs): raise ValueError("Source HC served gate refused")
    monkeypatch.setattr(hc, "split", refused)
    with pytest.raises(ValueError, match="served gate refused"):
        model(mx.array([[1]], mx.int32), cache)
    assert model._weight_executions == 0 and model.scheduled_hc_calls == {}
    assert all(s.pending for s in cache)
    from mlx2.runtime.cache_operations import CacheOperationsUnsupported
    for state in cache:
        with pytest.raises(CacheOperationsUnsupported, match="incomplete|pending|publication"):
            state.snapshot()
    with pytest.raises(ValueError, match="identity/topology/frontier"):
        model(mx.array([[1]], mx.int32), cache)
