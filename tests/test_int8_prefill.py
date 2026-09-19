"""Int8 (W8A8) NAX prefill: policy, scoping, fail-closed, and GPU numerics.

CPU tests (the default for the suite) never launch a Metal kernel: they cover
policy parsing, eligibility, per-model scoping, removal, device refusal, the
serving/qualification/Prometheus surfaces, and that sub-threshold (decode)
calls stay bit-exact on the stock path.  Numerics tests need an M5-class GPU
and an explicit opt-in (``MLX2_TEST_INT8_NAX=1`` or
``MLX2_TEST_OPTIONAL_METAL=1``).
"""

import os

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.contracts import Fidelity
from mlx2.runtime import int8_prefill as ip

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _linear(k, n, *, bits=None, bias=False, dtype=mx.bfloat16):
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = (mx.random.normal((n, k)) * 0.02).astype(dtype)
    if bias:
        layer.bias = (mx.random.normal((n,)) * 0.02).astype(dtype)
    if bits:
        layer = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)
    return layer


class _MLP(nn.Module):
    def __init__(self, hidden, inter, bits):
        super().__init__()
        self.gate_proj = _linear(hidden, inter, bits=bits)
        self.up_proj = _linear(hidden, inter, bits=bits)
        self.down_proj = _linear(inter, hidden, bits=bits)
        # MoE router: tiny, must never be touched.
        self.gate = _linear(hidden, 128, bits=None)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, hidden, bits):
        super().__init__()
        self.q_proj = _linear(hidden, hidden, bits=bits)
        self.k_proj = _linear(hidden, 384, bits=bits)  # 384 % 128 == 0
        self.kv_a_proj = _linear(hidden, 576, bits=bits)  # 576 % 128 != 0
        self.o_proj = _linear(hidden, hidden, bits=bits)


class _Layer(nn.Module):
    def __init__(self, hidden, inter, bits):
        super().__init__()
        self.self_attn = _Attn(hidden, bits)
        self.mlp = _MLP(hidden, inter, bits)


class _Model(nn.Module):
    def __init__(self, hidden=512, inter=1024, bits=6, layers=2):
        super().__init__()
        self.layers = [_Layer(hidden, inter, bits) for _ in range(layers)]
        self.mtp = _MLP(hidden, inter, bits)
        self.lm_head = _linear(hidden, 40960, bits=bits)


@pytest.fixture
def fake_device(monkeypatch):
    """Bypass the device probe for CPU-only structural tests."""
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")


def _kinds(model):
    return {path: type(m) for path, m in model.named_modules()}


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def test_policy_defaults_off_and_parses_cli_values():
    assert ip.Int8PrefillPolicy().enabled is False
    for value in (None, False, "off"):
        assert ip.Int8PrefillPolicy.from_value(value).enabled is False
    mlp = ip.Int8PrefillPolicy.from_value("mlp")
    assert (mlp.enabled, mlp.scope, mlp.row_threshold) == (True, "mlp", 512)
    assert ip.Int8PrefillPolicy.from_value("all").scope == "all"
    mapped = ip.Int8PrefillPolicy.from_value(
        {"enabled": True, "scope": "all", "row_threshold": 1024, "cache": "ttl"}
    )
    assert mapped.row_threshold == 1024 and mapped.cache == "ttl"
    assert mlp.fidelity is Fidelity.APPROXIMATE
    assert ip.Int8PrefillPolicy().fidelity is Fidelity.EXACT


@pytest.mark.parametrize(
    "value",
    [
        "on",
        True,
        {"scope": "attn"},
        {"enabled": True, "bogus": 1},
        {"enabled": 1},
        {"enabled": True, "row_threshold": 1},
        {"enabled": True, "row_threshold": True},
        {"enabled": True, "cache": "forever"},
        {"enabled": True, "ttl_s": float("inf")},
    ],
)
def test_policy_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        ip.Int8PrefillPolicy.from_value(value)


def test_revision_tracks_numerics_not_memory_lifetime():
    base = ip.Int8PrefillPolicy(enabled=True)
    assert base.revision == ip.Int8PrefillPolicy(enabled=True, cache="ttl").revision
    assert base.revision != ip.Int8PrefillPolicy(enabled=True, scope="all").revision
    assert base.revision != ip.Int8PrefillPolicy(enabled=True, row_threshold=1024).revision


def test_apc_namespace_is_identity_when_off_and_distinct_when_on():
    base = "text-token-v1"
    assert ip.apc_semantic_fingerprint(base, None) == base
    assert ip.apc_semantic_fingerprint(base, ip.Int8PrefillPolicy()) == base
    mlp = ip.apc_semantic_fingerprint(base, ip.Int8PrefillPolicy.from_value("mlp"))
    every = ip.apc_semantic_fingerprint(base, ip.Int8PrefillPolicy.from_value("all"))
    assert mlp != base and every != base and mlp != every
    tenant = ("text-token-v1", "tenant", "a")
    assert ip.apc_semantic_fingerprint(tenant, ip.Int8PrefillPolicy.from_value("mlp"))[0] == tenant


def test_apc_keys_under_int8_do_not_alias_exact_keys():
    from mlx2.runtime.apc_v2 import APCv2

    exact = APCv2.key("m", semantic_fingerprint="text-token-v1")
    approx = APCv2.key(
        "m",
        semantic_fingerprint=ip.apc_semantic_fingerprint(
            "text-token-v1", ip.Int8PrefillPolicy.from_value("mlp")
        ),
    )
    assert exact != approx and hash(approx) is not None


def test_decode_row_bound_fails_closed():
    policy = ip.Int8PrefillPolicy.from_value("mlp")
    assert ip.max_decode_rows(max_lanes=4, config={"num_draft": 3}, speculation="ordinary") == 8
    assert ip.max_decode_rows(max_lanes=4, config={"num_draft": 3}, speculation="self_mtp") == 20
    assert ip.max_decode_rows(
        max_lanes=2, config={}, speculation="prompt_lookup", prompt_lookup_policy={}
    ) == 20
    ip.validate_decode_row_bound(policy, 511)
    with pytest.raises(ip.Int8PrefillError):
        ip.validate_decode_row_bound(policy, 512)
    ip.validate_decode_row_bound(ip.Int8PrefillPolicy(), 10_000)  # disabled: no-op


# --------------------------------------------------------------------------
# device gating
# --------------------------------------------------------------------------


def test_enabled_policy_fails_closed_without_supported_gpu():
    # The suite pins the CPU as default device, which is never eligible.
    model = _Model(bits=4)
    before = _kinds(model)
    with pytest.raises(ip.Int8PrefillError, match="M5-class"):
        ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    assert _kinds(model) == before


def test_device_support_rejects_older_architectures(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"device_name": "Apple M4 Max", "architecture": "applegpu_g16s"}
    )
    ok, reason = ip.device_support()
    assert not ok and "Metal 4" in reason
    monkeypatch.setattr(
        mx, "device_info", lambda: {"device_name": "Apple M5 Max", "architecture": "applegpu_g17s"}
    )
    assert ip.device_support()[0]


def test_disabled_policy_is_inert():
    model = _Model(bits=4)
    before = _kinds(model)
    handle = ip.apply(model, ip.Int8PrefillPolicy())
    assert not handle.active and handle.modules == () and handle.receipt() is None
    assert _kinds(model) == before
    assert ip.remove(handle) is False


# --------------------------------------------------------------------------
# eligibility and scoping (structural, CPU)
# --------------------------------------------------------------------------


def test_module_spec_matrix():
    for bits in (2, 3, 4, 5, 6, 8):
        spec, reason = ip.module_spec("x", _linear(512, 1024, bits=bits))
        assert spec is not None, (bits, reason)
        assert (spec.kind, spec.n, spec.k, spec.bits) == ("quantized", 1024, 512, bits)
    spec, _ = ip.module_spec("x", _linear(512, 1024))
    assert spec.kind == "linear"
    assert ip.module_spec("x", _linear(512, 1024, dtype=mx.float32))[0] is None
    assert ip.module_spec("x", _linear(512, 576, bits=6))[0] is None  # N % 128
    assert ip.module_spec("x", _linear(512, 65536, bits=6))[0] is None  # vocab
    fp4 = nn.QuantizedLinear(512, 1024, bias=False, group_size=32, bits=4, mode="mxfp4")
    spec, reason = ip.module_spec("x", fp4)
    assert spec is None and "affine" in reason
    assert ip.module_spec("x", nn.Embedding(10, 512))[0] is None


def test_mlp_scope_binds_only_mlp_projections(fake_device):
    model = _Model(bits=6)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        assert handle.modules == tuple(
            sorted(
                f"layers.{i}.mlp.{name}"
                for i in range(2)
                for name in ("down_proj", "gate_proj", "up_proj")
            )
        )
        # Router, MTP head, attention and lm_head untouched.
        assert type(model.layers[0].mlp.gate) is nn.Linear
        assert type(model.mtp.up_proj) is nn.QuantizedLinear
        assert type(model.lm_head) is nn.QuantizedLinear
        assert type(model.layers[0].self_attn.q_proj) is nn.QuantizedLinear
        bound = model.layers[0].mlp.up_proj
        assert isinstance(bound, nn.QuantizedLinear) and type(bound) is not nn.QuantizedLinear
    finally:
        ip.remove(handle)


def test_all_scope_adds_attention_but_not_ineligible_or_excluded(fake_device):
    model = _Model(bits=8)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("all"))
    try:
        paths = set(handle.modules)
        assert "layers.0.self_attn.q_proj" in paths
        assert "layers.0.self_attn.k_proj" in paths
        assert "layers.0.self_attn.kv_a_proj" not in paths  # N=576
        assert not any(p.startswith(("lm_head", "mtp")) or p.endswith(".gate") for p in paths)
        assert handle.status()["skipped"] >= 2
    finally:
        ip.remove(handle)


def test_install_is_scoped_to_one_model_and_parameters_unchanged(fake_device):
    target, draft = _Model(bits=4), _Model(bits=4)
    names = [k for k, _ in tree_flatten(target.parameters())]
    draft_kinds = _kinds(draft)
    handle = ip.apply(target, ip.Int8PrefillPolicy.from_value("all"))
    try:
        assert _kinds(draft) == draft_kinds
        assert [k for k, _ in tree_flatten(target.parameters())] == names
    finally:
        ip.remove(handle)


def test_remove_restores_classes_and_allows_reapply(fake_device):
    model = _Model(bits=6)
    before = _kinds(model)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("all"))
    with pytest.raises(ip.Int8PrefillError, match="already"):
        ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    assert ip.remove(handle) is True
    assert _kinds(model) == before
    assert all(ip._STATE_ATTR not in m.__dict__ for _, m in model.named_modules())
    assert ip.remove(handle) is False
    again = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    assert again.active
    ip.remove(again)


def test_scope_selecting_nothing_fails_closed(fake_device):
    model = _Model(bits=6)
    with pytest.raises(ip.Int8PrefillError, match="no eligible"):
        ip.apply(
            model,
            ip.Int8PrefillPolicy.from_value("mlp"),
            select=lambda path, module: False,
        )


def test_sub_threshold_calls_are_stock_and_bit_exact(fake_device):
    mx.random.seed(3)
    model = _Model(hidden=512, inter=1024, bits=6)
    layer = model.layers[0].mlp
    x = mx.random.normal((2, 7, 512)).astype(mx.bfloat16)  # 14 rows: decode/verify
    reference = layer(x)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        got = layer(x)
        assert mx.array_equal(got, reference).item()
        assert handle.counts["fallback_rows"] == 3
        assert handle.counts["engaged_calls"] == 0
    finally:
        ip.remove(handle)


def test_unsupported_dtype_falls_back_even_above_threshold(fake_device):
    model = _Model(hidden=512, inter=1024, bits=4)
    layer = model.layers[0].mlp.up_proj
    x = mx.random.normal((600, 512)).astype(mx.float16)
    reference = layer(x)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        assert mx.array_equal(layer(x), reference).item()
        assert handle.counts["fallback_dtype"] == 1
    finally:
        ip.remove(handle)


# --------------------------------------------------------------------------
# adapter / serving / qualification / telemetry surfaces
# --------------------------------------------------------------------------


class _Adapter:
    def __init__(self, scopes=("mlp",)):
        self.model = _Model(bits=6)
        self._scopes = scopes

    def int8_prefill_supported(self):
        return self._scopes


def test_adapter_must_declare_scope(fake_device):
    adapter = _Adapter(scopes=("mlp",))
    with pytest.raises(ip.Int8PrefillError, match="does not declare"):
        ip.apply_for_adapter(adapter, "all")

    class Silent:
        model = _Model(bits=6)

    with pytest.raises(ip.Int8PrefillError, match="does not declare"):
        ip.apply_for_adapter(Silent(), "mlp")
    handle = ip.apply_for_adapter(adapter, "mlp")
    assert handle.active
    ip.remove(handle)
    with pytest.raises(ValueError):
        ip.adapter_scopes(type("A", (), {"int8_prefill_supported": lambda self: ("attn",)})())


def test_bind_for_serving_records_settings_and_refuses_wide_verify(fake_device):
    adapter = _Adapter()
    handle, settings = ip.bind_for_serving(
        adapter,
        "mlp",
        max_lanes=4,
        config={"num_draft": 3},
        speculation="self_mtp",
    )
    try:
        assert settings["enabled"] and settings["fidelity"] == "approximate"
        assert settings["max_decode_rows"] == 20
        assert settings["revision"] == ip.Int8PrefillPolicy.from_value("mlp").revision
        receipt = handle.receipt()
        assert receipt["fidelity"] == "approximate" and receipt["modules"] == 6
    finally:
        ip.remove(handle)
    with pytest.raises(ip.Int8PrefillError, match="row_threshold"):
        ip.bind_for_serving(
            _Adapter(),
            "mlp",
            max_lanes=64,
            config={"num_draft": 8},
            speculation="self_mtp",
        )


def test_qualification_demands_observed_int8_and_approximate_tier():
    from mlx2.qualification import required_feature_checks

    for speculation in ("ordinary", "self_mtp", "external_draft", "prompt_lookup"):
        settings = {
            "speculation": speculation,
            "mtp": speculation == "self_mtp",
            "int8_prefill": {"enabled": True, "scope": "mlp"},
        }
        assert "feature_int8_prefill" in required_feature_checks(settings)
        settings.pop("int8_prefill")
        assert "feature_int8_prefill" not in required_feature_checks(settings)


def test_server_flag_defaults_off_and_accepts_scopes():
    from mlx2.server import build_parser

    parser = build_parser()
    assert parser.parse_args(["--model", "m"]).int8_prefill == "off"
    assert parser.parse_args(["--model", "m", "--int8-prefill", "all"]).int8_prefill == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", "--int8-prefill", "attn"])


def test_engine_status_and_prometheus_counters(fake_device):
    from mlx2.prometheus import PrometheusBuilder, _add_int8_prefill

    class Engine:
        int8_prefill_policy = ip.Int8PrefillPolicy()
        int8_prefill_handle = None

    off = ip.engine_status(Engine())
    assert off["enabled"] is False and off["fidelity"] == "exact"
    builder = PrometheusBuilder()
    _add_int8_prefill(builder, Engine())
    text = builder.render()
    assert 'mlx2_int8_prefill_enabled{scope="off"} 0' in text
    assert 'mlx2_int8_prefill_calls_total{outcome="engaged"} 0' in text

    engine = Engine()
    engine.int8_prefill_policy = ip.Int8PrefillPolicy.from_value("mlp")
    engine.int8_prefill_handle = ip.apply(_Model(bits=6), engine.int8_prefill_policy)
    try:
        engine.int8_prefill_handle.counts["engaged_calls"] = 5
        engine.int8_prefill_handle.counts["engaged_rows"] = 10240
        status = ip.engine_status(engine)
        assert status["fidelity"] == "approximate" and status["modules"] == 6
        assert status["module_kinds"] == {"q6": 6} and status["experts"] == "stock"
        builder = PrometheusBuilder()
        _add_int8_prefill(builder, engine)
        text = builder.render()
        assert 'mlx2_int8_prefill_enabled{scope="mlp"} 1' in text
        assert 'mlx2_int8_prefill_calls_total{outcome="engaged"} 5' in text
        assert "mlx2_int8_prefill_rows_total 10240" in text
    finally:
        ip.remove(engine.int8_prefill_handle)


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


class _One(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.mlp = layer

    def __call__(self, x):
        return self.mlp(x)


def _rel_cos(y, ref):
    y, ref = y.astype(mx.float32), ref.astype(mx.float32)
    rel = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
    cos = (mx.sum(y * ref) / (mx.linalg.norm(y) * mx.linalg.norm(ref))).item()
    return rel, cos


# Xing4.0 geometry: dense MLP, attention out, MLA q_b / kv_b, shared expert.
_SHAPES = [(3584, 9216), (9216, 3584), (4096, 3584), (768, 6144), (512, 8192), (3584, 1024)]


@pytest.mark.parametrize("bits", [None, 4, 6, 8])
@pytest.mark.parametrize("k,n", _SHAPES)
def test_gpu_int8_prefill_error_bound(gpu, bits, k, n):
    mx.random.seed(k + n + (bits or 0))
    model = _One(_linear(k, n, bits=bits))
    x = mx.random.normal((2048, k)).astype(mx.bfloat16)
    decode = x[:16]
    reference, decode_reference = model(x), model(decode)
    mx.eval(reference, decode_reference)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        got, decode_got = model(x), model(decode)
        mx.eval(got, decode_got)
        rel, cos = _rel_cos(got, reference)
        # Measured ~0.011-0.013 relative L2 on Gaussian activations.
        assert rel < 0.025 and cos > 0.9995, (rel, cos)
        assert mx.array_equal(decode_got, decode_reference).item()
        assert handle.counts["engaged_calls"] == 1
        assert handle.counts["engaged_rows"] == 2048
        assert handle.counts["fallback_rows"] == 1
    finally:
        ip.remove(handle)
    assert mx.array_equal(model(x), reference).item()


def test_gpu_probe_and_bias_and_3d_input(gpu):
    assert ip.require_supported_device()
    mx.random.seed(11)
    model = _One(_linear(1024, 2048, bias=True))
    x = mx.random.normal((2, 400, 1024)).astype(mx.bfloat16)  # 800 rows
    reference = model(x)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        got = model(x)
        assert got.shape == reference.shape and got.dtype == mx.bfloat16
        rel, cos = _rel_cos(got, reference)
        assert rel < 0.03 and cos > 0.999, (rel, cos)
        # bf16 Linear under the default "auto" cache keeps a resident copy.
        assert handle.weight_bytes() == 2048 * 1024 + 2048 * 4
    finally:
        ip.remove(handle)
    assert handle.weight_bytes() == 0


def test_gpu_shared_activation_is_quantized_once(gpu):
    model = _Model(hidden=512, inter=1024, bits=6, layers=1)
    mlp = model.layers[0].mlp
    x = mx.random.normal((1024, 512)).astype(mx.bfloat16)
    handle = ip.apply(model, ip.Int8PrefillPolicy.from_value("mlp"))
    try:
        mx.eval(mlp(x))
        # gate_proj and up_proj share ``x``; down_proj has its own input.
        assert handle.counts["activation_reuse"] == 1
        assert handle.counts["engaged_calls"] == 3
    finally:
        ip.remove(handle)


def test_gpu_ttl_cache_builds_once_and_evicts(gpu):
    model = _One(_linear(1024, 2048, bits=6))
    x = mx.random.normal((1024, 1024)).astype(mx.bfloat16)
    handle = ip.apply(model, {"enabled": True, "scope": "mlp", "cache": "ttl", "ttl_s": 60})
    try:
        first = model(x)
        second = model(x * 1)
        mx.eval(first, second)
        assert handle.counts["weight_builds"] == 1
        assert handle.weight_bytes() == 2048 * 1024 + 2048 * 4
        assert handle.evict_weights() == 1 and handle.weight_bytes() == 0
        assert mx.array_equal(model(x), first).item()  # deterministic rebuild
    finally:
        ip.remove(handle)


# --------------------------------------------------------------------------
# serving engine integration (CPU; sub-threshold prompts stay on stock path)
# --------------------------------------------------------------------------


def _serving_model(hidden=256, inter=512):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        full_attention_interval=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=256,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)  # bf16 nn.Linear projections are eligible
    model.eval()
    mx.eval(model.parameters())
    return model


def _serving_engine(model, scopes, **kwargs):
    from test_approximate_kv_serving import make_adapter

    from mlx2.serving import ServingEngine

    adapter = make_adapter(model, operations=None)
    if scopes is not None:
        adapter.int8_prefill_supported = lambda self: scopes
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False, **kwargs
    )
    return engine


@pytest.fixture
def serving_host(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def test_serving_default_off_records_nothing(serving_host):
    engine = _serving_engine(_serving_model(), None)
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "int8_prefill" not in status["settings"]
        assert status["int8_prefill"]["enabled"] is False
        assert engine.int8_prefill_handle is None
    finally:
        engine.close()


def test_serving_int8_fails_closed_on_cpu_device(serving_host):
    engine = _serving_engine(_serving_model(), ("mlp",), int8_prefill="mlp")
    try:
        engine.thread.join(60)
        assert not engine.ready.is_set()
        assert "M5-class" in engine.error
    finally:
        engine.close()


def test_serving_int8_refuses_undeclared_scope(serving_host, fake_device):
    engine = _serving_engine(_serving_model(), ("mlp",), int8_prefill="all")
    try:
        engine.thread.join(60)
        assert not engine.ready.is_set()
        assert "does not declare int8 prefill scope 'all'" in engine.error
    finally:
        engine.close()


def test_serving_int8_binds_settings_receipts_and_namespace(serving_host, fake_device):
    from test_approximate_kv_serving import run

    from mlx2.runtime import apc_v2

    keys = []
    original = apc_v2.APCv2.key

    def spy(*args, **kwargs):
        key = original(*args, **kwargs)
        keys.append(key)
        return key

    apc_v2.APCv2.key = staticmethod(spy)
    model = _serving_model()
    engine = _serving_engine(model, ("mlp",), int8_prefill="mlp")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        recorded = status["settings"]["int8_prefill"]
        assert recorded["enabled"] and recorded["scope"] == "mlp"
        assert recorded["fidelity"] == "approximate"
        assert recorded["revision"] == ip.Int8PrefillPolicy.from_value("mlp").revision
        assert status["int8_prefill"]["active"] and status["int8_prefill"]["modules"] > 0
        _, receipt = run(engine, list(range(1, 20)), max_tokens=3)
        assert receipt["int8_prefill"]["fidelity"] == "approximate"
        # Short prompt: every call was below the threshold (stock numerics).
        counts = engine.int8_prefill_handle.counts
        assert counts["engaged_calls"] == 0 and counts["fallback_rows"] > 0
        assert keys and all(
            k.semantic_fingerprint
            == ip.apc_semantic_fingerprint("text-token-v1", ip.Int8PrefillPolicy.from_value("mlp"))
            for k in keys
        )
    finally:
        apc_v2.APCv2.key = staticmethod(original)
        engine.close()
    # Worker shutdown removes the install from the served model.
    assert not any(
        ip._STATE_ATTR in m.__dict__ for _, m in model.named_modules()
    )
