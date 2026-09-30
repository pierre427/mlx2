"""CPU contracts for the candidate; Metal execution has an explicit harness."""

from collections import Counter

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import gated_delta as gdn
from mlx2.runtime.models import tensorfold_prefill as prefill
from mlx2.runtime.prefill_plan import prefill_rows, recurrence_segments


def quantized(k=64, n=33, bias=False):
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = layer.weight.astype(mx.bfloat16)
    if bias:
        layer.bias = layer.bias.astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4)


@pytest.mark.parametrize("alignment", [8, 16])
def test_scan_partition_covers_every_token_and_avoids_short_tail(alignment):
    for length in [*range(1050), 4097, 8192, 8193, 32769]:
        spans = list(recurrence_segments(length, alignment=alignment))
        assert sum(b - a for a, b in spans) == length
        assert all(a == (spans[i - 1][1] if i else 0) for i, (a, b) in enumerate(spans))
        assert all(0 < b - a <= 256 for a, b in spans)
        if length >= 17:
            assert all(b - a >= 17 for a, b in spans)
        assert not spans or spans[-1][1] == length


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": -1},
        {"length": True},
        {"length": 33, "alignment": 7},
        {"length": 33, "maximum": 17},
    ],
)
def test_invalid_scan_geometry(kwargs):
    with pytest.raises(ValueError):
        list(recurrence_segments(**kwargs))


@pytest.mark.parametrize(
    "shape,rows",
    [
        ((1, 18, 64), 18),
        ((4, 65, 64), 260),
        ((32, 1, 64), 0),
        ((2, 17, 64), 0),
        ((128, 64), 0),
        ((0, 65, 64), 0),
    ],
)
def test_sequence_axis_controls_admission(shape, rows):
    assert prefill_rows(shape) == rows


def test_policy_is_explicit_and_receipted():
    default = FlashNextPolicy().as_dict()
    assert "tensorfold_prefill" not in default and "gdn_prefill_chunk" not in default
    candidate = FlashNextPolicy(tensorfold_prefill=True, gdn_prefill_chunk=8)
    assert candidate.as_dict()["tensorfold_prefill"] is True
    assert candidate.as_dict()["gdn_prefill_chunk"] == 8
    assert "MLX_GDN_CORE" not in candidate.environment()
    for values in (
        {"tensorfold_prefill": 1},
        {"gdn_prefill_chunk": True},
        {"gdn_prefill_chunk": 32},
        {"gdn_prefill_chunk": 8, "gdn_core": True},
    ):
        with pytest.raises(ValueError):
            FlashNextPolicy.from_mapping(values)


def test_prefill_cache_namespace_changes_with_the_numerical_law():
    from mlx2.runtime.prefill_plan import apc_prefill_fingerprint, execution_identity

    projection = {
        "kernel": "native",
        "installed": 4,
        "names_sha256": "abc",
        "tile": [32, 64, 64],
        "counters": {},
    }
    scan = {"chunk_size": 8, "segment_max_rows": 2048, "layers": 48, "counters": {}}
    plain = "text-token-v1"
    assert apc_prefill_fingerprint(plain, execution_identity()) is plain
    identity = execution_identity(projection, scan)
    first = apc_prefill_fingerprint(plain, identity)
    projection["counters"]["calls"] = 99
    assert first == apc_prefill_fingerprint(plain, execution_identity(projection, scan))
    assert first != apc_prefill_fingerprint(
        plain, execution_identity(projection, {**scan, "chunk_size": 16})
    )
    assert first != apc_prefill_fingerprint(
        plain, execution_identity(projection, {**scan, "segment_max_rows": 512})
    )
    assert first != apc_prefill_fingerprint(
        plain, execution_identity({**projection, "kernel": "metal"}, scan)
    )
    assert first != apc_prefill_fingerprint((plain, "tenant", "two"), identity)


@pytest.mark.parametrize(
    "shape,speculating,expected",
    [
        ((2, 65, 64), False, True),
        ((32, 1, 64), False, False),
        ((1, 128, 64), True, False),
        ((1, 17, 64), False, False),
    ],
)
def test_decoder_scope_excludes_wide_verification_and_resets(
    shape, speculating, expected
):
    from types import SimpleNamespace

    class Decoder:
        def __call__(self, x, mask=None, cache=None):
            assert prefill._PREFILL_SCOPE.get() is expected
            if mask == "fail":
                raise RuntimeError("injected")
            return x

    decoder = prefill._scoped_decoder_type(Decoder)()
    x, cache = SimpleNamespace(shape=shape), SimpleNamespace(speculating=speculating)
    assert decoder(x, cache=cache) is x
    assert not prefill._PREFILL_SCOPE.get()
    with pytest.raises(RuntimeError, match="injected"):
        decoder(x, "fail", cache)
    assert not prefill._PREFILL_SCOPE.get()


def test_packed_native_group_replaces_weights_with_views_and_invalidates_on_reload():
    modules = [quantized(n=33, bias=True), quantized(n=7)]
    x = mx.random.normal((2, 19, 64)).astype(mx.bfloat16)
    old = [
        (mx.array(m.weight), mx.array(m.scales), mx.array(m.biases)) for m in modules
    ]
    reference = [m(x) for m in modules]
    group = prefill.PackedProjectionGroup(modules)
    assert group.matches(modules)
    for m, original in zip(modules, old):
        for key, value in zip(("weight", "scales", "biases"), original):
            assert mx.array_equal(getattr(m, key), value).item()
    assert group.weight.nbytes == sum(m.weight.nbytes for m in modules)
    for a, b in zip(group(x, modules), reference):
        assert mx.array_equal(a, b).item()
    modules[0].weight = mx.zeros_like(modules[0].weight)
    assert not group.matches(modules)


def test_gdn_candidate_policy_rejects_dead_and_invalid_knobs():
    for settings in (
        {"gdn_prefill_segment_rows": 256},
        {"gdn_prefill_chunk": 8, "gdn_prefill_segment_rows": 17},
        {"tensorfold_prefill_backend": "metal"},
        {"tensorfold_prefill": True, "tensorfold_prefill_backend": "guess"},
    ):
        with pytest.raises(ValueError):
            FlashNextPolicy.from_mapping(settings)
    policy = FlashNextPolicy(
        tensorfold_prefill=True,
        tensorfold_prefill_backend="metal",
        gdn_prefill_chunk=16,
        gdn_prefill_segment_rows=512,
    )
    assert policy.as_dict()["gdn_prefill_segment_rows"] == 512


def test_install_preserves_weights_reference_decode_and_expert_ownership():
    model = nn.Module()
    model.dense = quantized()
    model.switch_mlp = nn.Module()
    model.switch_mlp.dense = quantized()
    model.mtp = nn.Module()
    model.mtp.dense = quantized()
    model.eval()
    x = mx.ones((32, 1, 64), dtype=mx.bfloat16)
    reference = model.dense(x)
    weight = model.dense.weight
    keys = set(dict(model.named_modules()))
    receipt = prefill.install(model, backend="metal")
    assert receipt["installed"] == 1
    assert model.dense.weight is weight
    assert set(dict(model.named_modules())) == keys
    assert type(model.switch_mlp.dense) is nn.QuantizedLinear
    assert type(model.mtp.dense) is nn.QuantizedLinear
    assert mx.array_equal(model.dense(x), reference).item()
    assert receipt["counters"]["reference_calls"] == 1


@pytest.mark.parametrize(
    "batch,rows,widths,biases",
    [
        (1, 19, (7,), (False,)),
        (4, 65, (33, 48, 64, 7), (True, False, True, False)),
    ],
)
def test_launch_geometry_and_outputs_with_cpu_quantized_oracle(
    monkeypatch, batch, rows, widths, biases
):
    modules = [quantized(n=n, bias=b) for n, b in zip(widths, biases)]
    x = mx.random.normal((batch, rows, 64)).astype(mx.bfloat16)
    expected = [m(x) for m in modules]
    seen = {}

    def kernel(ws, bs):
        assert ws == widths and bs == biases

        def call(**kwargs):
            seen.update(kwargs)
            inputs = kwargs["inputs"]
            assert inputs[1] == batch * rows
            i, ys = 2, []
            for n, biased in zip(widths, biases):
                w, s, b = inputs[i : i + 3]
                y = mx.quantized_matmul(
                    inputs[0],
                    w,
                    scales=s,
                    biases=b,
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
                i += 3
                if biased:
                    y = y + inputs[i]
                    i += 1
                ys.append(y)
            assert i == len(inputs)
            return ys

        return call

    monkeypatch.setattr(prefill, "_kernel", kernel)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M5 Max"})
    actual = prefill.project(x, modules)
    assert seen["grid"] == (
        128 * ((batch * rows + 31) // 32),
        sum((n + 63) // 64 for n in widths),
        1,
    )
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_projection_rejects_wrong_format_and_requires_metal():
    m = quantized()
    with pytest.raises(ValueError, match="geometry"):
        prefill.project(mx.ones((19, 128), dtype=mx.bfloat16), [m])
    with pytest.raises(RuntimeError, match="Metal"):
        prefill.project(mx.ones((19, 64), dtype=mx.bfloat16), [m])
    m.scales = m.scales.astype(mx.float32)
    assert not prefill.eligible(m)


@pytest.mark.parametrize(
    "biases", [(False, False), (True, True), (True, False), (False, True)]
)
def test_swiglu_fused_launch_has_one_final_output(monkeypatch, biases):
    from mlx2.runtime.models.activations import swiglu

    gate, up = (quantized(n=33, bias=biased) for biased in biases)
    x = mx.ones((2, 19, 64), dtype=mx.bfloat16)
    expected = swiglu(gate(x), up(x))
    seen = {}

    def kernel(width, bs):
        assert width == 33 and bs == biases

        def launch(**kwargs):
            seen.update(kwargs)
            inputs = kwargs["inputs"]
            outputs = []
            bias_index = 8
            for i, biased in enumerate(biases):
                w, s, b = inputs[2 + 3 * i : 5 + 3 * i]
                y = mx.quantized_matmul(
                    inputs[0],
                    w,
                    scales=s,
                    biases=b,
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
                if biased:
                    y = y + inputs[bias_index]
                    bias_index += 1
                outputs.append(y)
            assert bias_index == len(inputs)
            return [swiglu(*outputs)]

        return launch

    monkeypatch.setattr(prefill, "_swiglu_kernel", kernel)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M5 Max"})
    actual = prefill.project_swiglu(x, gate, up)
    assert mx.array_equal(actual, expected).item()
    assert seen["output_shapes"] == [(38, 33)]


def test_mlp_hook_uses_fused_gate_up_only_in_prefill(monkeypatch):
    from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

    mlp = Qwen3NextMLP(64, 128)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = (
        quantized(n=128),
        quantized(n=128),
        quantized(k=128, n=64),
    )
    mlp.eval()
    x = mx.random.normal((1, 19, 64)).astype(mx.bfloat16)
    expected = mlp(x)
    receipt = prefill.install(mlp, backend="metal")
    monkeypatch.setattr(prefill, "_admitted", lambda x: bool(prefill_rows(x.shape)))
    calls = []

    def fused(x, gate, up):
        from mlx2.runtime.models.activations import swiglu

        calls.append(x.shape)
        # Invoke the preserved reference methods, not the new Metal candidate.
        return swiglu(
            nn.QuantizedLinear.__call__(gate, x), nn.QuantizedLinear.__call__(up, x)
        )

    monkeypatch.setattr(prefill, "project_swiglu", fused)
    monkeypatch.setattr(
        prefill,
        "project",
        lambda x, ms: tuple(nn.QuantizedLinear.__call__(m, x) for m in ms),
    )
    assert mx.array_equal(mlp(x), expected).item()
    mlp(x[:, :1])
    assert calls == [(1, 19, 64)]
    assert receipt["counters"]["swiglu_intermediate_elements_avoided"] == 2 * 19 * 128


def test_scan_option_cannot_change_rollback_or_verify_calls(monkeypatch):
    from types import SimpleNamespace

    from mlx2.runtime.models import qwen3_5, qwen4_exp

    layer = SimpleNamespace(A_log=None, dt_bias=None, _prefill_scan_chunk=8)
    for module in (qwen3_5, qwen4_exp):
        seen = []
        monkeypatch.setattr(
            module, "gated_delta_update", lambda *args, seen=seen, **kw: seen.append(kw)
        )
        fn = module.GatedDeltaNet._gated_delta_update
        fn(layer, None, None, None, None, None, None, None, True)
        fn(layer, None, None, None, None, None, None, None, True, prefill=True)
        assert [kw["prefill_chunk_size"] for kw in seen] == [0, 8]


def test_scan_carry_matches_the_actual_delta_recurrence_on_cpu(monkeypatch):
    # Small math oracle checks state carry independently of the stub above.
    q = mx.random.normal((2, 273, 1, 2)) * 0.1
    k = mx.random.normal((2, 273, 1, 2)) * 0.1
    v = mx.random.normal((2, 273, 1, 3))
    gate, beta = mx.full((2, 273, 1), 0.98), mx.full((2, 273, 1), 0.5)
    state = mx.ones((2, 1, 3, 2)) * 0.2
    monkeypatch.setattr(gdn, "_core_layout_supported", lambda *args: True)

    def core(q, k, v, g, beta, *, initial_state, **kw):
        return gdn.gated_delta_ops(q, k, v, g, beta, initial_state)

    monkeypatch.setattr(gdn, "_core_gated_delta_update", core)
    actual = gdn._chunked_prefill(q, k, v, gate, beta, state, None, 8, Counter())
    expected = gdn.gated_delta_ops(q, k, v, gate, beta, state)
    for a, b in zip(actual, expected):
        assert mx.array_equal(a, b).item()


def test_grouped_projection_fuses_only_prefill_and_preserves_order(monkeypatch):
    from mlx2.runtime.models.qwen3_5 import GatedDeltaNet

    layer = nn.Module()
    for name, n in zip(
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), (64, 32, 8, 8)
    ):
        setattr(layer, name, quantized(n=n))
    layer.eval()
    counts = Counter()
    object.__setattr__(layer, "_prefill_counts", counts)
    monkeypatch.setattr(prefill, "_admitted", lambda x: bool(prefill_rows(x.shape)))
    calls = []

    def project(x, modules):
        calls.append(tuple(modules))
        return tuple(m(x) for m in modules)

    monkeypatch.setattr(prefill, "project", project)
    x = mx.ones((2, 33, 64), dtype=mx.bfloat16)
    ys = GatedDeltaNet._input_projections(layer, x)
    assert [y.shape[-1] for y in ys] == [64, 32, 8, 8]
    assert len(calls) == 1 and counts["projection_launches_saved"] == 3
    assert prefill.grouped_input(layer, x[:, :1]) is None


def test_scan_continuation_consumes_each_segment_and_carries_state(monkeypatch):
    # A cheap affine recurrence with nonzero initial state exposes a missing,
    # repeated or reset segment. This tests orchestration, not the Metal math.
    q = mx.arange(513, dtype=mx.float32).reshape(1, 513, 1, 1)
    v = q + 1
    state = mx.array([3.0])
    stats, seen = Counter(), []
    monkeypatch.setattr(gdn, "_core_layout_supported", lambda *a: True)

    def core(q, k, v, g, beta, *, initial_state, stream, chunk_size):
        seen.append((int(q[0, 0, 0, 0].item()), q.shape[1], initial_state.item()))
        y = mx.cumsum(v, axis=1) + initial_state
        return y, y[:, -1].reshape(1)

    monkeypatch.setattr(gdn, "_core_gated_delta_update", core)
    gate = mx.ones((1, 513, 1))
    y, end = gdn._chunked_prefill(q, q, v, gate, gate, state, None, 8, stats)
    expected = mx.cumsum(v, axis=1) + state
    assert mx.array_equal(y, expected).item()
    assert end.item() == expected[:, -1].item()
    assert [n for _, n, _ in seen] == [256, 240, 17]
    assert stats == {"calls": 1, "segments": 3, "tokens": 513}
    # Warm continuation of the returned state, just as an APC boundary resumes.
    _, resumed = gdn._chunked_prefill(
        q[:, :33], q[:, :33], v[:, :33], gate[:, :33], gate[:, :33], end, None, 8, stats
    )
    assert resumed.item() == (end + v[:, :33].sum()).item()


@pytest.mark.parametrize(
    "case,reason",
    [
        ("mask", "masked_or_vector_gate"),
        ("short", "short_sequence"),
        ("layout", "unsupported_layout"),
        ("m3", "chunk16_requires_m5"),
    ],
)
def test_scan_refusal_does_not_call_core_or_mutate_state(monkeypatch, case, reason):
    n = 8 if case == "short" else 33
    q = mx.zeros((1, n, 1, 1))
    gate = mx.ones((1, n, 1))
    state, stats = mx.array([7.0]), Counter()
    monkeypatch.setattr(gdn, "_core_layout_supported", lambda *a: case != "layout")
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Pro"})

    def forbidden(*a, **kw):
        raise AssertionError("refused scan reached the primitive")

    monkeypatch.setattr(gdn, "_core_gated_delta_update", forbidden)
    result = gdn._chunked_prefill(
        q,
        q,
        q,
        gate,
        gate,
        state,
        mx.ones((1, n)) if case == "mask" else None,
        16 if case == "m3" else 8,
        stats,
    )
    assert result is None and state.item() == 7.0
    assert stats["fallback_" + reason] == 1


def test_native_install_does_not_wrap_unrelated_projections():
    from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

    model = nn.Module()
    model.mlp = Qwen3NextMLP(64, 128)
    model.other = quantized()
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(
            model.mlp,
            name,
            quantized(n=128) if name != "down_proj" else quantized(k=128, n=64),
        )
    receipt = prefill.install(model)
    assert receipt["installed"] == 2
    assert type(model.other) is nn.QuantizedLinear
    assert type(model.mlp.down_proj) is nn.QuantizedLinear
    assert type(model.mlp.gate_proj) is nn.QuantizedLinear
    assert not hasattr(model.other, "_prefill_counts")


def test_native_install_rejects_a_model_without_supported_groups():
    model = nn.Module()
    model.linear = quantized()
    with pytest.raises(ValueError, match="no eligible"):
        prefill.install(model)
    assert type(model.linear) is nn.QuantizedLinear
