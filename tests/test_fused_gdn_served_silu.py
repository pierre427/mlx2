"""Fused GDN kernels follow the served SiLU (omlx#4122).

The kernels reproduce compiled ``nn.silu`` with a copied sigmoid. MLX #4461
(v0.32.3) serves ``metal::precise::exp`` instead of ``metal::exp``; the decode
kernels compile the served spelling, a probe that matches neither makes the
fused paths fall back, and every verify-family kernel must compile the decode
kernel's header (a verify row is its one-token decode step).  CPU only: the
probe itself needs Metal and runs only in the opt-in test; everything else
injects its outcome.
"""

import os
import re

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_fused_gdn as fused
from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
from mlx2.runtime.models import qwen4_fused_gdn_verify as verify


def test_all_16bit_encodings_cover_every_bit_pattern():
    for dtype, nans in ((mx.bfloat16, 254), (mx.float16, 2046)):
        values = fused.all_16bit_encodings(dtype)
        assert values.shape == (1 << 16,) and values.dtype == dtype
        assert (np.array(values.view(mx.uint16)) == np.arange(1 << 16)).all()
        assert int(mx.isnan(values).sum().item()) == nans
        assert int(mx.isinf(values).sum().item()) == 2
    with pytest.raises(ValueError):
        fused.all_16bit_encodings(mx.float32)


def test_same_bits_is_exact_except_that_nan_matches_nan():
    values = fused.all_16bit_encodings(mx.bfloat16)
    assert fused.same_bits(values, values)
    quiet = mx.array([0x7FC0, 0xFFC1], dtype=mx.uint16).view(mx.bfloat16)
    other = mx.array([0x7FC1, 0x7FC0], dtype=mx.uint16).view(mx.bfloat16)
    assert fused.same_bits(quiet, other)
    # One ULP on one finite value is a mismatch; allclose would accept it.
    bits = np.array(values.view(mx.uint16))
    bits[1000] += 1
    assert not fused.same_bits(values, mx.array(bits).view(mx.bfloat16))
    assert not fused.same_bits(mx.array([0.0]), mx.array([-0.0]))
    wide = mx.array([1.0, float("nan")])
    assert fused.same_bits(wide, mx.array([1.0, float("nan")]))
    assert not fused.same_bits(wide, wide.astype(mx.bfloat16))


def test_selector_takes_the_matching_spelling_or_none():
    both = {"metal::exp": True, "metal::precise::exp": True}
    assert fused.select_silu_exp(both) == "metal::exp"
    precise = {"metal::exp": False, "metal::precise::exp": True}
    assert fused.select_silu_exp(precise) == "metal::precise::exp"
    assert fused.select_silu_exp({"metal::exp": False}) is None
    assert fused.select_silu_exp({}) is None


@pytest.fixture
def probe(monkeypatch):
    """Reset the per-process probe and install an injected outcome."""
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", False)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", None)
    calls = []

    def install(outcome):
        def injected():
            calls.append(outcome)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(fused, "_probe_served_silu", injected)
        return calls

    return install


@pytest.mark.parametrize(
    ("outcome", "refusal"),
    [
        ({"metal::exp": True, "metal::precise::exp": False}, None),
        (
            {"metal::exp": False, "metal::precise::exp": True},
            None,
        ),
        ({"metal::exp": False, "metal::precise::exp": False},
         "served SiLU matches no kernel form"),
        (RuntimeError("kernel failed to compile"), "served SiLU matches no kernel form"),
    ],
)
def test_refusal_follows_one_probe_per_process(probe, outcome, refusal):
    calls = probe(outcome)
    assert fused.served_silu_refusal() == refusal
    assert fused.served_silu_refusal() == refusal
    assert len(calls) == 1


class _Cache(list):
    speculating = False

    def rollback_spans(self, steps, mask):
        return ()

    def record_rollback(self, *args, **kwargs):
        pass

    def advance(self, steps):
        pass


def _accept(*args, **kwargs):
    return fused.FusedGdnAdmission(True, "eligible")


def _never(*args, **kwargs):
    raise AssertionError("a kernel ran although the served SiLU refused it")


def _inputs(width):
    return (mx.zeros((1, width, 8)), mx.zeros((1, width, 8)),
            mx.zeros((1, width, 4)), mx.zeros((1, width, 4)))


REFUSAL = "served SiLU uses metal::precise::exp"


def test_qwen36_fused_decode_declines_before_any_kernel(monkeypatch):
    from mlx2.runtime.models import qwen36_35b
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5_moe_text", hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=128,
    )
    layer = qwen36_35b.GatedDeltaNet(args)
    layer.set_fused_gdn_decode_mode("fused")
    monkeypatch.setattr(qwen36_35b, "admit_qwen4_fused_gdn_decode", _accept)
    monkeypatch.setattr(qwen36_35b, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen36_35b, "served_silu_refusal", lambda: REFUSAL)
    monkeypatch.setattr(qwen36_35b, "probe_qwen36_gdn", _never)
    monkeypatch.setattr(qwen36_35b, "qwen4_fused_gdn_decode", _never)
    cache = _Cache([mx.zeros((1, 2, 8)), mx.zeros((1, 4, 8, 8))])
    assert layer._try_fused_decode(*_inputs(1), None, cache) is None
    assert layer.fused_gdn_decode_last_fallback == REFUSAL
    assert layer.fused_gdn_decode_fallback_reasons == {REFUSAL: 1}


def test_flash_next_fused_paths_decline_before_any_kernel(monkeypatch):
    from mlx2.runtime.models import qwen4_exp
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    model, _vocab = tiny_qwen4_mtp()
    layer = next(
        layer.linear_attn
        for layer in model.language_model.model.layers
        if getattr(layer, "linear_attn", None) is not None
    )
    layer.set_fused_gdn_decode_mode("fused")
    layer.set_fused_gdn_verify_mode("fused")
    layer.set_fused_gdn_prefill_mode("fused")
    for name in ("admit_qwen4_fused_gdn_decode", "admit_qwen4_fused_gdn_verify"):
        monkeypatch.setattr(qwen4_exp, name, _accept)
    monkeypatch.setattr(prefill, "admit_qwen4_fused_gdn_prefill", _accept)
    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: REFUSAL)
    for name in (
        "probe_qwen4_fused_gdn_decode", "probe_qwen4_fused_gdn_verify",
        "probe_qwen4_fused_gdn_replay_verify", "probe_qwen4_fused_gdn_catchup",
        "qwen4_fused_gdn_decode", "qwen4_fused_gdn_verify",
    ):
        monkeypatch.setattr(qwen4_exp, name, _never)
    monkeypatch.setattr(prefill, "qwen4_gdn_prefill_prework", _never)
    state = [mx.zeros((1, 3, 8)), mx.zeros((1, 4, 8, 8))]

    assert layer._try_fused_decode(*_inputs(1), None, _Cache(state)) is None
    assert layer.fused_gdn_decode_last_fallback == REFUSAL
    assert layer._try_fused_verify(*_inputs(3), None, _Cache(state)) is None
    assert layer.fused_gdn_verify_last_fallback == REFUSAL
    assert layer._try_fused_prefill(*_inputs(64), None, _Cache(state)) is None
    assert layer.fused_gdn_prefill_last_fallback == REFUSAL


def _conv_silu_spellings(source):
    """Exp spellings of the conv SiLU sites, which copy compiled nn.silu."""
    return re.findall(r"T\(1\) / \(T\(1\) \+ (metal::(?:precise::)?exp)\(metal::abs\(conv\)\)\)",
                      source)


def test_kernel_sources_keep_one_silu_contract():
    """The probe checks ``mlx_sigmoid_fast``; every conv SiLU site must use it
    (or its exact spelling), and the eager-sigmoid sites stay precise."""
    header = fused._HEADER
    fast = header[header.index("mlx_sigmoid_fast(U x)"):]
    fast = fast[: fast.index("}")]
    assert f"{fused.KERNEL_SILU_EXP}(metal::abs(x))" in fast
    for source in (fused._SOURCE, fused._SOURCE_OUTPROJ, verify._SOURCE):
        assert "T sl = xb * mlx_sigmoid_fast(xb);" in source
        # beta: eager mx.sigmoid, precise on every MLX build
        assert "mlx_sigmoid_precise(b[" in source
    assert _conv_silu_spellings(prefill.prefill_prework_source()) == [
        fused.KERNEL_SILU_EXP
    ]
    # Qwen4's output gate is eager mx.sigmoid: precise on every MLX build.
    assert "metal::precise::exp(metal::abs(zv))" in prefill.prefill_norm_gate_source()


@pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal probe",
)
def test_metal_probe_selects_the_kernel_form_on_the_pinned_build(monkeypatch):
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", False)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", None)
    with mx.stream(mx.gpu):
        matches = fused._probe_served_silu()
    # MLX without #4461 serves the fast sigmoid; with it, the precise one.
    assert fused.select_silu_exp(matches) in fused._SILU_EXP_CANDIDATES
    assert matches["metal::exp"] != matches["metal::precise::exp"]


def test_device_fault_in_served_silu_probe_is_raised_not_cached(probe):
    calls = probe(
        RuntimeError(
            "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
            "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
        )
    )
    with pytest.raises(RuntimeError):
        fused.served_silu_exp()
    assert not fused._SERVED_SILU_COMPLETE
    with pytest.raises(RuntimeError):
        fused.served_silu_exp()
    assert len(calls) == 2



_VERIFY_BUILDERS = (
    (verify._kernel, ((False, False), (True, False), (False, True), (True, True))),
    (verify._catchup_kernel, ((False, False), (True, False), (False, True), (True, True))),
    (verify._replay_verify_kernel, ((False, False), (True, False), (False, True), (True, True))),
    (verify._batch_kernel, ((False, False), (True, False), (False, True), (True, True))),
    (verify._batch_replay_kernel, ((False,), (True,))),
)


@pytest.mark.parametrize("spelling", fused._SILU_EXP_CANDIDATES)
def test_every_verify_kernel_compiles_the_decode_kernels_silu(monkeypatch, spelling):
    """A verify row must be its one-token decode step bit for bit, so every
    verify-family kernel compiles the header the decode kernel compiles.

    Regression (2026-10-17): the MLX 0.32.4 repin respelled only the decode
    kernels' SiLU to the served ``metal::precise::exp``; snapshot verify,
    compact replay, catch-up and the batched verify kept ``metal::exp``, and
    Flash-Next row-exact verify rows differed from decode in GDN layers
    (one bf16 conv output near -6.84 rounds apart) while receipts said
    ``row_exact``.
    """
    built = []

    def capture(**kwargs):
        built.append(kwargs)
        return object()

    builders = (fused._kernel, fused._kernel_batch) + tuple(b for b, _ in _VERIFY_BUILDERS)
    for builder in builders:
        builder.cache_clear()
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", True)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", spelling)
    monkeypatch.setattr(mx.fast, "metal_kernel", capture)
    try:
        fused._kernel(False)
        decode = built[-1]
        assert decode["header"] == fused._served_silu_header(spelling)
        if spelling != fused.KERNEL_SILU_EXP:
            assert decode["header"] != fused._HEADER
        names = set()
        for builder, variants in _VERIFY_BUILDERS:
            for args in variants:
                builder(*args)
                kernel = built[-1]
                assert kernel["header"] == decode["header"], (builder.__name__, args)
                # A distinct name per spelling: MLX caches libraries by name.
                assert kernel["name"].endswith(decode["name"].rsplit("_", 1)[-1])
                names.add(kernel["name"])
        assert len(names) == sum(len(v) for _, v in _VERIFY_BUILDERS)
    finally:
        for builder in builders:
            builder.cache_clear()


# -- the kernels that still hard-coded metal::exp after the 0.32.4 repin ------
# (2026-10-17): the Qwen3.8 tree/cohort GDN pre kernel, the fused prefill
# prework and Qwen3.5 swish norm gate, and the disabled decode+out-projection.
# Their Metal value sweeps live in tests/test_gdn_exp_spelling_metal.py.


def _set_served(monkeypatch, spelling):
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", True)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", spelling)


@pytest.mark.parametrize("spelling", fused._SILU_EXP_CANDIDATES)
def test_outproj_kernel_compiles_the_decode_kernels_silu(monkeypatch, spelling):
    """The decode+out-projection kernel's GDN core is the decode step, so it
    compiles the decode header (it compiled the raw ``metal::exp`` _HEADER),
    and it stays disabled."""
    _set_served(monkeypatch, spelling)
    spec = fused._outproj_kernel_spec()
    header, suffix = fused._served_decode_header()
    assert spec["header"] == header == fused._served_silu_header(spelling)
    assert spec["name"].endswith(suffix)
    if spelling != fused.KERNEL_SILU_EXP:
        assert spec["header"] != fused._HEADER
    fused._kernel_outproj.cache_clear()
    with pytest.raises(RuntimeError, match="fused_outproj is disabled"):
        fused._kernel_outproj()


def test_outproj_spec_refuses_without_a_served_spelling(monkeypatch):
    _set_served(monkeypatch, None)
    with pytest.raises(RuntimeError, match="served SiLU matches no kernel form"):
        fused._outproj_kernel_spec()


def _gate_spelling(monkeypatch, gate, dtype, spelling):
    monkeypatch.setitem(gate._spellings, str(dtype), spelling)


@pytest.mark.parametrize("spelling", fused._SILU_EXP_CANDIDATES)
def test_tree_pre_kernel_compiles_the_served_silu(monkeypatch, spelling):
    from mlx2.runtime.models import qwen38_tree_gdn as tree

    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(kw) or object())
    monkeypatch.setattr(tree, "_kernel_cache", {})
    monkeypatch.setattr(tree, "_PARITY", {spelling: None})
    _gate_spelling(monkeypatch, tree._SILU_GATE, mx.bfloat16, spelling)
    tree._kernel()
    source = built[-1]["source"]
    assert source == tree.source(spelling)
    sigmoid = re.findall(r"bfloat\((metal::(?:precise::)?exp)\(metal::abs\(conv\)\)\)", source)
    assert sigmoid == [spelling]
    # compute_g's softplus keeps its own (independently checked) spelling.
    assert "bfloat(metal::exp(lo - hi))" in source
    other = next(s for s in fused._SILU_EXP_CANDIDATES if s != spelling)
    assert tree.source(other) != source
    # MLX caches libraries by name: one name per source.
    assert built[-1]["name"].startswith("stream_gdn_pre_")
    assert tree.stats()["pre_silu_exp"] == spelling
    assert tree.stats()["pre_parity"] == "exact"


def test_tree_pre_source_keeps_the_corrected_tensorfold_text():
    """Apart from the SiLU site, the fast spelling is the vendored source."""
    from mlx2.runtime.models import qwen38_tree_gdn as tree
    from mlx2.runtime.tensorfold_qwen38 import stream_gdn

    def body(text):
        lines = [line for line in text.splitlines() if not line.strip().startswith("//")]
        return [line for line in lines if line.strip()]

    assert body(tree.source(tree.KERNEL_SILU_EXP)) == body(stream_gdn._PRE)


def test_tree_pre_kernel_fails_closed(monkeypatch):
    from mlx2.runtime.models import qwen38_tree_gdn as tree

    monkeypatch.setattr(mx.fast, "metal_kernel", _never)
    monkeypatch.setattr(tree, "_kernel_cache", {})
    _gate_spelling(monkeypatch, tree._SILU_GATE, mx.bfloat16, None)
    with pytest.raises(RuntimeError, match="served compiled nn.silu matches no kernel form"):
        tree._kernel()
    _gate_spelling(monkeypatch, tree._SILU_GATE, mx.bfloat16, "metal::precise::exp")
    monkeypatch.setattr(tree, "_PARITY", {"metal::precise::exp": "stream_gdn.gdn_pre: v differs"})
    with pytest.raises(RuntimeError, match="v differs"):
        tree._kernel()


def test_tree_pre_parity_runs_once_and_raises_device_faults(monkeypatch):
    from mlx2.runtime.models import qwen38_tree_gdn as tree

    calls = []
    monkeypatch.setattr(tree, "_PARITY", {})
    _gate_spelling(monkeypatch, tree._SILU_GATE, mx.bfloat16, "metal::precise::exp")
    monkeypatch.setattr(tree, "_decode_parity", lambda s: calls.append(s) or None)
    assert tree.pre_refusal() is None and tree.pre_refusal() is None
    assert calls == ["metal::precise::exp"]

    monkeypatch.setattr(tree, "_PARITY", {})
    fault = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
    monkeypatch.setattr(tree, "_decode_parity", lambda s: (_ for _ in ()).throw(fault))
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        tree.pre_refusal()
    assert tree._PARITY == {}
    monkeypatch.setattr(tree, "_decode_parity", lambda s: (_ for _ in ()).throw(ValueError("x")))
    assert "parity probe failed" in tree.pre_refusal()


@pytest.mark.parametrize("numerics", prefill.PREWORK_NUMERICS)
def test_prefill_prework_source_respells_only_the_conv_silu(numerics):
    base = prefill.prefill_prework_source(numerics)
    served = prefill.prefill_prework_source(numerics, "metal::precise::exp")
    changed = [(a, b) for a, b in zip(base.splitlines(), served.splitlines()) if a != b]
    assert len(changed) == 1
    assert _conv_silu_spellings(served) == ["metal::precise::exp"]
    with pytest.raises(ValueError):
        prefill.prefill_prework_source(numerics, "fast::exp")


def test_prefill_swish_gate_respells_and_sigmoid_gate_stays_precise():
    swish = prefill.prefill_norm_gate_source("swish", "metal::precise::exp")
    assert "float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));" in swish
    assert "zv * (zv < 0.0f ? sy : 1.0f - sy)" in swish
    # The Qwen4 gate stands in for eager mx.sigmoid: precise on every spelling.
    for spelling in fused._SILU_EXP_CANDIDATES:
        assert prefill.prefill_norm_gate_source("sigmoid", spelling) == (
            prefill.prefill_norm_gate_source()
        )


@pytest.mark.parametrize("numerics,gate", sorted(prefill.ARCHITECTURE_CONTRACTS.values()))
@pytest.mark.parametrize("spelling", fused._SILU_EXP_CANDIDATES)
def test_prefill_kernels_compile_the_served_spellings(monkeypatch, numerics, gate, spelling):
    """Regression (2026-10-17): the prefill prework (Flash-Next, default on)
    and the 27B swish gate compiled ``metal::exp`` while MLX 0.32.4 serves
    ``metal::precise::exp``, so a conv output of -6.84375 (and its head's q/k
    norms) differed from the eager path that the route claims to equal."""
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(kw) or object())
    monkeypatch.setattr(prefill, "_KERNELS", {})
    _gate_spelling(monkeypatch, prefill._PREWORK_SILU_GATE, mx.bfloat16, spelling)
    _gate_spelling(monkeypatch, prefill._SWISH_GATE, mx.float32, spelling)
    prefill._kernels(numerics, gate)
    prework, norm_gate = built[-2:]
    assert prework["source"] == prefill.prefill_prework_source(numerics, spelling)
    assert _conv_silu_spellings(prework["source"]) == [spelling]
    suffix = "_precise" if "precise" in spelling else "_fast"
    assert prework["name"] == prefill._PREWORK_NAMES[numerics] + suffix
    if gate == "swish":
        assert norm_gate["source"] == prefill.prefill_norm_gate_source("swish", spelling)
        assert norm_gate["name"] == prefill._NORM_GATE_NAMES["swish"] + suffix
    else:
        assert norm_gate["source"] == prefill.prefill_norm_gate_source()


def test_prefill_kernels_and_refusal_fail_closed(monkeypatch):
    monkeypatch.setattr(mx.fast, "metal_kernel", _never)
    monkeypatch.setattr(prefill, "_KERNELS", {})
    monkeypatch.setattr(prefill, "_PARITY", {})
    _gate_spelling(monkeypatch, prefill._PREWORK_SILU_GATE, mx.bfloat16, None)
    with pytest.raises(RuntimeError, match="matches no kernel form"):
        prefill._kernels("qwen4", "sigmoid")
    assert "matches no kernel form" in prefill.prefill_refusal("qwen4", "sigmoid")
    _gate_spelling(monkeypatch, prefill._PREWORK_SILU_GATE, mx.bfloat16, "metal::precise::exp")
    _gate_spelling(monkeypatch, prefill._SWISH_GATE, mx.float32, None)
    assert "fp32 nn.silu matches no kernel form" in prefill.prefill_refusal("qwen35", "swish")
    _gate_spelling(monkeypatch, prefill._SWISH_GATE, mx.float32, "metal::precise::exp")
    calls = []
    monkeypatch.setattr(prefill, "eager_parity", lambda *a: calls.append(a) or "q differs")
    assert prefill.prefill_refusal("qwen35", "swish") == "q differs"
    assert prefill.prefill_refusal("qwen35", "swish") == "q differs"
    assert calls == [("qwen35", "swish", "metal::precise::exp", "metal::precise::exp")]
    fault = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
    monkeypatch.setattr(prefill, "eager_parity", lambda *a: (_ for _ in ()).throw(fault))
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        prefill.prefill_refusal("qwen4", "sigmoid")


def test_flash_next_prefill_declines_on_a_parity_refusal(monkeypatch):
    from mlx2.runtime.models import qwen4_exp
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    model, _vocab = tiny_qwen4_mtp()
    layer = next(
        layer.linear_attn
        for layer in model.language_model.model.layers
        if getattr(layer, "linear_attn", None) is not None
    )
    layer.set_fused_gdn_prefill_mode("fused")
    monkeypatch.setattr(prefill, "admit_qwen4_fused_gdn_prefill", _accept)
    monkeypatch.setattr(prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: None)
    seen = []
    reason = "fused GDN prefill v differs from the eager path (qwen4/sigmoid)"
    monkeypatch.setattr(prefill, "prefill_refusal", lambda *a: seen.append(a) or reason)
    monkeypatch.setattr(prefill, "qwen4_gdn_prefill_prework", _never)
    state = [mx.zeros((1, 3, 8)), mx.zeros((1, 4, 8, 8))]
    assert layer._try_fused_prefill(*_inputs(64), None, _Cache(state)) is None
    assert layer.fused_gdn_prefill_last_fallback == reason
    assert seen == [("qwen4", "sigmoid")]
