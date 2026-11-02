"""glm53_kda_fused=decode_v1: the one-launch KDA decode body vs the reference op chain.

CPU tests prove the policy wiring: model and adapter validation, the APCv2
profile suffix only off the reference, the compile-regions policy key, the
kernel source substitution, and that decode_v1 off Metal runs the reference
chain bit for bit and counts the decline. The Metal tests
(MLX2_RUN_GPU_TESTS=1) compare the launch with the chain bitwise: random and
real-scale inputs, edge cases, several decode tokens chained with both states
carried, and the launch inside an ``mx.compile`` region as
glm53_compile_regions traces it.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime.models import glm53_kda_fused as kf
from mlx2.runtime.models import precise_ops
from mlx2.runtime.models.glm53_gguf import GLUE_COUNTERS, Glm53GGUFModel, Glm53KDACache
from mlx2.runtime.models.gated_delta import gated_delta_kernel

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
HEADS, DIM = 64, 128
INNER = HEADS * DIM
HIDDEN = 32


def bits(a):
    return np.array(a).view(np.uint32)


def same(a, b):
    mx.eval(a, b)
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


class KdaWeights:
    """One synthetic KDA layer at the checkpoint's geometry.

    ``scale`` multiplies every projection, so 1 gives real-scale activations
    (qkv std ~1, gates ~N(0, 2)) and large values stress the norms.
    """

    def __init__(self, seed=541, scale=1.0):
        rng = np.random.default_rng(seed)
        self.values = {
            "blk.0.ssm_dt.bias": rng.normal(-2, 0.5, INNER).astype(np.float32),
            "blk.0.ssm_a": rng.uniform(0.5, 8, HEADS).astype(np.float32),
            "blk.0.ssm_norm.weight": rng.normal(1, 0.2, DIM).astype(np.float32),
            **{f"blk.0.ssm_conv1d_{p}.weight": rng.normal(0, 0.4, (INNER, 1, 4)).astype(np.float32)
               for p in "qkv"},
        }
        self.rng, self.scale, self.mats = rng, scale, {}

    def array(self, name):
        return mx.array(self.values[name])

    def linear(self, name):
        def call(x):
            if name not in self.mats:
                out = {"ssm_f_a": 16, "ssm_g_a": 16, "ssm_beta": HEADS, "attn_output": HIDDEN}.get(
                    name.split(".")[2], INNER)
                std = self.scale / np.sqrt(x.shape[-1])
                if name.split(".")[2] in ("ssm_f_b", "ssm_g_b", "ssm_beta"):
                    std *= 2.0
                self.mats[name] = mx.array(
                    self.rng.normal(0, std, (x.shape[-1], out)).astype(np.float32))
            return x @ self.mats[name]

        return call


def model(weights, **policy):
    return Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5}, weights, identity="synthetic", **policy
    )


def tokens(seed, count, scale=1.0):
    rng = np.random.default_rng(seed)
    return mx.array((rng.normal(size=(1, count, HIDDEN)) * scale).astype(np.float32))


# ---- CPU ----------------------------------------------------------------------


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_policy_is_validated_and_defaults_to_the_reference(cpu):
    weights = KdaWeights()
    assert model(weights).kda_fused == "reference"
    assert model(weights, kda_fused="decode_v1").kda_fused == "decode_v1"
    assert {"kda_fused", "kda_fused_declined"} <= set(GLUE_COUNTERS)
    for bad in ("fused_v1", "on", None):
        with pytest.raises(ValueError, match="glue policy"):
            model(weights, kda_fused=bad)


def test_decode_v1_off_metal_runs_the_reference_chain_and_counts_declines(cpu):
    weights = KdaWeights()
    reference, fused = model(weights), model(weights, kda_fused="decode_v1")
    one, two = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
    x = tokens(542, 5)
    for chunk in (x[:, :3], x[:, 3:4], x[:, 4:5]):  # a prompt chunk, then two decode tokens
        want, got = reference._kda(chunk, one, "blk.0"), fused._kda(chunk, two, "blk.0")
        assert same(got, want)
        for a, b in zip(one.state, two.state):
            assert same(a, b)
    assert one.offset == two.offset == 5
    assert fused.calls["kda_fused"] == 0 and fused.calls["kda_fused_declined"] == 3
    assert reference.calls["kda_fused_declined"] == 0
    # A decline never projects twice: the fused model ran the reference recurrence.
    assert fused.calls["kda_reference"] == reference.calls["kda_reference"] == 3


def test_kernel_source_substitutes_every_placeholder_and_pins_constants():
    src = kf.kernel_source(1e-5)
    for token in ("CONV_TAP", "RMS_SQ", "SIGMOID_", "L2_EPS", "Q_SCALE", "RMS_EPS"):
        assert token not in src
    assert float.fromhex(kf._hexf(1e-6)[:-1]) == float(np.float32(1e-6))
    assert float.fromhex(kf._hexf(DIM**-0.5)[:-1]) == float(np.float32(DIM**-0.5))
    assert float.fromhex(kf._hexf(1e-5)[:-1]) == float(np.float32(1e-5))
    fused = kf.kernel_source(1e-5, True, True)
    plain = kf.kernel_source(1e-5, False, False)
    assert fused.count("metal::fma") == 8 and plain.count("metal::fma") == 0


def test_admission_declines_off_metal_and_on_wrong_geometry(cpu):
    f32 = lambda *s: mx.zeros(s, mx.float32)
    args = [f32(1, 1, 3 * INNER), f32(1, 3, 3 * INNER), f32(3 * INNER, 4, 1), f32(1, 1, INNER),
            f32(INNER), f32(HEADS), f32(1, 1, HEADS), f32(1, 1, INNER), f32(1, HEADS, DIM, DIM),
            f32(DIM)]
    assert not kf.kda_fused_admits(*args)  # CPU


def test_compile_regions_policy_key_includes_kda_fused():
    from mlx2.runtime.models import glm53_compile_regions as regions

    weights = KdaWeights()
    m = model(weights)
    layers = regions.CompiledLayers(m)
    before = layers._policy_key()
    m.kda_fused = "decode_v1"
    assert layers._policy_key() != before


def _adapter(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-kda-fused", policy)
    return adapter


def test_adapter_key_is_validated_and_extends_the_profile_only_off_the_reference():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    default = _adapter({})
    reference = _adapter({"glm53_kda_fused": "reference"})
    fused = _adapter({"glm53_kda_fused": "decode_v1"})
    # decode_v1 is the adapter default since 2026-10-31 (Pierre); "reference" opts out.
    assert default.kernel_policy["glm53_kda_fused"] == "decode_v1"
    assert default.profile == fused.profile and "kda_fused" not in reference.profile
    assert fused.profile.endswith(":kda_fused=decode_v1") or ":kda_fused=decode_v1:" in fused.profile
    assert fused.profile != reference.profile
    for bad in ({"glm53_kda_fused": "fused_v1"}, {"glm53_kda_fused": "on"}):
        with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
            object.__new__(Glm53GGUFAdapter)._load("/nonexistent", bad)


# ---- Metal (MLX2_RUN_GPU_TESTS=1 only) ------------------------------------------

metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def reference_chain(mixed, conv, filters, a, dt_bias, ssm_a, beta, gate, state, norm, eps):
    """Glm53GGUFModel._kda's reference chain after the projections, verbatim (L = 1)."""
    b, length = 1, 1
    conv_input = mx.concatenate((conv, mixed), axis=1)
    new_conv = mx.contiguous(conv_input[:, -3:])
    out = nn.silu(mx.conv1d(conv_input, filters, groups=3 * INNER))
    q, k, v = [t.reshape(b, length, HEADS, DIM) for t in mx.split(out, 3, axis=-1)]
    q = q / mx.maximum(mx.sqrt((q * q).sum(-1, keepdims=True)), 1e-6) * DIM**-0.5
    k = k / mx.maximum(mx.sqrt((k * k).sum(-1, keepdims=True)), 1e-6)
    a = a.reshape(b, length, HEADS, DIM) + dt_bias.reshape(HEADS, DIM)
    decay = mx.exp(-5 * mx.sigmoid(a * ssm_a.reshape(HEADS, 1)))
    beta = mx.sigmoid(beta)
    y, new_state = gated_delta_kernel(q, k, v, decay, beta, state)
    gated = mx.sigmoid(gate.reshape(b, length, HEADS, DIM))
    y = mx.fast.rms_norm(y.astype(mx.float32), norm, eps) * gated
    return y.reshape(b, length, INNER), new_conv, new_state


def random_inputs(rng, *, scale=1.0, state_scale=0.05, conv_scale=1.0, zero_state=False):
    f = lambda shape, s, loc=0.0: mx.array((rng.normal(loc, 1, shape) * s).astype(np.float32))
    state = (mx.zeros((1, HEADS, DIM, DIM), mx.float32) if zero_state
             else f((1, HEADS, DIM, DIM), state_scale))
    return dict(
        mixed=f((1, 1, 3 * INNER), scale), conv=f((1, 3, 3 * INNER), conv_scale * scale),
        filters=f((3 * INNER, 4, 1), 0.4), a=f((1, 1, INNER), 2.0),
        dt_bias=f((INNER,), 0.5, -2.0), ssm_a=mx.array(rng.uniform(0.5, 8, HEADS).astype(np.float32)),
        beta=f((1, 1, HEADS), 2.0), gate=f((1, 1, INNER), 2.0), state=state,
        norm=f((DIM,), 0.2, 1.0),
    )


def _check(inputs, eps=1e-5, contraction=(kf.CONV_FMA, kf.RMS_FMA)):
    assert kf.kda_fused_admits(*inputs.values())
    want = reference_chain(*inputs.values(), eps)
    got = kf.fused_kda_decode(*inputs.values(), rms_eps=eps, _contraction=contraction)
    return [same(g, w) for g, w in zip(got, want)]


@metal
def test_gpu_fused_launch_is_bitwise_the_reference_chain(gpu):
    rng = np.random.default_rng(543)
    cases = {
        "random": random_inputs(rng),
        "real_scale": random_inputs(rng, scale=1.0, state_scale=0.3),
        "zero_state_zero_conv": random_inputs(rng, zero_state=True, conv_scale=0.0),
        "large": random_inputs(rng, scale=300.0, state_scale=50.0),
        "tiny": random_inputs(rng, scale=1e-7, state_scale=1e-6),  # norms hit the 1e-6 floor
    }
    edge = random_inputs(rng)
    mixed = np.array(edge["mixed"])
    mixed[0, 0, :DIM] = 0.0  # head 0 of q is all zero after a zero conv window
    edge["conv"] = mx.array(np.where(np.arange(3 * INNER) < DIM, 0.0, np.array(edge["conv"])).astype(np.float32))
    mixed[0, 0, INNER + 5] = 1e30  # an overflowing square in one k head
    edge["mixed"] = mx.array(mixed)
    cases["edge"] = edge
    for name, inputs in cases.items():
        assert all(_check(inputs)), name


@metal
def test_gpu_contraction_choices_are_the_mlx_build(gpu):
    """Pins CONV_FMA / RMS_FMA: the other choices are not bitwise (on this build)."""
    rng = np.random.default_rng(544)
    inputs = random_inputs(rng, state_scale=0.3)
    assert all(_check(inputs))
    for contraction in ((not kf.CONV_FMA, kf.RMS_FMA), (kf.CONV_FMA, not kf.RMS_FMA)):
        trials = [_check(random_inputs(rng, state_scale=0.3), contraction=contraction)
                  for _ in range(3)]
        assert not all(all(t) for t in trials), contraction


@metal
def test_gpu_model_decode_steps_chain_bitwise_with_states_carried(gpu):
    for scale, small in ((1.0, "reference"), (1.0, "fused_v1"), (40.0, "fused_v1")):
        weights = KdaWeights(seed=545, scale=scale)
        reference = model(weights, kda_small=small)
        fused = model(weights, kda_small=small, kda_fused="decode_v1")
        one, two = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
        x = tokens(546, 9)
        # First token with no state at all, a prompt chunk, then six decode tokens.
        for chunk in [x[:, :1], x[:, 1:3]] + [x[:, i:i + 1] for i in range(3, 9)]:
            want = reference._kda(chunk, one, "blk.0")
            got = fused._kda(chunk, two, "blk.0")
            assert same(got, want), (scale, small, chunk.shape)
            for a, b in zip(one.state, two.state):
                assert same(a, b)
            pre_want = reference._kda(chunk, _clone(one), "blk.0", _return_pre_output=True)
            pre_got = fused._kda(chunk, _clone(two), "blk.0", _return_pre_output=True)
            assert same(pre_got, pre_want)
        assert one.offset == two.offset == 9
        assert fused.calls["kda_fused"] == 2 * 7 and fused.calls["kda_fused_declined"] == 2
        assert fused.calls["kda_kernel"] == 2 and reference.calls["kda_kernel"] == 2 * 8


def _clone(cache):
    c = Glm53KDACache("synthetic")
    c.conv, c.recurrent, c.offset = cache.conv, cache.recurrent, cache.offset
    return c


@metal
def test_gpu_fused_inside_a_compiled_region_equals_eager_reference(gpu):
    """As glm53_compile_regions traces a KDA layer: states in, states out, precise span."""
    weights = KdaWeights(seed=547)
    reference = model(weights, kda_small="fused_v1")
    fused = model(weights, kda_small="fused_v1", kda_fused="decode_v1")
    eager, traced_cache = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
    x = tokens(548, 7)
    reference._kda(x[:, :2], eager, "blk.0")
    fused._kda(x[:, :2], traced_cache, "blk.0")  # warm-up: memos hold real arrays

    def body(h, conv, recurrent):
        state = SimpleNamespace(conv=conv, recurrent=recurrent, offset=0)
        token = precise_ops._IN_PRECISE_SPAN.set(True)
        try:
            out = fused._kda(h, state, "blk.0")
        finally:
            precise_ops._IN_PRECISE_SPAN.reset(token)
        return out, state.conv, state.recurrent

    compiled = mx.compile(body)
    for i in range(2, 7):
        want = reference._kda(x[:, i:i + 1], eager, "blk.0")
        got, traced_cache.conv, traced_cache.recurrent = compiled(
            x[:, i:i + 1], traced_cache.conv, traced_cache.recurrent)
        assert same(got, want), i
        assert same(traced_cache.conv, eager.conv) and same(traced_cache.recurrent, eager.recurrent)


@metal
def test_gpu_admission_declines_wrong_dtype_and_geometry(gpu):
    rng = np.random.default_rng(549)
    good = random_inputs(rng)
    assert kf.kda_fused_admits(*good.values())
    bad = {
        "mixed_f16": {"mixed": good["mixed"].astype(mx.float16)},
        "state_bf16": {"state": good["state"].astype(mx.bfloat16)},
        "two_tokens": {"mixed": mx.concatenate([good["mixed"]] * 2, axis=1)},
        "two_rows": {"conv": mx.concatenate([good["conv"]] * 2, axis=0)},
        "short_conv": {"conv": good["conv"][:, 1:]},
        "beta_width": {"beta": good["beta"][..., :32]},
        "gate_two_tokens": {"gate": mx.concatenate([good["gate"]] * 2, axis=1)},
        "state_heads": {"state": good["state"][:, :32]},
        "norm_width": {"norm": mx.concatenate([good["norm"]] * 2)},
        "filters_taps": {"filters": good["filters"][:, :3]},
        "ssm_a_width": {"ssm_a": good["ssm_a"][:32]},
        "missing_state": {"state": None},
    }
    for name, change in bad.items():
        assert not kf.kda_fused_admits(*{**good, **change}.values()), name


@metal
def test_gpu_model_declines_prefill_batch_and_reference_math(gpu):
    weights = KdaWeights(seed=550)
    for policy, x in (({}, tokens(551, 2)),
                      ({}, mx.concatenate([tokens(552, 1)] * 2, axis=0)),
                      ({"math_policy": "reference"}, tokens(553, 1))):
        reference = model(weights, **policy)
        fused = model(weights, kda_fused="decode_v1", **policy)
        one, two = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
        assert same(fused._kda(x, two, "blk.0"), reference._kda(x, one, "blk.0"))
        for a, b in zip(one.state, two.state):
            assert same(a, b)
        assert fused.calls["kda_fused"] == 0 and fused.calls["kda_fused_declined"] == 1, policy


@metal
def test_gpu_compiled_layers_runner_replays_fused_counters_and_matches_eager(gpu):
    """glm53_compile_regions.CompiledLayers over one real KDA layer (stub HC/MLP)."""
    from mlx2.runtime.models import glm53_compile_regions as regions
    from mlx2.runtime.models.glm53_gguf import hc_expand

    def build():
        m = model(KdaWeights(seed=554), kda_small="fused_v1", kda_fused="decode_v1")
        m.layers = [SimpleNamespace(is_linear=True)]

        def hc_in(h, half, norm):
            post = mx.full((*h.shape[:-1],), 0.5)
            comb = mx.broadcast_to(mx.eye(4) * 0.25, (*h.shape[:-2], 4, 4))
            return h.mean(-2), post, comb

        m._hc_in = hc_in
        m._mlp = lambda x, prefix, index: x * 2
        return m

    def eager_layer(m, cache, h):
        x, post, comb = m._hc_in(h, "", "")
        h = hc_expand(m._kda(x, cache, "blk.0"), h, post, comb, fused=False)
        x, post, comb = m._hc_in(h, "", "")
        return hc_expand(m._mlp(x, "blk.0", 0), h, post, comb, fused=False)

    eager_model, compiled_model = build(), build()
    compiled_model.weights = eager_model.weights  # same synthetic matrices
    eager_cache, compiled_cache = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
    rng = np.random.default_rng(555)
    prompt = mx.array(rng.normal(size=(1, 2, 4, HIDDEN)).astype(np.float32))
    eager_layer(eager_model, eager_cache, prompt)
    eager_layer(compiled_model, compiled_cache, prompt)  # declined prompt; the warm-up builds memos
    layers = regions.CompiledLayers(compiled_model)
    for step in range(4):
        h = mx.array(rng.normal(size=(1, 1, 4, HIDDEN)).astype(np.float32))
        want = eager_layer(eager_model, eager_cache, h)
        got = layers.layer(0, compiled_model.layers[0], compiled_cache, h, 2 + step)
        assert same(got, want), step
        assert same(compiled_cache.conv, eager_cache.conv)
        assert same(compiled_cache.recurrent, eager_cache.recurrent)
        assert compiled_cache.offset == eager_cache.offset == 3 + step
        assert compiled_model.calls["kda_fused"] == eager_model.calls["kda_fused"] == 1 + step
    assert len(layers.regions) == 1 and layers.regions[("kda", 0)].delta is not None
