"""glm53_hc_fused and glm53_router=fused_v2: bitwise equality with the launches they replace.

CPU tests cover the policy wiring (model and adapter validation, profile
suffixes, the mega_v1 requirement), the off-Metal declines and the MLX gemv
geometry both kernels transcribe. The Metal tests (MLX2_RUN_GPU_TESTS=1)
compare each kernel with the stock composition bit for bit and run a synthetic
Glm53GGUFModel through chained B1 decode steps, eager and under
glm53_compile_regions=v1, against the glm53_hc=mega_v1 / glm53_router=fused_v1
reference.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_glue_metal as glue
from mlx2.runtime.models import glm53_hc_fused as fused
from mlx2.runtime.models.glm53_gguf import (
    HC_FUSED_COUNTERS,
    Glm53GGUFModel,
)

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
EXPERTS, HIDDEN, VOCAB, INNER = 144, 4096, 64, 32


def bits(a):
    return np.array(a).view(np.uint32)


def mlx_gemv_geometry(n, k):
    """Matmul::eval_gpu's gemv choice (MLX 0.32.4 matmul.cpp gemv_axbpy) for
    x[1,k] @ w.T with w row-major [n,k]: the non-transposed kernel."""
    tm, tn, sm, sn, bn = 4, 4, 1, 32, 1
    bm = 8 if n >= 4096 else 4
    if k <= 64:
        bm, sm, sn = 1, 8, 4
    elif k >= 16 * n:
        bm, bn = 1, 8
    tm = 1 if n < tm else tm
    return {"bm": bm, "bn": bn, "sm": sm, "sn": sn, "tm": tm, "tn": tn}


class SyntheticWeights:
    """Just enough of GGUFWeights for Glm53GGUFModel's HC, router, MLP and head."""

    def __init__(self, layers, seed=5310):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.values, self.tensors, self._native_linear_fp32 = {}, {}, {}
        self.scheduled_dense_calls, self.scheduled_grouped_calls, self.scheduled_expert_calls = {}, {}, {}

        def put(name, value, linear=False):
            self.values[name] = mx.array(value.astype(np.float32))
            if linear:
                self.tensors[name] = SimpleNamespace(ggml_type=0, shape=value.shape[::-1])

        put("token_embd.weight", rng.normal(0, 1, (VOCAB, HIDDEN)))
        put("output_norm.weight", rng.normal(1, 0.1, HIDDEN))
        put("output.weight", rng.normal(0, 0.02, (VOCAB, HIDDEN)), linear=True)
        for i in range(layers):
            for half, norm in (("hc_attn", "attn_norm"), ("hc_ffn", "ffn_norm")):
                put(f"blk.{i}.{half}_fn", rng.normal(0, 0.01, (24, 4 * HIDDEN)), linear=True)
                put(f"blk.{i}.{half}_scale", rng.uniform(0.1, 1.5, 3))
                put(f"blk.{i}.{half}_base", rng.normal(0, 0.5, 24))
                put(f"blk.{i}.{norm}.weight", rng.normal(1, 0.1, HIDDEN))
            for suffix in ("", "_shexp"):
                put(f"blk.{i}.ffn_gate{suffix}.weight", rng.normal(0, 0.02, (INNER, HIDDEN)), linear=True)
                put(f"blk.{i}.ffn_up{suffix}.weight", rng.normal(0, 0.02, (INNER, HIDDEN)), linear=True)
                put(f"blk.{i}.ffn_down{suffix}.weight", rng.normal(0, 0.1, (HIDDEN, INNER)), linear=True)
            put(f"blk.{i}.ffn_gate_inp.weight", rng.normal(0, 0.05, (EXPERTS, HIDDEN)), linear=True)
            put(f"blk.{i}.exp_probs_b.bias", rng.normal(0, 0.05, EXPERTS))
            put(f"blk.{i}.expert_scale", rng.normal(0, 1, (EXPERTS, HIDDEN)))

    def array(self, name):
        return self.values[name]

    def linear(self, name):
        if name not in self._native_linear_fp32:
            self._native_linear_fp32[name] = self.values[name].T
        operand = self._native_linear_fp32[name]
        return lambda x: x @ operand

    def embedding(self, ids):
        return self.values["token_embd.weight"][ids]

    def moe(self, x, ids, scores, prefix):
        """Depends on every selected id and weight: wrong routing changes bits."""
        table = self.values[prefix + ".expert_scale"]
        picked = table[ids.reshape(-1)].reshape(*ids.shape, HIDDEN)
        return (picked * scores[..., None]).sum(-2) * x


def synthetic_model(weights, layers, **policy):
    config = {"layer_types": ["linear_attention"] * layers, "rms_norm_eps": 1e-5,
              "max_position_embeddings": 4096, "hidden_size": HIDDEN}
    model = Glm53GGUFModel(config, weights, identity="synthetic", **policy)
    proj = mx.array(np.random.default_rng(5311).normal(0, 0.02, (HIDDEN, HIDDEN)).astype(np.float32))

    def kda(x, cache, prefix, **_):
        """Stand-in linear attention with recurrent state (cache semantics only)."""
        if cache.conv is None:
            cache.conv = mx.zeros((1, 3, HIDDEN), mx.float32)
            cache.recurrent = mx.zeros((1, HIDDEN), mx.float32)
        cache.conv = mx.concatenate([cache.conv[:, 1:], x.reshape(1, 1, HIDDEN)], 1)
        cache.recurrent = cache.recurrent * 0.5 + x.reshape(1, HIDDEN)
        cache.offset += 1
        return (x + cache.recurrent.reshape(x.shape) + cache.conv.sum(1).reshape(x.shape)) @ proj

    model._kda = kda
    return model


# ---- CPU ----------------------------------------------------------------------


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_kernels_transcribe_the_mlx_gemv_geometry():
    hc = {"bm": 1, "bn": fused.BN, "sm": 1, "sn": fused.SN, "tm": fused.TM, "tn": fused.TN}
    assert mlx_gemv_geometry(24, 16384) == hc
    router = {"bm": 1, "bn": glue.GEMV_BN, "sm": 1, "sn": 32, "tm": glue.GEMV_TM, "tn": glue.GEMV_TN}
    assert mlx_gemv_geometry(EXPERTS, HIDDEN) == router
    assert fused.TILES == 24 // fused.RPT and fused.BLOCK_N == fused.BN * fused.SN * fused.TN
    # The router transcription keeps _ROUTER_SOURCE's law verbatim.
    source = glue._router_v2_source()
    assert glue._ROUTER_READ not in source and "const float x = lg[i];" in source
    body = glue._ROUTER_SOURCE.lstrip("\n")[len(glue._ROUTER_PROLOGUE):]
    assert body.replace(glue._ROUTER_READ, "const float x = lg[i];") in source


def test_policy_values_and_the_mega_requirement_are_validated(cpu):
    weights = SyntheticWeights(0)
    assert synthetic_model(weights, 0).hc_fused_policy == "reference"
    with pytest.raises(ValueError, match="HC fused policy"):
        synthetic_model(weights, 0, hc_fused=True, hc_mega=True, hc_fused_policy="v2")
    with pytest.raises(ValueError, match="requires glm53_hc=mega_v1"):
        synthetic_model(weights, 0, hc_fused=True, hc_fused_policy="v1")
    with pytest.raises(ValueError, match="requires glm53_hc=mega_v1"):
        synthetic_model(weights, 0, hc_mega=True, hc_fused_policy="tail_v1")
    for value in ("v1", "tail_v1"):
        model = synthetic_model(weights, 0, hc_fused=True, hc_mega=True, hc_fused_policy=value)
        assert all(model.calls[name] == 0 for name in HC_FUSED_COUNTERS)
    assert synthetic_model(weights, 0, router="fused_v2").router == "fused_v2"


def test_off_metal_both_policies_decline_count_and_keep_the_reference(cpu):
    weights = SyntheticWeights(4)
    reference = synthetic_model(weights, 4, router="fused_v1")
    v2 = synthetic_model(weights, 4, router="fused_v2")
    x = mx.array(np.random.default_rng(5312).normal(size=(1, 1, HIDDEN)).astype(np.float32))
    want, got = reference._mlp(x, "blk.3", 3), v2._mlp(x, "blk.3", 3)
    assert np.array_equal(bits(got), bits(want))
    assert v2.calls["router_fused_v2_declined"] == 1 and v2.calls["router_fused_v2"] == 0
    hc = synthetic_model(weights, 4, hc_fused=True, hc_mega=True, hc_fused_policy="tail_v1")
    h = mx.zeros((1, 1, 4, HIDDEN), mx.float32)
    assert hc._hc_fused(h, "blk.0.hc_attn", "blk.0.attn_norm.weight", None) is None
    assert hc.calls["hc_fused_declined"] == 1
    # Multi-row boundaries bypass the fused kernel and are counted once each.
    rows = mx.zeros((1, 3, 4, HIDDEN), mx.float32)
    hc.hc_mega = hc.hc_fused = False  # the ordinary composition runs on the CPU
    hc._hc_in(rows, "blk.0.hc_attn", "blk.0.attn_norm.weight")
    hc._hc_mid(mx.zeros((1, 3, HIDDEN), mx.float32), rows, mx.ones((1, 3, 4)),
               mx.broadcast_to(mx.eye(4), (1, 3, 4, 4)), "blk.0.hc_ffn", "blk.0.ffn_norm.weight")
    assert hc.calls["hc_fused_declined"] == 3


def test_f32_rows_is_the_reference_operand_and_refuses_other_storage(cpu):
    weights = SyntheticWeights(1)
    model = synthetic_model(weights, 1)
    rows = model._f32_rows("blk.0.hc_attn_fn")
    assert rows.shape == (24, 4 * HIDDEN) and np.array_equal(np.array(rows), np.array(weights.values["blk.0.hc_attn_fn"]))
    weights.tensors["blk.0.ffn_gate_inp.weight"] = SimpleNamespace(ggml_type=1, shape=(HIDDEN, EXPERTS))
    assert model._f32_rows("blk.0.ffn_gate_inp.weight") is None


def _adapter(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-hc-fused", policy)
    return adapter


def test_adapter_keys_validate_and_extend_the_profile_only_off_the_reference():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    default = _adapter({"glm53_hc": "mega_v1"})
    assert default.kernel_policy["glm53_hc_fused"] == "reference" and "hc_fused" not in default.profile
    v1 = _adapter({"glm53_hc": "mega_v1", "glm53_hc_fused": "v1"})
    tail = _adapter({"glm53_hc": "mega_v1", "glm53_hc_fused": "tail_v1"})
    router = _adapter({"glm53_hc": "mega_v1", "glm53_router": "fused_v2"})
    assert ":hc_fused=v1" in v1.profile and ":hc_fused=tail_v1" in tail.profile
    assert ":router=fused_v2" in router.profile
    assert len({default.profile, v1.profile, tail.profile, router.profile}) == 4
    for bad in ({"glm53_hc_fused": "v1"}, {"glm53_hc": "fused_fp32", "glm53_hc_fused": "tail_v1"},
                {"glm53_hc": "mega_v1", "glm53_hc_fused": "v2"}):
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


def _hc_case(rng, magnitude, special=None):
    h = (rng.normal(size=(1, 1, 4, HIDDEN)) * magnitude).astype(np.float32)
    if special == "zeros":
        h[..., ::3] = 0.0
        h[..., 1::7] = -0.0
    elif special == "subnormal":
        h[..., ::5] = np.float32(1e-40)
    branch = (rng.normal(size=(1, 1, HIDDEN)) * magnitude).astype(np.float32)
    return [mx.array(v) for v in (
        h, branch, rng.uniform(0, 2, (1, 1, 4)).astype(np.float32),
        rng.uniform(0, 1, (1, 1, 4, 4)).astype(np.float32),
        (rng.normal(size=(24, 4 * HIDDEN)) * 10 ** rng.uniform(-3, -1)).astype(np.float32),
        rng.uniform(0.1, 2.0, 3).astype(np.float32), rng.normal(size=24).astype(np.float32),
        rng.normal(size=HIDDEN).astype(np.float32))]


@pytest.fixture(params=[1, 2, 4], ids=lambda r: f"rpt{r}")
def hc_rpt(request, monkeypatch):
    monkeypatch.setattr(fused, "RPT", request.param)
    monkeypatch.setattr(fused, "TILES", 24 // request.param)
    return request.param


@pytest.fixture(params=[1, 4], ids=lambda r: f"rpt{r}")
def router_rpt(request, monkeypatch):
    monkeypatch.setattr(glue, "GEMV_RPT", request.param)
    return request.param


@metal
def test_gpu_hc_kernels_are_bitwise_the_mega_composition(gpu, hc_rpt):
    rng = np.random.default_rng(5313)
    cases = [(m, None) for m in (1e-3, 0.05, 1.0, 7.0, 30.0, 300.0, 3e4)]
    cases += [(1.0, "zeros"), (1.0, "subnormal")]
    for trial, (magnitude, special) in enumerate(cases * 2):
        h, branch, post, comb, fn, scale, base, weight = _hc_case(rng, magnitude, special)
        for expand in (None, (branch, h, post, comb)):
            want = fused.reference(h, fn, scale, base, weight, rms_eps=1e-5, expand=expand)
            got_h, got_mixes = fused.project(h, fn, rms_eps=1e-5, expand=expand)
            tail = fused.boundary(h, fn, scale, base, weight, rms_eps=1e-5, expand=expand)
            mx.eval(want, got_h, got_mixes, tail)
            label = (trial, magnitude, special, expand is not None)
            assert np.array_equal(bits(got_h), bits(want[0])), label
            assert np.array_equal(bits(got_mixes), bits(want[5])), label  # the gemv itself
            for index, (a, b) in enumerate(zip(tail, want[:5])):
                assert np.array_equal(bits(a), bits(b)), (label, index)


@metal
def test_gpu_tail_counter_survives_chained_and_independent_launches(gpu, hc_rpt):
    rng = np.random.default_rng(5314)
    h, branch, post, comb, fn, scale, base, weight = _hc_case(rng, 1.0)
    counter = fused.counter()
    want_h, got_h = h, h
    want_x, got_x = None, None
    for _ in range(24):  # one graph: each launch depends on the previous one's h
        want_h, want_x, _, _, _, _ = fused.reference(
            want_h, fn, scale, base, weight, rms_eps=1e-5, expand=(branch, want_h, post, comb))
        got_h, got_x, _, _, _ = fused.boundary(
            got_h, fn, scale, base, weight, counter, rms_eps=1e-5, expand=(branch, got_h, post, comb))
        want_h, got_h = want_h * 0.5, got_h * 0.5
    mx.eval(want_h, got_h, want_x, got_x)
    assert np.array_equal(bits(got_h), bits(want_h)) and np.array_equal(bits(got_x), bits(want_x))
    assert int(np.array(counter)[0]) == 0
    # Independent launches in one graph, each with its own counter.
    outs = [fused.boundary(h * (i + 1), fn, scale, base, weight, fused.counter(), rms_eps=1e-5)[1]
            for i in range(8)]
    refs = [fused.reference(h * (i + 1), fn, scale, base, weight, rms_eps=1e-5)[1] for i in range(8)]
    mx.eval(outs, refs)
    for a, b in zip(outs, refs):
        assert np.array_equal(bits(a), bits(b))


@metal
def test_gpu_router_gemv_is_bitwise_gemv_then_the_v1_router(gpu, router_rpt):
    rng = np.random.default_rng(5315)
    for trial in range(24):
        x = mx.array((rng.normal(size=(1, 1, HIDDEN)) * 10 ** rng.uniform(-2, 2)).astype(np.float32))
        w = rng.normal(0, 0.05, (EXPERTS, HIDDEN)).astype(np.float32)
        if trial % 4 == 1:
            w[10:30] = w[10]  # exact logit ties straddling the top-8 cut
        if trial % 4 == 2:
            w[:] = 0.0  # every logit 0: the bias alone orders, ties broken by index
        w, b = mx.array(w), mx.array(rng.normal(0, 0.05 if trial % 2 else 0, EXPERTS).astype(np.float32))
        assert glue.router_gemv_admits(x, w, b, 8)
        logits = x @ w.T
        want = glue.fused_router(logits, b)
        counter = glue.router_counter()  # reused below only after this launch completes
        got = glue.fused_router_gemv(x, w, b, counter)
        mx.eval(got)
        ids, weights, scratch = glue.fused_router_gemv(x, w, b, counter, with_logits=True)
        mx.eval(want, got, logits, ids, weights, scratch)
        assert np.array_equal(bits(scratch), bits(logits.reshape(-1))), trial  # the gemv itself
        for ids_, weights_ in (got, (ids, weights)):
            np.testing.assert_array_equal(np.array(ids_), np.array(want[0]), err_msg=str(trial))
            assert np.array_equal(bits(weights_), bits(want[1])), trial
        assert got[0].shape == (1, 1, 8)
        assert int(np.array(counter)[0]) == 0  # the last threadgroup reset it


POLICIES = {
    "reference": {"hc_fused": True, "hc_mega": True, "router": "fused_v1"},
    "v1": {"hc_fused": True, "hc_mega": True, "router": "fused_v2", "hc_fused_policy": "v1"},
    "tail_v1": {"hc_fused": True, "hc_mega": True, "router": "fused_v2", "hc_fused_policy": "tail_v1"},
}


def _decode_chain(model, steps, regions):
    model.compile_regions = "v1" if regions else "off"
    cache = model.make_cache()
    token, out = 3, []
    for _ in range(steps):
        logits = model(mx.array([[token]], dtype=mx.int32), cache)
        mx.eval(logits, [c.state for c in cache])
        out.append(bits(logits).copy())
        token = int(np.argmax(np.array(logits).reshape(-1)))
    return out, [bits(a).copy() for c in cache for a in c.state]


@metal
@pytest.mark.parametrize("regions", [False, True], ids=["eager", "compile_regions"])
def test_gpu_model_chained_decode_is_bitwise_the_mega_v1_reference(gpu, regions):
    layers, steps = 5, 6  # three dense layers, two MoE layers
    weights = SyntheticWeights(layers)
    runs = {}
    for name, policy in POLICIES.items():
        model = synthetic_model(weights, layers, **policy)
        runs[name] = (_decode_chain(model, steps, regions), dict(model.calls))
    (want_logits, want_state), _ = runs["reference"]
    for name in ("v1", "tail_v1"):
        (logits, state), calls = runs[name]
        for step, (a, b) in enumerate(zip(logits, want_logits)):
            assert np.array_equal(a, b), (name, step)
        for a, b in zip(state, want_state):
            assert np.array_equal(a, b), name
        counter = "hc_fused_tail" if name == "tail_v1" else "hc_fused_boundary"
        # Per step: one layer-start boundary and one mid boundary per layer.
        assert calls[counter] == 2 * layers * steps and calls["hc_fused_declined"] == 0, calls
        assert calls["router_fused_v2"] == 2 * steps and calls["router_fused_v2_declined"] == 0, calls


def test_compile_region_counters_are_fresh_inputs_never_captured(cpu):
    from mlx2.runtime.models import glm53_compile_regions as regions

    block = mx.array(np.zeros(regions.COUNTER_SLOTS * 16, np.uint32))
    with glue.counter_supply(block):
        first, second = glue.arrival_counter(), fused.counter()
    assert first.shape == second.shape == (16,)
    assert glue.arrival_counter().shape == (16,)  # outside a trace: a fresh array
    with glue.counter_supply(block), pytest.raises(RuntimeError, match="exhausted"):
        for _ in range(regions.COUNTER_SLOTS + 1):
            glue.arrival_counter()


@metal
@pytest.mark.parametrize("regions", [False, True], ids=["eager", "compile_regions"])
def test_gpu_independent_forwards_evaluated_together_stay_bitwise(gpu, regions):
    """Two B1 forwards on separate caches in one eval: no ordering between their
    tail launches, so each launch must own its arrival counter."""
    layers = 4
    weights = SyntheticWeights(layers, seed=5316)
    out = {}
    for name in ("reference", "tail_v1"):
        model = synthetic_model(weights, layers, **POLICIES[name])
        model.compile_regions = "v1" if regions else "off"
        caches = [model.make_cache(), model.make_cache()]
        rows = []
        for step in range(5):  # steps 0-1 eager/warm-up, then traced replays
            logits = [model(mx.array([[3 + step + 7 * i]], dtype=mx.int32), c) for i, c in enumerate(caches)]
            mx.eval(logits, [c.state for cache in caches for c in cache])
            rows.append([bits(a).copy() for a in logits])
        out[name] = rows
    for step, (got, want) in enumerate(zip(out["tail_v1"], out["reference"])):
        for a, b in zip(got, want):
            assert np.array_equal(a, b), step
