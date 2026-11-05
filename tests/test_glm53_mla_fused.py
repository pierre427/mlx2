"""glm53_mla_fused=decode_v1: the B1 decode MLA glue in four kernels vs the reference chain.

CPU tests prove the wiring: model and adapter validation, the profile suffix
only off the reference, the arm4-law requirement, kernel source substitution,
the pool cache's ``pool_fn`` hook, the additive-mask attention entry, and that
decode_v1 off Metal declines and runs the reference chain bit for bit. The
Metal tests (MLX2_RUN_GPU_TESTS=1) gate each kernel bitwise against its op
chain and run twin models (reference vs decode_v1) through decode steps that
cross T = 2048 (dense to selecting) and several pool completions, from a
fresh and from a warm pool cache, with and without grouped projections, and
inside full forwards under glm53_compile_regions=v1: every output and every
committed cache word (latent and raw planes, pooled plane) equal.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from test_glm53_layer_tree_cpu import (
    LAYER_TYPES,
    MLA_LAYERS,
    TreeWeights,
    cache_bits,
    live_cache,
)

from mlx2.runtime.models import glm53_gguf_indexer as indexer
from mlx2.runtime.models import glm53_mla_fused as fused
from mlx2.runtime.models import glm53_sparse_attention as sa
from mlx2.runtime.models.glm53_gguf import (
    MLA_FUSED_COUNTERS,
    Glm53GGUFModel,
    sparse_attention_phase,
)
from mlx2.runtime.models.glm53_gguf_indexer import Glm53PoolCache

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
ARM4 = {"indexer_pools": "incremental_v1", "indexer_select": "topk_v1_kernel",
        "sparse_attention": "gathered_v1"}


class MlaWeights(TreeWeights):
    """TreeWeights at the checkpoint's MLA geometry (q_a rank 1536) with
    non-trivial norm weights and biases."""

    def array(self, name):
        if name not in self.arrays:
            r = self.rng
            if name.endswith("attn_q_a_norm.weight"):
                value = 1 + r.normal(0, 0.2, 1536)
            elif name.endswith("attn_kv_a_norm.weight"):
                value = 1 + r.normal(0, 0.2, 512)
            elif name.endswith("k_norm.weight"):
                value = 1 + r.normal(0, 0.2, 128)
            elif name.endswith("k_norm.bias"):
                value = r.normal(0, 0.1, 128)
            else:
                return super().array(name)
            self.arrays[name] = mx.array(np.asarray(value, dtype=np.float32))
        return self.arrays[name]

    @staticmethod
    def _out(name):
        if name.endswith("attn_q_a.weight"):
            return 1536
        return TreeWeights._out(name)


def make(weights, **policy):
    config = {"layer_types": LAYER_TYPES, "rms_norm_eps": 1e-5,
              "max_position_embeddings": 65536, "vocab_size": 16}
    return Glm53GGUFModel(config, weights, identity="synthetic-glm53-tree",
                          math_policy="reference", **{**ARM4, **policy})


def same(a, b):
    mx.eval(a, b)
    return a.shape == b.shape and np.array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))


def run_layers(weights, tokens, steps, policy, *, warm_pools=False, seed=77):
    """``steps`` B1 decode tokens through both MLA layers: (outputs, caches, model)."""
    model = make(weights, **policy)
    cache = live_cache(model, tokens, warm_pools=warm_pools)
    rng = np.random.default_rng(seed)
    outs = []
    for step in range(steps):
        for index in MLA_LAYERS:
            x = mx.array(rng.normal(size=(1, 1, 4)).astype(np.float32))
            out = model._mla(x, cache[index], f"blk.{index}", frontier=tokens + step)
            mx.eval(out)
            outs.append(out)
    return outs, cache, model


# ---- CPU ------------------------------------------------------------------------


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_policy_is_validated_and_defaults_to_the_reference(cpu):
    weights = MlaWeights()
    assert make(weights).mla_fused == "reference"
    assert make(weights, mla_fused="decode_v1").mla_fused == "decode_v1"
    assert set(MLA_FUSED_COUNTERS) == set(fused.COUNTERS)
    assert set(MLA_FUSED_COUNTERS) <= set(make(weights).calls)
    for bad in ("v1", "on", None):
        with pytest.raises(ValueError, match="MLA fused policy"):
            make(weights, mla_fused=bad)


@pytest.mark.parametrize("tokens", [2045, 2300])
def test_decode_v1_off_metal_declines_and_runs_the_reference_bitwise(cpu, tokens):
    weights = MlaWeights()
    want, want_cache, _ = run_layers(weights, tokens, 4, {})
    got, got_cache, model = run_layers(weights, tokens, 4, {"mla_fused": "decode_v1"})
    assert all(same(g, w) for g, w in zip(got, want, strict=True))
    assert cache_bits(got_cache) == cache_bits(want_cache)
    assert model.calls["mla_fused_declined"] == 8
    assert model.calls["mla_fused"] == model.calls["mla_fused_dense"] == model.calls["mla_fused_pool"] == 0


def test_kernel_sources_substitute_every_placeholder_and_pin_constants():
    src = fused.kernel_sources(1e-5)
    for token in ("RMS_EPS", "INV128", "LN_EPS", "C_DIM", "C_HEADS", "SCORE_TG"):
        assert all(token not in body for body in src.values()), token
    assert "uint axis = tg == 0 ? 1536 : 512;" in src["prep"]
    assert "raw[128 + c]" in src["prep"] and "* 512 + c]" in src["gather"]
    assert float.fromhex(fused._hexf(1 / 128)[:-1]) == 1 / 128
    for value in (1e-5, 1e-6, 128**-0.5, 32**-0.5):
        assert float.fromhex(fused._hexf(value)[:-1]) == float(np.float32(value))
    # Separate MLX kernels in the reference are never contracted here; the
    # rms accumulation is the AOT build's fused multiply-add.
    assert src["prep"].count("metal::fma") == 1
    assert all("fp contract(off)" in src[k] for k in ("prep", "pool", "score"))


def test_pool_cache_pool_fn_replaces_only_the_suffix_computation(cpu):
    rng = np.random.default_rng(5)
    raw = mx.array(rng.normal(size=(64, 256)).astype(np.float32))
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    rows = lambda a, z: raw[a:z]
    want, got, seen = Glm53PoolCache(), Glm53PoolCache(), []

    def pool_fn(r, a):
        seen.append(r.shape)
        return indexer.pool_fixed_order(r, a)

    for total in (37, 40, 41, 44, 64):
        w = want.extend(rows, total, ape)
        g = got.extend(rows, total, ape, pool_fn=pool_fn)
        assert same(g, w) and (got.complete, got.last_from, got.last_computed) == (
            want.complete, want.last_from, want.last_computed)
    assert seen == [(36, 256), (4, 256), (4, 256), (20, 256)]


def test_masked_attention_entry_equals_the_valid_mask_entry_bitwise(cpu):
    rng = np.random.default_rng(6)
    q = mx.array(rng.normal(size=(1, 8, 1, 32)).astype(np.float32))
    latent = mx.array(rng.normal(size=(1, 1, 50, 32)).astype(np.float32))
    ids = mx.array(np.array([[[0, 3, 7, -1, 12, -1, 49]]], np.int32))
    rows, valid = sa.gather_latent(latent, ids)
    mask = mx.where(valid, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))
    assert same(sa.gathered_attention_masked(q, rows, mask), sa.gathered_attention(q, rows, valid))


def _adapter(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-mla-fused", policy)
    return adapter


@pytest.mark.skipif(not mx.metal.is_available(), reason="decode_v1 requires Metal at load")
def test_adapter_key_is_validated_requires_arm4_and_extends_the_profile_only_off_the_reference():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    default, reference = _adapter({}), _adapter({"glm53_mla_fused": "reference"})
    on = _adapter({"glm53_mla_fused": "decode_v1"})
    # decode_v1 is the default wherever it can engage (Pierre, 2026-11-01).
    assert default.kernel_policy["glm53_mla_fused"] == "decode_v1"
    assert default.profile == on.profile and "mla_fused" not in reference.profile
    assert ":mla_fused=decode_v1" in on.profile and on.profile != reference.profile
    # Under non-arm4 sparse laws the default falls back instead of refusing.
    legacy = _adapter({"glm53_indexer_select": "topk_v1"})
    assert legacy.kernel_policy["glm53_mla_fused"] == "reference"
    for bad in ({"glm53_mla_fused": "decode_v2"}, {"glm53_mla_fused": "on"}):
        with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
            object.__new__(Glm53GGUFAdapter)._load("/nonexistent", bad)
    assert ":sparse_attention=fused_decode_v1" in _adapter(
        {"glm53_mla_fused": "decode_v1", "glm53_sparse_attention": "fused_decode_v1"}).profile
    for law in ({"glm53_indexer_select": "topk_v1"}, {"glm53_indexer_pools": "incremental_v1_audit"},
                {"glm53_sparse_attention": "mask_reference", "glm53_sparse_prefill": "dense_reference",
                 "glm53_indexer_select": "argsort", "glm53_indexer_pools": "recompute"}):
        with pytest.raises(ValueError, match="glm53_mla_fused=decode_v1 requires"):
            object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {"glm53_mla_fused": "decode_v1", **law})


# ---- Metal (MLX2_RUN_GPU_TESTS=1 only) ------------------------------------------

metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@metal
def test_gpu_each_kernel_is_bitwise_its_op_chain(gpu):
    fused._check_prep(1e-5)
    fused._check_prep(1e-6)
    fused._check_pool()
    fused._check_score()
    fused._check_gather()
    assert fused.ready(1e-5) and fused._GATE["error"] is None


@metal
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("tokens,warm", [(2041, False), (3001, True), (3001, False)])
def test_gpu_decode_across_the_threshold_and_pool_completions_is_bitwise(gpu, tokens, warm, group):
    """2041 -> 2061: seven dense steps (T <= 2048), the first selecting step
    pools 512 pools from 0 (op chain), then one-pool kernel completions at
    T = 2052, 2056, 2060. 3001 warm: completions from the first step; 3001
    cold: a restored cache (no pools) pools 750 at once first."""
    steps = 20
    weights = MlaWeights()
    policy = {"mla_group": group}
    want, want_cache, ref = run_layers(weights, tokens, steps, policy, warm_pools=warm)
    got, got_cache, model = run_layers(weights, tokens, steps, {**policy, "mla_fused": "decode_v1"},
                                       warm_pools=warm)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert same(g, w), i
    assert cache_bits(got_cache) == cache_bits(want_cache)
    dense = sum(1 for t in range(tokens + 1, tokens + steps + 1) if t <= 2048)
    # The first selecting step of a cold cache pools from 0 with the op chain
    # (never a completion step here); every completion is one kernel pool.
    completions = sum(1 for t in range(tokens + 1, tokens + steps + 1) if t > 2048 and t % 4 == 0)
    assert model.calls["mla_fused_declined"] == 0
    assert model.calls["mla_fused_dense"] == 2 * dense
    assert model.calls["mla_fused"] == 2 * (steps - dense)
    assert model.calls["mla_fused_pool"] == 2 * completions
    for name in ("pooled_indexer", "indexer_select_kernel", "indexer_pool_extend",
                 "indexer_pools_computed", "indexer_pool_recompute_full", "sparse_attention_gathered",
                 "sparse_gathered_rows", "mla", "kv_append_ring"):
        assert model.calls[name] == ref.calls[name], name


@metal
def test_gpu_declines_prompt_rows_wider_rows_and_other_laws(gpu):
    weights = MlaWeights()
    for policy in ({"indexer_select": "topk_v1"}, {"sparse_attention": "mask_reference",
                                                   "indexer_select": "argsort", "indexer_pools": "recompute"}):
        _, _, model = run_layers(weights, 2300, 1, {**policy, "mla_fused": "decode_v1"})
        assert model.calls["mla_fused_declined"] == 2 and model.calls["mla_fused"] == 0
    model = make(weights, mla_fused="decode_v1")
    cache = live_cache(model, 2300)
    x = mx.array(np.random.default_rng(3).normal(size=(1, 2, 4)).astype(np.float32))
    mx.eval(model._mla(x, cache[1], "blk.1", frontier=2300))
    with sparse_attention_phase("prompt"):
        mx.eval(model._mla(x[:, :1], cache[1], "blk.1", frontier=2302))
    # Prompt rows and multi-row verify are not candidates, not declines.
    assert model.calls["mla_fused_not_candidate"] == 2 and model.calls["mla_fused_declined"] == 0
    assert model.calls["mla_fused"] == 0


@metal
@pytest.mark.parametrize("tokens", [2044, 3001])
def test_gpu_full_forwards_under_compile_regions_are_bitwise(gpu, tokens):
    """Whole decode forwards (KDA layers compiled, MLA attention eager)."""
    weights = MlaWeights()
    logits = {}
    states = {}
    for name, policy in (("reference", {}), ("decode_v1", {"mla_fused": "decode_v1"})):
        model = make(weights, **policy)
        model.compile_regions = "v1"
        cache = live_cache(model, tokens, warm_pools=tokens > 2048)
        out = []
        for step in range(9):
            y = model(mx.array([[1 + step % 7]]), cache)
            mx.eval(y)
            out.append(y)
        logits[name], states[name] = out, cache_bits(cache)
        if name == "decode_v1":
            assert model.calls["mla_fused"] + model.calls["mla_fused_dense"] == 2 * 9
            assert model.calls["mla_fused_declined"] == 0
    assert all(same(g, w) for g, w in zip(logits["decode_v1"], logits["reference"], strict=True))
    assert states["decode_v1"] == states["reference"]


@metal
def test_gpu_failed_gate_declines_and_reports(gpu, monkeypatch):
    monkeypatch.setitem(fused._GATE, "error", "ValueError: synthetic")
    _, _, model = run_layers(MlaWeights(), 2300, 1, {"mla_fused": "decode_v1"})
    assert model.calls["mla_fused_declined"] == 2
    assert fused.route_status()["gate_error"] == "ValueError: synthetic"
