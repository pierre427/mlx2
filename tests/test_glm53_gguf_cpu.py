"""Synthetic CPU operators/state contracts; no checkpoint/GPU execution."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_gguf as gguf
from mlx2.runtime.models import glm53_gguf_indexer as indexer
from mlx2.runtime.models import glm53_sparse_attention as sparse
from mlx2.runtime.models.cache import (
    CacheList,
    KVCache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel, Glm53KDACache, Glm53MLACache
from mlx2.runtime.models.glm53_gguf_indexer import pooled_selection


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class Weights:
    def __init__(self):
        self.rng = np.random.default_rng(75)
        self.arrays = {}

    def array(self, name):
        if name not in self.arrays:
            if "conv1d" in name:
                value = self.rng.normal(0, 0.2, (8192, 1, 4)).astype(np.float32)
            elif name.endswith("ssm_a"):
                value = self.rng.uniform(1, 8, 64).astype(np.float32)
            elif name.endswith("ssm_dt.bias"):
                value = self.rng.normal(-2, 0.2, 8192).astype(np.float32)
            elif name.endswith("ssm_norm.weight"):
                value = np.ones(128, dtype=np.float32)
            elif name.endswith("kpool_ape"):
                value = self.rng.normal(0, 0.1, (4, 128)).astype(np.float32)
            elif name.endswith("k_norm.weight"):
                value = np.ones(128, dtype=np.float32)
            elif name.endswith("k_norm.bias"):
                value = np.zeros(128, dtype=np.float32)
            elif name.endswith("attn_q_a_norm.weight"):
                value = np.ones(32, dtype=np.float32)
            elif name.endswith("attn_kv_a_norm.weight"):
                value = np.ones(512, dtype=np.float32)
            else:
                raise AssertionError(name)
            self.arrays[name] = mx.array(value)
        return self.arrays[name]

    def linear(self, name):
        def call(x):
            if name not in self.arrays:
                if "indexer.attn_q_b" in name:
                    out = 4096
                elif "indexer.proj" in name:
                    out = 32
                elif "indexer" in name or name.endswith(
                    ("ssm_f_a.weight", "ssm_g_a.weight")
                ):
                    out = 128
                elif name.endswith("ssm_beta.weight"):
                    out = 64
                elif name.endswith("attn_output.weight"):
                    out = 4
                elif name.endswith("attn_q_a.weight"):
                    out = 32
                elif name.endswith("attn_kv_a_mqa.weight"):
                    out = 512
                elif name.endswith("attn_q_b.weight"):
                    out = 64 * 256
                else:
                    out = 8192
                value = self.rng.normal(0, 0.05, (x.shape[-1], out)).astype(np.float32)
                self.arrays[name] = mx.array(value)
            return x @ self.arrays[name]

        return call

    def grouped(self, x, name, *, groups):
        # The GGUF F32 grouped law: einsum over (groups, out, in) weights.
        if name not in self.arrays:
            out = 512 if "attn_k_b" in name else 256
            value = self.rng.normal(0, 0.05, (groups, out, x.shape[-1])).astype(np.float32)
            self.arrays[name] = mx.array(value)
        return mx.einsum("...gi,goi->...go", x, self.arrays[name])

    def multi_linear(self, inputs, names, *, joined=1):
        # Equal to the separate projections (the K-block contract).
        self.multi_calls = getattr(self, "multi_calls", []) + [tuple(names)]
        return [self.linear(name)(x) for x, name in zip(inputs, names, strict=True)]


def test_kda_chunk_and_incremental_state_match():
    w = Weights()
    m = Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5},
        w,
        identity="synthetic",
        math_policy="reference",
    )
    x = mx.array(np.random.default_rng(8).normal(size=(1, 5, 4)).astype(np.float32))
    chunk = Glm53KDACache("synthetic")
    one = Glm53KDACache("synthetic")
    a = m._kda(x, chunk, "blk.0")
    b = mx.concatenate(
        [m._kda(x[:, i : i + 1], one, "blk.0") for i in range(5)], axis=1
    )
    mx.eval(a, b, chunk.state, one.state)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5, rtol=2e-5)
    for left, right in zip(chunk.state, one.state):
        np.testing.assert_allclose(
            np.array(left), np.array(right), atol=2e-6, rtol=2e-5
        )
    assert chunk.offset == one.offset == 5


def test_kda_checkpoint_roundtrip_and_wrong_abi_refused(tmp_path):
    c = Glm53KDACache("synthetic@source-policy")
    c.conv = mx.zeros((1, 3, 24576))
    c.recurrent = mx.zeros((1, 64, 128, 128))
    c.offset = 17
    path = tmp_path / "cache.safetensors"
    save_prompt_cache(str(path), [c])
    restored = load_prompt_cache(str(path))[0]
    assert type(restored) is Glm53KDACache
    assert restored.meta_state == c.meta_state
    assert restored.cache_operations_contract.rollback is None
    assert restored.size() == 17
    with pytest.raises(ValueError, match="cannot trim"):
        restored.trim(1)
    with pytest.raises(ValueError, match="geometry"):
        Glm53KDACache.from_state((mx.zeros((1, 1, 2)), c.recurrent), c.meta_state)
    with pytest.raises(ValueError, match="identity"):
        restored.restore_exact_decode_checkpoint(("other",))


def test_indexer_pool_boundary_and_per_query_causality():
    w = Weights()
    rng = np.random.default_rng(33)
    cache = KVCache()
    x = mx.array(rng.normal(size=(1, 2049, 4)).astype(np.float32))
    qr = mx.array(rng.normal(size=(1, 2049, 6)).astype(np.float32))
    ids = pooled_selection(x, qr, cache, w, "blk.3")
    a = np.array(ids)
    # Each query has its own tail and future complete pools remain hidden.
    for t in [0, 3, 100, 2047, 2048]:
        actual = a[0, t]
        actual = actual[actual >= 0]
        np.testing.assert_array_equal(np.sort(actual), np.arange(t + 1))
    more = mx.array(rng.normal(size=(1, 7, 4)).astype(np.float32))
    more_q = mx.array(rng.normal(size=(1, 7, 6)).astype(np.float32))
    selected = np.array(pooled_selection(more, more_q, cache, w, "blk.3"))
    for i in range(7):
        actual = selected[0, i]
        actual = actual[actual >= 0]
        assert np.all(actual <= 2049 + i)
        assert len(np.unique(actual)) == len(actual)
        assert len(actual) == 2048 + (2050 + i) % 4
    # Restore raw indexer state, then append: no pooled Python attribute survives.
    restored = KVCache.from_state(cache.state, cache.meta_state)
    next_x = more[:, :1]
    next_q = more_q[:, :1]
    left = pooled_selection(next_x, next_q, cache, w, "blk.3")
    right = pooled_selection(next_x, next_q, restored, w, "blk.3")
    np.testing.assert_array_equal(np.array(left), np.array(right))


def test_topology_and_memory_are_glm_specific_and_revision_bound():
    from mlx2.adapters.glm53_gguf import CACHE_LAYOUT, Glm53GGUFAdapter
    from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget
    from mlx2.runtime.attention_topology import apcv2_cache_layout_fingerprint
    from mlx2.runtime.cow_cache import _build_layer_segment_manifest
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    config = {
        "layer_types": [
            "linear_attention" if (i + 1) % 4 else "deepseek_sparse_attention"
            for i in range(45)
        ]
    }
    a = object.__new__(Glm53GGUFAdapter)
    a.config = config
    a.identity = {"fingerprint": "synthetic-revision"}
    a.layout = CACHE_LAYOUT
    assert apcv2_cache_layout_fingerprint(a)[0] == CACHE_LAYOUT
    budget = Glm53CacheBudget.from_config(config)
    ast = Glm53CacheBudget.from_config(config, tree_rows=16)
    assert ast.project(4096) > 8 * budget.project(4096)
    assert budget.snapshot_bytes(257) > budget.snapshot_bytes(256)
    c = Glm53KDACache("synthetic")
    c.offset = 17
    c.conv = mx.zeros((1, 3, 24576))
    c.recurrent = mx.zeros((1, 64, 128, 128))
    m = Glm53MLACache()
    for kv, dim in zip(m.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 17, dim)), mx.zeros((1, 1, 17, 0)))
    manifest = _build_layer_segment_manifest([c, m], None, covered_tokens=17)
    assert any(s.key.kind.value == "gdn_recurrent" for s in manifest.segments)
    with pytest.raises(ValueError):
        Glm53CacheBudget.from_config(config, mtp=True)


def test_composite_cache_keeps_abi_when_extracting_a_request(tmp_path):
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    c = Glm53MLACache()
    for kv, dim in zip(c.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 9, dim)), mx.zeros((1, 1, 9, 0)))
    extracted = c.extract(0)
    assert type(extracted) is Glm53MLACache
    assert extracted.offset == 9
    path = tmp_path / "hybrid.safetensors"
    save_prompt_cache(str(path), [extracted])
    loaded = load_prompt_cache(str(path))[0]
    assert type(loaded) is Glm53MLACache and loaded.offset == 9


def test_codec_copy_preserves_composite_class_and_operations_contract():
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache
    from mlx2.runtime.recurrent_state_codec import (
        RecurrentStateCodecPolicy,
        encode_prompt_cache,
    )

    cache = Glm53MLACache()
    encoded, _ = encode_prompt_cache([cache], RecurrentStateCodecPolicy(enabled=True))
    assert type(encoded[0]) is Glm53MLACache
    assert encoded[0] is not cache
    assert encoded[0].cache_operations_contract == cache.cache_operations_contract


def test_mla_checkpoint_refuses_wrong_shape_and_inconsistent_frontiers():
    from mlx2.runtime.models.glm53_gguf import Glm53MLACache

    cache = Glm53MLACache()
    for kv, width in zip(cache.caches, (512, 256)):
        kv.update_and_fetch(mx.zeros((1, 1, 3, width)), mx.zeros((1, 1, 3, 0)))
    state = cache.state
    with pytest.raises(ValueError, match="geometry"):
        Glm53MLACache.from_state(
            [(mx.zeros((1, 1, 3, 511)), state[0][1]), state[1]], cache.meta_state
        )
    cache.caches[1].offset = 2
    with pytest.raises(ValueError, match="frontiers"):
        Glm53MLACache.from_state(state, cache.meta_state)


def test_preprojected_kda_and_separate_output_preserve_reference_state():
    weights = Weights()
    model = Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5},
        weights,
        identity="synthetic",
        math_policy="reference",
    )
    x = mx.array(np.random.default_rng(84).normal(size=(1, 3, 4)).astype(np.float32))
    ordinary = Glm53KDACache("synthetic")
    prepared = Glm53KDACache("synthetic")
    expected = model._kda(x, ordinary, "blk.0")
    project = lambda value, suffix: weights.linear("blk.0." + suffix)(value)
    projected = {
        "qkv": mx.concatenate(
            [project(x, "attn_" + name + ".weight") for name in ("q", "k", "v")],
            axis=-1,
        ),
        "a": project(project(x, "ssm_f_a.weight"), "ssm_f_b.weight"),
        "beta": project(x, "ssm_beta.weight"),
        "gate": project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight"),
    }
    core = model._kda(
        x, prepared, "blk.0", _projected=projected, _return_pre_output=True
    )
    actual = project(core, "attn_output.weight")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    for a, b in zip(ordinary.state, prepared.state):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    assert ordinary.offset == prepared.offset == 3


# ---------------------------------------------------------------------------
# Sparse DSA P3: policies, pooled-key plane, _mla seams, budget, call sites
# (docs/design/glm53-sparse-dsa.md sections 6.5, 10, 12). Synthetic weights,
# synthetic committed caches of ~2K tokens, a handful of query rows.

PREFIX = "blk.3"
SPARSE = {"indexer_pools": "incremental_v1", "indexer_select": "topk_v1", "sparse_attention": "gathered_v1"}


def bits(a):
    return np.array(a).view(np.uint32)


def mla_model(weights, **policy):
    return Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5}, weights, identity="synthetic",
        math_policy="reference", **policy,
    )


def committed(seed, tokens):
    """A Glm53MLACache holding ``tokens`` random committed rows (twins share a seed)."""
    rng = np.random.default_rng(seed)
    cache = Glm53MLACache()
    for plane, width in zip(cache.caches, (512, 256), strict=True):
        rows = mx.array(rng.normal(size=(1, 1, tokens, width)).astype(np.float32))
        plane.update_and_fetch(rows, mx.zeros((1, 1, tokens, 0)))
    mx.eval([c.state for c in cache.caches])
    return cache


def hidden(seed, length):
    return mx.array(np.random.default_rng(seed).normal(size=(1, length, 4)).astype(np.float32))


def legacy_mla(model, x, cache, prefix):
    """The pre-P3 _mla body (ungrouped projections), frozen for bit identity."""
    b, length, _ = x.shape
    project = lambda value, name: model._linear(value, prefix, name)
    qa = project(x, "attn_q_a.weight")
    qr = gguf.rms(qa, model.weights.array(prefix + ".attn_q_a_norm.weight"), model.args.rms_norm_eps)
    q = project(qr, "attn_q_b.weight").reshape(b, length, 64, 256)
    latent = gguf.rms(
        project(x, "attn_kv_a_mqa.weight"),
        model.weights.array(prefix + ".attn_kv_a_norm.weight"),
        model.args.rms_norm_eps,
    )[:, None]
    latent, _ = cache[0].update_and_fetch(latent, mx.zeros((b, 1, length, 0), dtype=x.dtype))
    selected = pooled_selection(x, qr, cache[1], model.weights, prefix, projected={})
    q = model.weights.grouped(q, prefix + ".attn_k_b.weight", groups=64).transpose(0, 2, 1, 3)
    total = latent.shape[2]
    pieces = []
    for start in range(0, length, 128):
        stop = min(start + 128, length)
        seen = total - length + stop
        keys = latent[:, :, :seen]
        if selected is None:
            pos = total - length + mx.arange(start, stop)
            mask = (mx.arange(seen)[None, :] <= pos[:, None])[None, None]
        else:
            ids = selected[:, start:stop]
            safe = mx.where(ids >= 0, ids, seen)
            mask = mx.zeros((b, stop - start, seen + 1), dtype=mx.bool_)
            mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[..., :seen][:, None]
        pieces.append(mx.fast.scaled_dot_product_attention(
            q[:, :, start:stop], keys, keys, scale=256**-0.5, mask=mask))
    out = mx.concatenate(pieces, axis=2).transpose(0, 2, 1, 3)
    out = model.weights.grouped(out, prefix + ".attn_v_b.weight", groups=64)
    return project(out.reshape(b, length, -1), "attn_output.weight")


@pytest.mark.parametrize(
    "tokens,length,prompt",
    [(2050, 1, False), (2046, 4, True), (2046, 4, False), (9, 3, True), (9, 130, True)],
)
def test_default_policies_reproduce_the_pre_split_mla_bit_for_bit(tokens, length, prompt):
    from contextlib import nullcontext

    weights = Weights()
    model = mla_model(weights)
    x = hidden(1, length)
    want_cache, got_cache = committed(2, tokens), committed(2, tokens)
    want = legacy_mla(model, x, want_cache, PREFIX)
    with gguf.sparse_attention_phase("prompt") if prompt else nullcontext():
        got = model._mla(x, got_cache, PREFIX, frontier=tokens)
    assert np.array_equal(bits(got), bits(want))
    for a, b in zip(got_cache.caches, want_cache.caches, strict=True):
        assert np.array_equal(bits(a.keys[:, :, : a.offset]), bits(b.keys[:, :, : b.offset]))
    assert got_cache.pools is None  # recompute attaches no derived plane
    selecting = tokens + length > 2048
    assert model.calls["pooled_indexer"] == model.calls["sparse_attention_mask"] == int(selecting)
    assert model.calls["indexer_select_argsort"] == model.calls["indexer_pool_recompute_full"] == int(selecting)
    assert model.calls["sparse_attention_gathered"] == model.calls["sparse_prefill_gathered_blocks"] == 0
    # The pre-output seam is the same core the layered tree projects itself.
    core_cache = committed(2, tokens)
    core = model._mla(hidden(1, length), core_cache, PREFIX, frontier=tokens, _return_pre_output=True)
    assert np.array_equal(bits(model._linear(core, PREFIX, "attn_output.weight")), bits(want))


def test_policy_values_and_combinations_are_validated():
    weights = Weights()
    assert mla_model(weights).indexer_pools == "recompute"
    for bad in (
        {"indexer_pools": "incremental"},
        {"sparse_attention": "gathered"},
        {"indexer_select": "topk_v1"},  # reference pooling runs only with argsort
        {"indexer_pools": "incremental_v1", "sparse_attention": "gathered_v1"},  # argsort ids
        {"indexer_pools": "incremental_v1", "sparse_prefill": "gathered_v1_all"},
    ):
        with pytest.raises(ValueError):
            mla_model(weights, **bad)
    mla_model(weights, **SPARSE, sparse_prefill="gathered_v1_8k")
    mla_model(weights, indexer_pools="recompute_fixed_order")  # fixed-order law with argsort


def adapter_policy(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-sparse-dsa", policy)
    return adapter


def test_adapter_profile_suffix_only_off_the_reference_law_and_refused_combinations():
    from mlx2.adapters.glm53_gguf import (
        SPARSE_DEFAULTS,
        SPARSE_KEYS,
        Glm53GGUFAdapter,
        _check_sparse_combination,
    )

    base = Glm53GGUFAdapter.profile + ":ordinary:hc=reference:ast_target=serial:draft_activation=target"
    # Decode glue levers are on by default (2026-10-31); this test pins the reference law.
    glue = {"glm53_router": "reference", "glm53_kda_small": "reference",
            "glm53_kda_fused": "reference", "glm53_compile_regions": "off",
            "glm53_mla_fused": "reference"}
    reference = adapter_policy({**glue, **{key: values[0] for key, values in SPARSE_KEYS.items()}})
    assert reference.profile == base
    assert all(reference.kernel_policy[key] == values[0] for key, values in SPARSE_KEYS.items())
    # No sparse keys: the unqualified arm4 default (Pierre, 2026-10-31), its own profile.
    default = adapter_policy(dict(glue))
    assert all(default.kernel_policy[key] == value for key, value in SPARSE_DEFAULTS.items())
    assert default.profile == base + (
        ":indexer_pools=incremental_v1:indexer_select=topk_v1_kernel"
        ":sparse_attention=gathered_v1:sparse_prefill=gathered_v1_8k"
    )
    sparse_policy = adapter_policy({**glue,
        "glm53_indexer_pools": "incremental_v1", "glm53_indexer_select": "topk_v1",
        "glm53_sparse_attention": "gathered_v1", "glm53_sparse_prefill": "gathered_v1_all",
    })
    assert sparse_policy.profile == base + (
        ":indexer_pools=incremental_v1:indexer_select=topk_v1"
        ":sparse_attention=gathered_v1:sparse_prefill=gathered_v1_all"
    )
    with pytest.raises(ValueError, match="unsupported"):
        object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {"glm53_sparse_attention": "dense"})

    def combo(**changes):
        policy = {"glm53_ast_target": "serial", **{k: v[0] for k, v in SPARSE_KEYS.items()}}
        policy.update({"glm53_" + k: v for k, v in changes.items()})
        _check_sparse_combination(policy)

    combo(indexer_pools="incremental_v1", indexer_select="topk_v1", sparse_attention="gathered_v1")
    with pytest.raises(ValueError, match="chronological"):
        combo(sparse_attention="gathered_v1")
    with pytest.raises(ValueError, match="batched_v1 requires"):
        combo(indexer_pools="incremental_v1", indexer_select="topk_v1", sparse_attention="gathered_v1",
              tree_rows_attention="batched_v1")  # serial target
    with pytest.raises(ValueError, match="tails_v1 requires"):
        combo(ast_target="layered", indexer_pools="recompute_fixed_order", indexer_select="topk_v1",
              sparse_attention="gathered_v1", tree_frontier="tails_v1")
    # Until the layered tree declares them, valid tree values fail closed
    # rather than silently running the per-row clone path.
    from mlx2.adapters import glm53_layer_tree

    if "batched_v1" not in getattr(glm53_layer_tree, "SUPPORTED_TREE_ROWS_ATTENTION", ()):
        with pytest.raises(ValueError, match="not implemented by the layered tree"):
            combo(ast_target="layered", indexer_pools="incremental_v1", indexer_select="topk_v1",
                  sparse_attention="gathered_v1", tree_rows_attention="batched_v1")


def oracle(q, latent, ids):
    q, latent, ids = np.array(q, np.float64), np.array(latent, np.float64)[0, 0], np.array(ids)[0]
    out = np.full(q.shape, np.nan)
    for row, selected in enumerate(ids):
        keys = latent[selected[selected >= 0]]
        s = q[0, :, row] @ keys.T * 256**-0.5
        p = np.exp(s - s.max(-1, keepdims=True))
        out[0, :, row] = (p / p.sum(-1, keepdims=True)) @ keys
    return out


def test_mask_and_gathered_consumers_share_ids_and_meet_the_oracle_bound():
    weights = Weights()
    mask = mla_model(weights, indexer_pools="incremental_v1", indexer_select="topk_v1")
    gathered = mla_model(weights, **SPARSE)
    x = hidden(3, 2)
    qa, ids_a, source_a = mask._mla_prepare(x, committed(4, 2050), PREFIX, frontier=2050)
    qb, ids_b, source_b = gathered._mla_prepare(x, committed(4, 2050), PREFIX, frontier=2050)
    # 9.1: the selection stage is shared, so the id sets are identical.
    assert np.array_equal(np.array(ids_a), np.array(ids_b))
    assert np.array_equal(bits(qa), bits(qb)) and np.array_equal(bits(source_a), bits(source_b))
    ids = np.array(ids_a)[0]
    for row, pos in enumerate((2050, 2051)):
        valid = ids[row][ids[row] >= 0]
        assert valid.max() <= pos and len(np.unique(valid)) == len(valid)
        assert np.all(np.diff(valid[: 512 * 4]) > 0)  # topk_v1: chronological
    want = oracle(qa, source_a, ids_a)
    out_mask = np.array(mask._mla_attend(qa, ids_a, source_a), np.float64)
    out_gathered = np.array(gathered._mla_attend(qb, ids_b, source_b), np.float64)
    # 9.5 at the model's geometry (2051 keys x 512 dims, small-norm outputs):
    # both consumers are new-vs-oracle within the same absolute bound.
    for got in (out_mask, out_gathered):
        assert np.max(np.abs(got - want)) <= 2e-5
    assert mask.calls["sparse_attention_mask"] == 1 and mask.calls["sparse_attention_gathered"] == 0
    assert gathered.calls["sparse_attention_gathered"] == 1 and gathered.calls["sparse_attention_mask"] == 0
    assert gathered.calls["sparse_gathered_rows"] == 2 * indexer.WIDTH


def test_empty_selection_contract_is_nan_for_gathered_consumers_only():
    """Design 4.1 erratum: the mask reference gives a finite uniform average."""
    weights = Weights()
    model = mla_model(weights, **SPARSE)
    rng = np.random.default_rng(5)
    latent = mx.array(rng.normal(size=(1, 1, 40, 512)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, 64, 2, 512)).astype(np.float32))
    ids = np.full((1, 2, indexer.WIDTH), -1, np.int32)
    ids[0, 0, :8] = np.arange(8)
    ids = mx.array(ids)
    masked = np.array(model._mla_masked(q, latent, ids, 2, 40))
    assert np.isfinite(masked).all()
    # boolean mask -> finfo.min scores -> every visible key weighs the same
    np.testing.assert_allclose(masked[0, :, 1], np.broadcast_to(np.array(latent)[0, 0].mean(0), (64, 512)),
                               atol=1e-5)
    for out in (model._mla_attend(q, ids, latent), sparse.sparse_prefill(q, latent, ids)):
        out = np.array(out)
        assert np.isnan(out[0, :, 1]).all() and np.isfinite(out[0, :, 0]).all()


def test_prompt_phase_splits_a_chunk_per_query_position():
    weights = Weights()
    model = mla_model(weights, **SPARSE, sparse_prefill="gathered_v1_all")
    x = hidden(6, 4)
    with gguf.sparse_attention_phase("prompt"):
        q, ids, source = model._mla_prepare(x, committed(7, 2046), PREFIX, frontier=2046)
        out = model._mla_attend(q, ids, source)
    # total 2050, threshold 2049: positions 2046, 2047 dense; 2048, 2049 gathered.
    assert model.calls["sparse_prefill_dense_pieces"] == 1 and model.calls["sparse_attention_mask"] == 1
    assert model.calls["sparse_prefill_gathered_blocks"] == 1
    assert model.calls["sparse_attention_gathered"] == 0  # prompt phase never takes the decode consumer
    dense = model._mla_masked(q, source, ids, 2, 2050)
    gathered = sparse.sparse_decode(q[:, :, 2:], source, ids[:, 2:], policy="gathered_v1")
    assert np.array_equal(bits(out[:, :, :2]), bits(dense))
    assert np.array_equal(bits(out[:, :, 2:]), bits(gathered))
    # Outside the prompt phase the decode policy decides, for every row.
    decode = model._mla_attend(q, ids, source)
    assert np.array_equal(bits(decode), bits(sparse.sparse_decode(q, source, ids, policy="gathered_v1")))


def test_one_token_prompt_chunk_equals_the_decode_step_bitwise():
    weights = Weights()
    model = mla_model(weights, **SPARSE, sparse_prefill="gathered_v1_all")
    x = hidden(8, 1)
    with gguf.sparse_attention_phase("prompt"):
        prompt = model._mla(x, committed(9, 2060), PREFIX, frontier=2060)
    decode = model._mla(x, committed(9, 2060), PREFIX, frontier=2060)
    assert np.array_equal(bits(prompt), bits(decode))
    assert model.calls["sparse_prefill_gathered_blocks"] == model.calls["sparse_attention_gathered"] == 1


def test_select_block_only_bounds_prompt_phase_topk():
    weights = Weights()
    model = mla_model(weights, **SPARSE)
    assert model._select_block() is None
    with gguf.sparse_attention_phase("prompt"):
        assert model._select_block() == gguf.SELECT_BLOCK
        assert mla_model(weights)._select_block() is None  # argsort selects globally
    with pytest.raises(ValueError), gguf.sparse_attention_phase("decode"):
        pass


def pools_equal_recompute(cache, ape):
    total = cache.caches[1].offset
    want = indexer.pool_fixed_order(cache.caches[1].keys[0, 0, : 4 * (total // 4)], ape)
    got = cache.pools.pooled[0, : cache.pools.complete]
    return cache.pools.complete == total // 4 and np.array_equal(bits(got), bits(want))


def test_incremental_pools_count_and_stay_equal_to_recompute_under_audit():
    weights = Weights()
    model = mla_model(weights, **dict(SPARSE, indexer_pools="incremental_v1_audit"))
    cache = committed(10, 2048)
    model._mla(hidden(11, 1), cache, PREFIX, frontier=2048)  # 2049: first extend, pools [0, 512)
    assert model.calls["indexer_pool_recompute_full"] == 1 and model.calls["indexer_pools_computed"] == 512
    for step, length in enumerate((1, 1, 1, 5, 1, 2)):
        model._mla(hidden(12 + step, length), cache, PREFIX, frontier=cache.caches[1].offset)
    ape = weights.array(PREFIX + ".indexer.kpool_ape")
    assert cache.caches[1].offset == 2060 and pools_equal_recompute(cache, ape)
    assert model.calls["indexer_pool_audit"] == 7
    assert model.calls["indexer_pools_computed"] == 515 and model.calls["indexer_pool_extend"] >= 2
    assert model.calls["indexer_select_topk"] == 7 and model.calls["indexer_select_argsort"] == 0


def test_topk_kernel_policy_declines_to_the_counted_ops_path_on_cpu():
    weights = Weights()
    kernel = mla_model(weights, **dict(SPARSE, indexer_select="topk_v1_kernel"))
    ops = mla_model(weights, **SPARSE)
    x = hidden(13, 1)
    a = kernel._mla(x, committed(14, 2050), PREFIX, frontier=2050)
    b = ops._mla(x, committed(14, 2050), PREFIX, frontier=2050)
    assert np.array_equal(bits(a), bits(b))
    assert kernel.calls["indexer_select_kernel_declined"] == 1 and kernel.calls["indexer_select_kernel"] == 0


def test_first_use_kernel_gate_launches_are_not_counted_as_selection(monkeypatch):
    gate = []

    def ready():  # a gate that launched synthetic cases, then declined
        if not gate:
            gate.append(True)
            indexer._SCHEDULED["select_kernel"] += 8
        return False

    monkeypatch.setattr(indexer, "select_kernel_ready", ready)
    model = mla_model(Weights(), **dict(SPARSE, indexer_select="topk_v1_kernel"))
    model._mla(hidden(13, 1), committed(14, 2050), PREFIX, frontier=2050)
    assert model.calls["indexer_select_kernel"] == 0
    assert model.calls["indexer_select_kernel_declined"] == model.calls["indexer_select_topk"] == 1


def test_gathered_decode_runs_in_row_groups_bitwise_equal_to_one_call():
    model = mla_model(Weights(), **SPARSE)
    x = hidden(31, 10)
    q, ids, source = model._mla_prepare(x, committed(32, 2050), PREFIX, frontier=2050)
    grouped = model._mla_attend(q, ids, source)
    assert model.calls["sparse_attention_gathered"] == 2  # rows 0-7, 8-9
    whole = sparse.sparse_decode(q, source, ids, policy="gathered_v1")
    assert np.array_equal(bits(grouped), bits(whole))


def test_audit_policy_raises_on_a_nonfinite_pool_score(monkeypatch):
    model = mla_model(Weights(), **dict(SPARSE, indexer_pools="incremental_v1_audit"))
    original = indexer.pool_scores

    def scores(q, w, pooled):
        out = original(q, w, pooled)
        return mx.concatenate([mx.full((*out.shape[:2], 1), mx.nan), out[..., 1:]], axis=-1)

    monkeypatch.setattr(indexer, "pool_scores", scores)
    with pytest.raises(ValueError, match="nonfinite pool score"):
        model._mla(hidden(33, 1), committed(34, 2050), PREFIX, frontier=2050)


def test_every_rewind_mutator_is_overridden_and_cuts_pools():
    overrides = {"trim", "trim_to_position", "trim_ragged"}
    observers = {"is_trimmable", "supports_ragged_trim", "preflight_ragged_trim", "snap_trim_position"}
    names = {n for n in dir(CacheList) if "trim" in n and callable(getattr(CacheList, n))}
    assert names <= overrides | observers, names - overrides - observers
    assert overrides <= set(Glm53MLACache.__dict__)
    assert {"state", "meta_state", "nbytes"} <= set(Glm53MLACache.__dict__)


def test_rewind_then_append_never_leaves_a_stale_pool():
    weights = Weights()
    ape = weights.array(PREFIX + ".indexer.kpool_ape")

    def advanced():
        model = mla_model(weights, **SPARSE)
        cache = committed(15, 2050)
        model._mla(hidden(16, 2), cache, PREFIX, frontier=2050)  # 2052: complete 513
        assert cache.pools.complete == 513
        return model, cache

    replacement = hidden(17, 1)
    model, cache = advanced()
    cache.trim_to_position(2051, 1)
    assert cache.pools.complete == 512
    model._mla(replacement, cache, PREFIX, frontier=2051)
    assert pools_equal_recompute(cache, ape)
    # Regression control: the inherited CacheList rewind leaves pool 512 stale.
    model, stale = advanced()
    CacheList.trim_to_position(stale, 2051, 1)
    model._mla(replacement, stale, PREFIX, frontier=2051)
    assert stale.pools.complete == 513 and not pools_equal_recompute(stale, ape)
    # trim cuts as well.
    model, cache = advanced()
    cache.trim(1)
    assert cache.pools.complete == 512
    # A metadata-only rewind drops the plane; the next call re-pools from 0.
    model, cache = advanced()
    classes, _ = cache.meta_state
    cache.meta_state = (classes, [("2051",), ("2051",)])
    assert cache.pools is None and cache.offset == 2051
    full = model.calls["indexer_pool_recompute_full"]
    model._mla(replacement, cache, PREFIX, frontier=2051)
    assert pools_equal_recompute(cache, ape) and model.calls["indexer_pool_recompute_full"] == full + 1


def test_trim_ragged_delegates_then_cuts(monkeypatch):
    calls = []

    def ragged(self, n, *, validate=True):
        calls.append((n, validate))
        for plane in self.caches:
            plane.trim(n[0])
        return n

    monkeypatch.setattr(CacheList, "trim_ragged", ragged)
    cache = committed(18, 2052)
    cache.pools = indexer.Glm53PoolCache()
    cache.pools.complete = 513
    assert cache.trim_ragged([3]) == [3] and calls == [([3], True)]
    assert cache.pools.complete == 512


def test_serialized_restore_drops_pools_and_recomputes_identically():
    weights = Weights()
    live_model, restored_model = mla_model(weights, **SPARSE), mla_model(weights, **SPARSE)
    live, other = committed(19, 2050), committed(19, 2050)
    for model, cache in ((live_model, live), (restored_model, other)):
        model._mla(hidden(20, 3), cache, PREFIX, frontier=2050)
    restored = Glm53MLACache.from_state(other.state, other.meta_state)
    assert restored.pools is None and type(restored) is Glm53MLACache
    other.state = other.state  # the state setter drops the plane too
    assert other.pools is None
    x = hidden(21, 1)
    a = live_model._mla(x, live, PREFIX, frontier=2053)
    before = restored_model.calls["indexer_pool_recompute_full"]
    b = restored_model._mla(x, restored, PREFIX, frontier=2053)
    assert np.array_equal(bits(a), bits(b))
    assert restored_model.calls["indexer_pool_recompute_full"] == before + 1
    assert np.array_equal(bits(live.pools.pooled[:, :513]), bits(restored.pools.pooled[:, :513]))


def test_saved_checkpoint_carries_no_pool_plane(tmp_path):
    model = mla_model(Weights(), **SPARSE)
    cache = committed(35, 2050)
    model._mla(hidden(36, 2), cache, PREFIX, frontier=2050)
    assert cache.pools.complete == 513
    path = tmp_path / "glm53-mla.safetensors"
    save_prompt_cache(str(path), [cache])
    loaded = load_prompt_cache(str(path))[0]
    assert type(loaded) is Glm53MLACache and loaded.pools is None and loaded.offset == 2052
    assert loaded.nbytes == sum(c.nbytes for c in loaded.caches)


def test_descriptor_restore_and_extract_keep_private_consistent_pools():
    from mlx2.adapters.glm53_ast_target import clone

    weights = Weights()
    ape = weights.array(PREFIX + ".indexer.kpool_ape")
    model = mla_model(weights, **SPARSE)
    cache = committed(22, 2050)
    model._mla(hidden(23, 1), cache, PREFIX, frontier=2050)  # 2051: complete 512
    snapshot = clone([cache])[0]
    extracted = cache.extract(0)
    for copy_ in (snapshot, extracted):
        assert copy_.pools is not cache.pools and copy_.pools.complete == 512
        assert pools_equal_recompute(copy_, ape)
    assert cache.nbytes == sum(c.nbytes for c in cache.caches) + cache.pools.pooled.nbytes
    # Siblings and the parent complete pool 512 with different fourth rows,
    # each privately: no append writes into another's buffers.
    model._mla(hidden(24, 1), snapshot, PREFIX, frontier=2051)
    model._mla(hidden(25, 1), extracted, PREFIX, frontier=2051)
    model._mla(hidden(37, 1), cache, PREFIX, frontier=2051)
    for copy_ in (snapshot, extracted, cache):
        assert copy_.offset == 2052 and pools_equal_recompute(copy_, ape)
    rows = [bits(c.caches[1].keys[0, 0, 2051]) for c in (snapshot, extracted, cache)]
    assert not np.array_equal(rows[0], rows[1]) and not np.array_equal(rows[1], rows[2])
    pools = [bits(c.pools.pooled[:, 512]) for c in (snapshot, extracted, cache)]
    assert not np.array_equal(pools[0], pools[1]) and not np.array_equal(pools[1], pools[2])


@pytest.mark.parametrize("tokens", [2050, 100])
def test_frontier_state_is_duck_typed_and_equals_the_cache_path(tokens):
    weights = Weights()
    model = mla_model(weights, **SPARSE)
    x = hidden(26, 1)
    want = model._mla(x, committed(27, tokens), PREFIX, frontier=tokens)
    base = committed(27, tokens)
    frontier = sparse.Glm53MLAFrontier(base, tokens)
    got = model._mla(x, frontier, PREFIX, frontier=tokens)
    assert np.array_equal(bits(got), bits(want))
    assert frontier.offset == tokens + 1 and base.offset == tokens  # base never written
    with pytest.raises(ValueError, match="frontiers require"):
        mla_model(weights, indexer_pools="recompute_fixed_order", indexer_select="topk_v1",
                  sparse_attention="gathered_v1")._mla(x, sparse.Glm53MLAFrontier(base, tokens), PREFIX)


def test_grouped_projections_decide_selection_from_the_integer_frontier():
    weights = Weights()
    plain = mla_model(weights)
    grouped = mla_model(weights, mla_group=True)
    for tokens, names in ((2050, 4), (9, 3)):
        x = hidden(28, 1)
        want = plain._mla(x, committed(29, tokens), PREFIX, frontier=tokens)
        weights.multi_calls = []
        got = grouped._mla(x, committed(29, tokens), PREFIX, frontier=tokens)
        assert np.array_equal(bits(got), bits(want))
        assert len(weights.multi_calls[0]) == names
    with pytest.raises(ValueError, match="frontier differs"):
        plain._mla(hidden(28, 1), committed(29, 9), PREFIX, frontier=8)


def test_q8_latent_knob_is_refused_with_gathered_paths():
    weights = Weights()
    reference = mla_model(weights)
    reference.mla_kv_q8 = True
    assert isinstance(reference._mla_cache().caches[0], gguf.Glm53LatentQ8Cache)
    for policy in (SPARSE, dict(SPARSE, sparse_attention="mask_reference", sparse_prefill="gathered_v1_8k")):
        model = mla_model(weights, **policy)
        model.mla_kv_q8 = True
        with pytest.raises(ValueError, match="dense-mask research knob"):
            model._mla_cache()
    fixed = mla_model(weights, indexer_pools="incremental_v1", indexer_select="topk_v1")
    fixed.mla_kv_q8 = True
    assert isinstance(fixed._mla_cache().caches[0], gguf.Glm53LatentQ8Cache)


def test_sparse_workspace_budget_is_policy_bound_and_planning_books_the_maximum():
    import inspect

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter
    from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget

    config = {"layer_types": ["linear_attention" if (i + 1) % 4 else "deepseek_sparse_attention"
                              for i in range(45)]}
    block = 32 * (2051 * 512 + 2 * 64 * 2051) * 4
    group = 8 * 2051 * 512 * 4
    default = Glm53CacheBudget.from_config(config)
    assert default.sparse_workspace_bytes == 0
    assert default.snapshot_bytes(4096) == default.fixed_bytes + 11 * (4096 + 256) * 768 * 4  # unchanged
    policy = {"glm53_sparse_prefill": "gathered_v1_8k", "glm53_sparse_attention": "gathered_v1",
              "glm53_indexer_pools": "incremental_v1"}
    bound = Glm53CacheBudget.from_config(config, kernel_policy=policy)
    assert bound.sparse_workspace_bytes == block + group
    assert bound.snapshot_bytes(4096) == bound.fixed_bytes + 11 * (4096 + 256) * 800 * 4
    assert bound.prefill_transient_bytes(4096, 32) == default.prefill_transient_bytes(4096, 32) + block + group
    assert bound.prefill_transient_bytes(4096, 0) == 0
    assert list(inspect.signature(Glm53CacheBudget.prefill_transient_bytes).parameters) == [
        "self", "context_tokens", "chunk_rows"]
    planning = Glm53GGUFAdapter.planning_cache_budget(config, mtp=False)
    assert planning.sparse_workspace_bytes == block + group and planning.indexed_floats_per_token == 800
    with pytest.raises(ValueError):
        Glm53CacheBudget.from_config(config, kernel_policy={"glm53_sparse_attention": "dense"})
    adapter = object.__new__(Glm53GGUFAdapter)
    adapter.config, adapter.draft_model, adapter.external_policy = config, None, {}
    adapter.kernel_policy = {"glm53_sparse_prefill": "gathered_v1_all", "glm53_sparse_attention": "mask_reference",
                             "glm53_indexer_pools": "recompute_fixed_order"}
    budget = adapter.cache_budget(mtp=False)
    assert budget.sparse_workspace_bytes == block and budget.indexed_floats_per_token == 768


class HeadWeights:
    """Just enough of the provider for the NextN call site."""

    def embedding(self, tokens):
        return mx.ones((*tokens.shape, 4096))

    def array(self, name):
        return mx.ones((4096,))

    def linear(self, name):
        return (lambda x: x[..., :4096]) if "eh_proj" in name else (lambda x: x[..., :8])


def test_mtp_head_passes_its_own_integer_frontier_and_fails_closed():
    from types import SimpleNamespace

    from mlx2.runtime.models.glm53_gguf_mtp import Glm53MTPHead

    seen = []

    def mla(x, cache, prefix, *, frontier=None):
        seen.append((prefix, frontier, type(frontier)))
        return mx.zeros_like(x)

    target = SimpleNamespace(weights=HeadWeights(), args=SimpleNamespace(rms_norm_eps=1e-5), _mla=mla,
                             _mlp=lambda x, prefix, index: mx.zeros_like(x))
    head = Glm53MTPHead(target)
    cache = [committed(30, 5)]
    head(mx.ones((1, 2, 4096)), mx.array([[1, 2]]), cache)
    assert seen == [("blk.45", 5, int)]
    cache[0].caches[0].offset = mx.array(5)
    with pytest.raises(ValueError, match="integer MLA frontier"):
        head(mx.ones((1, 2, 4096)), mx.array([[1, 2]]), cache)


def test_trunk_passes_the_kda_frontier_and_prompt_forwards_enter_the_phase(monkeypatch):
    phases, frontiers = [], []

    class TrunkWeights:
        expert_prefill_policy = dense_prefill_policy = "reference"

        def embedding(self, tokens):
            return mx.zeros((*tokens.shape, 4096))

        def array(self, name):
            return mx.ones((4096,))

        def linear(self, name):
            return lambda x: x[..., :8]

    model = Glm53GGUFModel(
        {"layer_types": ["linear_attention", "deepseek_sparse_attention"], "rms_norm_eps": 1e-5,
         "max_position_embeddings": 65536},
        TrunkWeights(), identity="synthetic",
    )
    monkeypatch.setattr(gguf, "hc_expand", lambda branch, h, post, comb, *, fused: h)
    monkeypatch.setattr(model, "_hc_in", lambda h, half, norm: (h[..., 0, :], None, None))
    monkeypatch.setattr(model, "_mlp", lambda x, prefix, index: x)

    def kda(x, cache, prefix):
        cache.offset += x.shape[1]
        return x

    def mla(x, cache, prefix, *, frontier=None):
        frontiers.append(frontier)
        phases.append(gguf.sparse_phase())
        for plane in cache.caches:
            plane.offset += x.shape[1]
        return x

    monkeypatch.setattr(model, "_kda", kda)
    monkeypatch.setattr(model, "_mla", mla)
    cache = model.make_cache()
    cache[0].offset = 7
    for plane in cache[1].caches:
        plane.offset = 7
    model(mx.array([[1, 2]]), cache)
    model.prefill_forward(mx.array([[3]]), cache)
    model.prefill_body(mx.array([[4]]), cache, [1])
    assert frontiers == [7, 9, 10] and phases == [None, "prompt", "prompt"]
    assert gguf.sparse_phase() is None
    with pytest.raises(RuntimeError), gguf.sparse_attention_phase("prompt"):
        raise RuntimeError("the phase is restored on the way out")
    assert gguf.sparse_phase() is None


def test_adapter_declares_the_think_close_token_for_thinking_budgets():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return {"</think>": [154842]}.get(text, [1, 2])

    adapter = object.__new__(Glm53GGUFAdapter)
    adapter.tokenizer = Tok()
    assert adapter.thinking_close_token_ids() == (154842,)

    class Split(Tok):
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    adapter.tokenizer = Split()
    assert adapter.thinking_close_token_ids() is None
