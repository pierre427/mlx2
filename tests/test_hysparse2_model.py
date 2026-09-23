"""HySparse2 prototype: selection, KV reuse, bridging and early-exit prefill.

CPU only (conftest). Oracles are independent of the code under test: a NumPy
stable sort for token selection, explicit masked dense attention for sparse
attention, and the model's own every-layer reference path for early exit.
"""

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_map

from mlx2.runtime.models import hysparse2_cost as cost
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.hysparse2 import Model, ModelArgs, select_tokens


def tiny_args(**overrides):
    base = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=8,
        swa_rope_dims=8,
        self_decoder_layers=["swa", "swa", "full", "swa", "swa", "swa"],
        cross_decoder_layers=["full", "sparse", "sparse", "full", "sparse"],
        sparse_topk=6,
        sparse_local_window=4,
    )
    base.update(overrides)
    return ModelArgs(**base)


def tiny_model(seed=0, **overrides):
    mx.random.seed(seed)
    model = Model(tiny_args(**overrides))
    # Non-trivial sinks, norms and hyper-connection weights.
    model.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.5 if p.ndim == 1 else p,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    return model


def cache_views(cache, window):
    out = []
    for c in cache:
        if isinstance(c, RotatingKVCache):
            k = c._temporal_order(c.keys)[..., -window:, :]
            v = c._temporal_order(c.values)[..., -window:, :]
        else:
            k = c.keys[..., : c.offset, :]
            v = c.values[..., : c.offset, :]
        out.append((c.offset, np.array(k), np.array(v)))
    return out


def assert_caches_close(a, b, window, atol=2e-5):
    for (oa, ka, va), (ob, kb, vb) in zip(cache_views(a, window), cache_views(b, window)):
        assert oa == ob
        np.testing.assert_allclose(ka, kb, atol=atol)
        np.testing.assert_allclose(va, vb, atol=atol)


# -- token selection ---------------------------------------------------------


def oracle_select(scores, qpos, budget, window):
    """Stable sort on (forced first, score desc, position asc)."""
    T = scores.shape[-1]
    K = min(budget + window, T)
    out = []
    for row, p in zip(scores, qpos):
        keys = []
        for t in range(T):
            visible = t <= p
            forced = visible and t > p - window
            rank = 0 if forced else (1 if visible else 2)
            keys.append((rank, -row[t] if rank == 1 else 0.0, t))
        chosen = sorted(t for (_, _, t) in sorted(keys)[:K])
        out.append(chosen)
    return np.array(out)


def test_select_tokens_matches_stable_sort_oracle_with_ties():
    rng = np.random.default_rng(3)
    T, L = 40, 5
    scores = rng.integers(0, 4, size=(1, L, T)).astype(np.float32)  # many ties
    qpos = np.array([39, 30, 20, 11, 9])
    idx, valid = select_tokens(mx.array(scores), mx.array(qpos), budget=6, window=4)
    idx = np.array(idx)[0]
    np.testing.assert_array_equal(idx, oracle_select(scores[0], qpos, 6, 4))
    assert (np.diff(idx, axis=-1) > 0).all()  # ascending, no duplicates
    np.testing.assert_array_equal(np.array(valid)[0], idx <= qpos[:, None])


def test_select_tokens_tie_break_does_not_rely_on_argpartition_order(monkeypatch):
    """On mlx 39400a0d4 argpartition happens to keep the lowest index among
    ties (200/200 probes on CPU and GPU), so the explicit tie-break is a
    guard. Pin it against an argpartition that prefers the highest index."""
    real = mx.argpartition

    def adversarial(a, kth, axis=-1):
        x = np.array(a)
        order = np.lexsort((-np.arange(x.shape[-1]) * np.ones_like(x), x), axis=-1)
        return mx.array(order.astype(np.int32))

    monkeypatch.setattr(mx, "argpartition", adversarial)
    rng = np.random.default_rng(7)
    scores = rng.integers(0, 3, size=(1, 4, 50)).astype(np.float32)
    qpos = np.array([49, 40, 33, 25])
    idx, _ = select_tokens(mx.array(scores), mx.array(qpos), budget=7, window=5)
    np.testing.assert_array_equal(np.array(idx)[0], oracle_select(scores[0], qpos, 7, 5))
    monkeypatch.setattr(mx, "argpartition", real)


def test_select_tokens_forces_local_window_over_higher_scores():
    T = 32
    scores = np.zeros((1, 1, T), np.float32)
    scores[0, 0, :10] = 100.0  # far tokens dominate by score
    idx, _ = select_tokens(mx.array(scores), mx.array([31]), budget=6, window=4)
    idx = set(np.array(idx)[0, 0].tolist())
    assert {28, 29, 30, 31} <= idx  # forced window survives
    assert set(range(6)) <= idx  # budget goes to the lowest-index top scorers
    assert len(idx) == 10


def test_select_tokens_short_context_selects_everything_visible():
    idx, valid = select_tokens(mx.zeros((1, 3, 7)), mx.array([6, 4, 0]), budget=6, window=4)
    np.testing.assert_array_equal(np.array(idx)[0], np.tile(np.arange(7), (3, 1)))
    np.testing.assert_array_equal(np.array(valid)[0], np.arange(7)[None] <= np.array([[6], [4], [0]]))


# -- KV reuse: sparse attention ------------------------------------------------


def test_sparse_attention_equals_dense_attention_restricted_to_selection():
    model = tiny_model()
    layer = model.cross_layers[1].self_attn
    rng = np.random.default_rng(0)
    B, L, T, D, H = 1, 3, 20, 16, 4
    x = mx.array(rng.normal(size=(B, L, 32)).astype(np.float32))
    keys = mx.array(rng.normal(size=(B, 1, T, D)).astype(np.float32))
    values = mx.array(rng.normal(size=(B, 1, T, D)).astype(np.float32))
    qpos = mx.array([17, 18, 19])
    idx, valid = select_tokens(mx.array(rng.normal(size=(B, L, T)).astype(np.float32)), qpos, 6, 4)
    got = np.array(layer(x, keys, values, idx, valid))

    q = np.array(layer.q_proj(x)).reshape(B, L, H, D)
    k, v, sinks = np.array(keys)[0, 0], np.array(values)[0, 0], np.array(layer.sinks)
    idx_np, valid_np = np.array(idx)[0], np.array(valid)[0]
    attn = np.zeros((L, H, D), np.float32)
    for l in range(L):
        allowed = np.zeros(T, bool)
        allowed[idx_np[l][valid_np[l]]] = True
        for h in range(H):
            s = np.where(allowed, k @ q[0, l, h] * D**-0.5, -np.inf)
            s = np.concatenate([[sinks[h]], s])
            p = np.exp(s - s.max())
            p = p / p.sum()
            attn[l, h] = p[1:] @ v
    gate = 1 / (1 + np.exp(-np.array(layer.gate_proj(x))[0]))
    want = np.array(layer.o_proj(mx.array((attn.reshape(L, -1) * gate)[None])))
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_sparse_budget_binds_on_long_context():
    """Falsifier: a path that silently attended densely would still pass the
    equivalence tests below, so the budget must visibly change the output."""
    toks = mx.array(np.random.default_rng(1).integers(0, 97, (1, 48)))
    sparse = tiny_model(seed=2)
    dense = tiny_model(seed=2, sparse_topk=1000)
    a = sparse(toks, sparse.make_cache())[:, -1]
    b = dense(toks, dense.make_cache())[:, -1]
    assert float(mx.abs(a - b).max()) > 1e-3


# -- KV bridging ---------------------------------------------------------------


def test_bridge_projects_the_input_of_the_self_full_layer():
    """Eq. (1): K_j = Proj_K_j(H_i^self), H_i^self the FA layer's input. Built
    by hand from the first two SWA layers, independent of Model.__call__."""
    model = tiny_model(seed=6)
    toks = mx.array(np.random.default_rng(6).integers(0, 97, (1, 25)))
    cache = model.make_cache()
    model(toks, cache)
    h = model.embed_tokens(toks)
    for layer in model.self_layers[:2]:
        h = layer(h, RotatingKVCache(max_size=8))
    for j, layer_index in enumerate((0, 3)):
        attn = model.cross_layers[layer_index].self_attn
        src = attn.bridge_norm(h)
        want_k = attn.k_proj(src).reshape(1, 25, 1, 16).transpose(0, 2, 1, 3)
        want_v = attn.v_proj(src).reshape(1, 25, 1, 16).transpose(0, 2, 1, 3)
        got = cache[6 + j]
        np.testing.assert_allclose(np.array(got.keys[..., :25, :]), np.array(want_k), atol=1e-5)
        np.testing.assert_allclose(np.array(got.values[..., :25, :]), np.array(want_v), atol=1e-5)


def test_bridged_caches_do_not_depend_on_cross_decoder_compute():
    toks = mx.array(np.random.default_rng(5).integers(0, 97, (1, 30)))
    model = tiny_model(seed=4)
    ref = model.make_cache()
    model(toks, ref)
    for layer in model.cross_layers:
        layer.self_attn.q_proj.weight = mx.zeros_like(layer.self_attn.q_proj.weight)
        layer.mlp.down_proj.weight = mx.zeros_like(layer.mlp.down_proj.weight)
    changed = model.make_cache()
    model(toks, changed)
    assert_caches_close(ref, changed, window=8, atol=0)


def test_cache_topology_has_no_sparse_layer_entries():
    model = tiny_model()
    cache = model.make_cache()
    assert [type(c) for c in cache] == [RotatingKVCache] * 2 + [KVCache] + [
        RotatingKVCache
    ] * 3 + [KVCache] * 2
    assert model.args.bridge_sources == [0, 0]


# -- early-exit prefill ----------------------------------------------------------


@pytest.mark.parametrize("hc_streams", [1, 3])
@pytest.mark.parametrize(
    "length,chunk",
    [(5, 4), (22, 8), (23, 8), (60, 16), (61, 61), (90, 7)],
)
def test_early_exit_prefill_matches_every_layer_reference(length, chunk, hc_streams):
    model = tiny_model(seed=length, hc_streams=hc_streams)
    toks = mx.array(np.random.default_rng(length).integers(0, 97, (1, length)))
    ref_cache = model.make_cache()
    ref = model(toks, ref_cache)[:, -1]
    ex_cache = model.make_cache()
    ex = model.prefill(toks, ex_cache, chunk_size=chunk)[:, -1]
    np.testing.assert_allclose(np.array(ex), np.array(ref), atol=2e-5)
    assert_caches_close(ref_cache, ex_cache, window=8)

    suffix = model.args.prefill_suffix_rows
    assert suffix == 8 + 2 * 7
    stats = model.stats
    assert stats.cross_rows == 1
    assert stats.bridge_rows == length
    assert stats.self_rows[:3] == [length] * 3
    if length > suffix:
        assert stats.self_full_query_rows == [suffix]
        assert stats.self_rows[3:] == [suffix, suffix - 7, suffix - 14]
    else:
        assert stats.self_full_query_rows == [length]

    # Decode continues identically from either cache.
    ya = yb = mx.argmax(ref, axis=-1)
    for _ in range(12):
        la = model(ya[:, None], ref_cache)[:, -1]
        lb = model(yb[:, None], ex_cache)[:, -1]
        np.testing.assert_allclose(np.array(lb), np.array(la), atol=2e-5)
        ya, yb = mx.argmax(la, axis=-1), mx.argmax(lb, axis=-1)
        assert ya.item() == yb.item()


def test_paper_exit_without_suffix_bound_matches_reference():
    model = tiny_model(seed=21)
    toks = mx.array(np.random.default_rng(21).integers(0, 97, (1, 75)))
    ref_cache = model.make_cache()
    ref = model(toks, ref_cache)[:, -1]
    ex_cache = model.make_cache()
    ex = model.prefill(toks, ex_cache, chunk_size=16, suffix_bound=False)[:, -1]
    np.testing.assert_allclose(np.array(ex), np.array(ref), atol=2e-5)
    assert_caches_close(ref_cache, ex_cache, window=8)
    assert model.stats.self_rows == [75] * 6
    assert model.stats.self_full_query_rows == [75]
    assert model.stats.cross_rows == 1


def test_early_exit_prefill_resumes_from_a_prefix_cache():
    """APC-style: a cached prefix, then an early-exit prefill of the rest."""
    model = tiny_model(seed=9)
    toks = mx.array(np.random.default_rng(9).integers(0, 97, (1, 70)))
    full = model.make_cache()
    want = model(toks, full)[:, -1]
    for split, chunk in ((20, 16), (60, 4), (69, 32)):
        resumed = model.make_cache()
        model(toks[:, :split], resumed)
        got = model.prefill(toks[:, split:], resumed, chunk_size=chunk)[:, -1]
        np.testing.assert_allclose(np.array(got), np.array(want), atol=2e-5)
        assert_caches_close(full, resumed, window=8)


def test_multi_row_verify_matches_sequential_decode():
    """Speculative verify: each row needs its own causal top-k and window."""
    model = tiny_model(seed=11)
    toks = mx.array(np.random.default_rng(11).integers(0, 97, (1, 40)))
    draft = mx.array([[3, 14, 15, 92]])
    seq = model.make_cache()
    model.prefill(toks, seq)
    step = [model(draft[:, i : i + 1], seq)[:, -1] for i in range(4)]
    batch = model.make_cache()
    model.prefill(toks, batch)
    rows = model(draft, batch)
    for i in range(4):
        np.testing.assert_allclose(np.array(rows[:, i]), np.array(step[i]), atol=2e-5)


def test_multiple_self_full_layers_and_bridge_sources():
    args = dict(
        self_decoder_layers=["swa", "full", "swa", "full", "swa", "swa"],
        cross_decoder_layers=["full", "sparse", "full", "sparse", "full"],
        bridge_sources=[0, 1, 1],
    )
    model = tiny_model(seed=13, **args)
    toks = mx.array(np.random.default_rng(13).integers(0, 97, (1, 50)))
    ref_cache = model.make_cache()
    ref = model(toks, ref_cache)[:, -1]
    ex_cache = model.make_cache()
    ex = model.prefill(toks, ex_cache, chunk_size=9)[:, -1]
    np.testing.assert_allclose(np.array(ex), np.array(ref), atol=2e-5)
    assert_caches_close(ref_cache, ex_cache, window=8)
    # Only the last self FA layer is suffix-bounded: 2 SWA after it.
    assert model.stats.self_full_query_rows == [50, 8 + 7]


def test_invalid_layouts_fail_closed():
    with pytest.raises(ValueError, match="preceding full"):
        tiny_args(cross_decoder_layers=["sparse", "full"])
    with pytest.raises(ValueError, match="bridge_sources"):
        tiny_args(bridge_sources=[0, 5])
    with pytest.raises(ValueError, match="needs a full-attention"):
        tiny_args(self_decoder_layers=["swa", "swa"])
    with pytest.raises(ValueError, match="single-sequence"):
        model = tiny_model()
        model.prefill(mx.zeros((2, 4), mx.int32), model.make_cache())


# -- cost model against the paper -------------------------------------------------


def test_cost_model_reproduces_paper_kv_cache_at_one_million_tokens():
    """Figure 4 / Section 4.2: 2.69, 6.72 and 12.09 GB with FP8 KV at 1M."""
    got = {d.name: round(cost.kv_cache_bytes(d, 2**20) / 1e9, 2) for d in cost.DESIGNS}
    assert got == {"HySparse2": 2.69, "HySparse": 6.72, "Hybrid SWA": 12.09}


def test_cost_model_reproduces_paper_prefill_ratios_at_one_million_tokens():
    """Section 4.2: 2.92x vs HySparse and 5.02x vs Hybrid SWA. The non-attention
    share is an assumption, so allow 5%."""
    f = {d.name: cost.prefill_flops_per_token(d, 2**20) for d in cost.DESIGNS}
    assert f["HySparse"] / f["HySparse2"] == pytest.approx(2.92, rel=0.05)
    assert f["Hybrid SWA"] / f["HySparse2"] == pytest.approx(5.02, rel=0.05)


def test_suffix_bound_removes_the_quadratic_prefill_term():
    short = cost.prefill_flops_per_token(cost.HYSPARSE2, 2**15, suffix_bound=True)
    long = cost.prefill_flops_per_token(cost.HYSPARSE2, 2**20, suffix_bound=True)
    assert long <= short * 1.01  # per-token cost stops growing with T
    paper = cost.prefill_flops_per_token(cost.HYSPARSE2, 2**20)
    assert paper / long > 10
