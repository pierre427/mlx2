"""GLM-5.3 gathered sparse attention laws (design 4.2, 4.3, 7); synthetic CPU only."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_sparse_attention as sa
from mlx2.runtime.models.glm53_gguf import Glm53MLACache

HEADS, DIM = 4, 16


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("GPU forbidden")))


def bits(a):
    return np.array(a).view(np.uint32)


def selection(rng, total, length, width=12, count=2, invalid_pool_row=None, empty_row=None):
    """expand_selection-shaped ids: ascending visible pools, then the tail, then -1."""
    ids = np.full((length, width), -1, dtype=np.int32)
    for l in range(length):
        pos = total - length + l
        pools = np.sort(rng.choice((pos + 1) // 4, size=count, replace=False))
        row = [p * 4 + j for p in pools for j in range(4)]
        if l == invalid_pool_row:
            row[:4] = [-1] * 4
        row += list(range(pos + 1 - (pos + 1) % 4, pos + 1))
        ids[l, : len(row)] = row
        if l == empty_row:
            ids[l] = -1
        assert ids[l].max() <= pos  # causality is an invariant of the ids
    return mx.array(ids[None])


def oracle(q, latent, ids):
    q, latent, ids = np.array(q, np.float64), np.array(latent, np.float64)[0, 0], np.array(ids)[0]
    out = np.full(q.shape, np.nan)
    for l, row in enumerate(ids):
        keys = latent[row[row >= 0]]
        if not len(keys):
            continue
        s = q[0, :, l] @ keys.T * sa.SCALE
        p = np.exp(s - s.max(-1, keepdims=True))
        out[0, :, l] = (p / p.sum(-1, keepdims=True)) @ keys
    return out


def mask_reference(q, latent, ids):
    """Today's _mla mask consumer for one piece over every key."""
    total = latent.shape[2]
    safe = mx.where(ids >= 0, ids, total)
    mask = mx.zeros((1, ids.shape[1], total + 1), dtype=mx.bool_)
    mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[..., :total][:, None]
    return mx.fast.scaled_dot_product_attention(q, latent, latent, scale=sa.SCALE, mask=mask)


def within_oracle(got, want):
    got = np.array(got, np.float64)
    assert np.max(np.abs(got - want)) <= 1e-5
    assert np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want**2)) <= 1e-6


def data(rng, total=40, length=5):
    latent = mx.array(rng.normal(size=(1, 1, total, DIM)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, HEADS, length, DIM)).astype(np.float32))
    return q, latent


def test_gathered_and_mask_consumers_are_within_the_float64_oracle_bound():
    rng = np.random.default_rng(1)
    q, latent = data(rng)
    ids = selection(rng, 40, 5, invalid_pool_row=2)
    want = oracle(q, latent, ids)
    gathered = sa.gathered_attention(q, *sa.gather_latent(latent, ids))
    within_oracle(gathered, want)
    within_oracle(mask_reference(q, latent, ids), want)
    counters = {}
    assert np.array_equal(bits(sa.sparse_decode(q, latent, ids, policy="gathered_v1", counters=counters)),
                          bits(gathered))
    assert counters == {"sparse_attention_gathered": 1, "sparse_gathered_rows": 5 * 12}


def test_a_query_without_a_valid_id_yields_a_nan_row_and_nothing_raises():
    rng = np.random.default_rng(2)
    q, latent = data(rng)
    ids = selection(rng, 40, 5, empty_row=3)
    for out in (
        sa.sparse_decode(q, latent, ids, policy="gathered_v1"),
        sa.sparse_prefill(q, latent, ids, block=2),
        sa.rows_attention(q.transpose(2, 1, 0, 3), [(latent, ids[:, l : l + 1]) for l in range(5)],
                          policy="gathered_v1").transpose(2, 1, 0, 3),
    ):
        out = np.array(out)
        assert np.isnan(out[0, :, 3]).all()
        assert np.isfinite(np.delete(out, 3, axis=2)).all()


def test_prefill_blocks_are_alignment_free_and_equal_decode_rows_bitwise():
    rng = np.random.default_rng(3)
    q, latent = data(rng, total=48, length=10)
    ids = selection(rng, 48, 10)
    whole = sa.sparse_decode(q, latent, ids, policy="gathered_v1")
    for block in (1, 3, 4, 10, 32):
        counters = {}
        out = sa.sparse_prefill(q, latent, ids, block=block, counters=counters)
        assert np.array_equal(bits(out), bits(whole)), block
        assert counters["sparse_prefill_gathered_blocks"] == -(-10 // block)
    for l in range(10):
        one = sa.sparse_decode(q[:, :, l : l + 1], latent, ids[:, l : l + 1], policy="gathered_v1")
        assert np.array_equal(bits(one), bits(whole[:, :, l : l + 1]))
    with pytest.raises(ValueError, match="block"):
        sa.sparse_prefill(q, latent, ids, block=0)


def tree_view(latent, split):
    return sa.TreeLatentView(latent[0, 0, :split], latent[0, 0, split:], split)


def test_tree_latent_view_gather_equals_gather_from_the_concatenation():
    rng = np.random.default_rng(4)
    _, latent = data(rng, total=40)
    ids = selection(rng, 40, 5, invalid_pool_row=1)
    want_rows, want_valid = sa.gather_latent(latent, ids)
    for split in (0, 1, 36, 39, 40):
        rows, valid = sa.gather_latent(tree_view(latent, split), ids)
        assert np.array_equal(bits(rows), bits(want_rows)), split
        assert np.array_equal(np.array(valid), np.array(want_valid))
    with pytest.raises(ValueError, match="split"):
        sa.TreeLatentView(latent[0, 0, :30], latent[0, 0, 30:], 29)


@pytest.mark.parametrize("batched", [True, False])
def test_rows_attention_equals_one_row_decode_bitwise_with_ragged_sources(monkeypatch, batched):
    monkeypatch.setattr(sa, "BATCHED_ROWS_EXACT", batched)
    rng = np.random.default_rng(5)
    _, a = data(rng, total=40)
    _, b = data(rng, total=37)
    q_rows = mx.array(rng.normal(size=(4, HEADS, 1, DIM)).astype(np.float32))
    sources = [
        (a, selection(rng, 40, 1)),
        (b, selection(rng, 37, 1, invalid_pool_row=0)),
        (tree_view(a, 38), selection(rng, 40, 1)),
        (tree_view(b, 37), selection(rng, 37, 1)),
    ]
    counters = {}
    out = sa.rows_attention(q_rows, sources, policy="gathered_v1", counters=counters)
    for r, (latent, ids) in enumerate(sources):
        one = sa.sparse_decode(q_rows[r : r + 1], latent, ids, policy="gathered_v1")
        assert np.array_equal(bits(out[r : r + 1]), bits(one)), r
    key = "tree_rows_attention_batched" if batched else "tree_rows_attention_per_row_fallback"
    assert counters == {key: 1, "sparse_gathered_rows": 4 * 12}


def test_selection_ids_are_shared_by_both_consumers_and_policies_fail_closed():
    rng = np.random.default_rng(6)
    q, latent = data(rng)
    ids = selection(rng, 40, 5)
    for policy in ("mask_reference", "gathered", None):
        with pytest.raises(ValueError, match="gathered law"):
            sa.sparse_decode(q, latent, ids, policy=policy)
        with pytest.raises(ValueError, match="gathered law"):
            sa.rows_attention(q[:, :, :1], [(latent, ids[:, :1])], policy=policy)
    with pytest.raises(ValueError, match="every query row"):
        sa.sparse_decode(q, latent, ids[:, :4], policy="gathered_v1")
    with pytest.raises(ValueError, match="every query row"):
        sa.sparse_decode(q, latent, None, policy="gathered_v1")


def test_prefill_threshold_is_per_query_position():
    assert sa.prefill_gathered_start(2048, 2048, 2049) == 2048  # no ids at all
    assert sa.prefill_gathered_start(4096, 4096, sa.PREFILL_THRESHOLDS["gathered_v1_all"]) == 2048
    assert sa.prefill_gathered_start(8200, 100, 8192) == 91  # pos 8191 is the first gathered query
    assert sa.prefill_gathered_start(8200, 100, 2049) == 0
    assert sa.prefill_gathered_start(8000, 100, 8192) == 100
    assert sa.prefill_gathered_start(8200, 100, None) == 100
    # Cold (one chunk) and warm (restore at 8191, suffix chunk) agree per position.
    cold = sa.prefill_gathered_start(8300, 8300, 8192)
    warm = 8191 + sa.prefill_gathered_start(8300, 8300 - 8191, 8192)
    assert cold == warm == 8191


def test_fused_kernel_spec_launch_plan_and_cpu_refusal():
    spec = sa.kernel_spec()
    assert spec["partial"]["name"] == "mlx2_glm53_sparse_decode_v1"
    assert spec["partial"]["input_names"] == ["q", "base", "tail", "ids", "meta"]
    assert "simd_sum(part) * SCALE" in spec["partial"]["source"]
    assert "id < T0 ? base[" in spec["partial"]["source"]
    assert "isnan(score)" in spec["partial"]["source"]
    assert "acc / l" in spec["combine"]["source"]
    assert "0.0625f" in spec["partial"]["header"]
    assert sa.SCALE == 0.0625
    plan = sa.launch_plan(8, 2051)
    assert plan["splits"] == 17
    assert plan["partial"]["grid"] == (256 * 17, 8, 8) and plan["partial"]["threadgroup"] == (256, 1, 1)
    assert plan["partial"]["output_shapes"][2] == (8, 64, 17, 512)
    assert plan["combine"]["grid"] == (512, 64, 8)
    for rows in (0, 9):
        with pytest.raises(ValueError, match="1..8 rows"):
            sa.launch_plan(rows, 2051)
    q = mx.zeros((1, 64, 1, 512))
    latent = mx.zeros((1, 1, 2100, 512))
    ids = mx.zeros((1, 1, 2051), dtype=mx.int32)
    with pytest.raises(RuntimeError, match="requires Metal"):
        sa.sparse_decode(q, latent, ids, policy="fused_decode_v1")
    with pytest.raises(RuntimeError, match="requires Metal"):
        sa.rows_attention(q, [(latent, ids)], policy="fused_decode_v1")
    with pytest.raises(ValueError, match="64 heads"):
        sa.fused_decode_attention(mx.zeros((1, 4, 1, 512)), latent, ids)
    status = sa.route_status()["policies"]["fused_decode_v1"]
    assert status == {"implemented": True, "qualified": False, "selected_by_default": False,
                      "first_use_gate": "not_run"}


def emulated_launch(calls):
    """Ops stand-in for one fused launch (same ABI), to test the orchestration on CPU."""

    def launch(q_rows, base, tail, ids, meta):
        split, depth, width = meta.tolist()
        assert ids.shape == (q_rows.shape[0], width) and q_rows.shape[0] <= 8 and base.shape[0] == split
        calls.append((q_rows.shape[0], split, depth))
        source = sa.TreeLatentView(base, tail[:depth], split) if depth else base[None, None]
        out = sa.gathered_attention(q_rows.transpose(1, 0, 2)[None], *sa.gather_latent(source, ids[None]))
        return out[0].transpose(1, 0, 2)

    return launch


def test_fused_orchestration_groups_rows_of_one_source_and_launches_per_tree_row(monkeypatch):
    calls = []
    monkeypatch.setattr(sa, "_require_metal", lambda: None)
    monkeypatch.setattr(sa, "_fused_launch", emulated_launch(calls))
    monkeypatch.setattr(sa, "_FUSED_GATE", {"passed": True})
    rng = np.random.default_rng(12)
    latent = mx.array(rng.normal(size=(1, 1, 50, 512)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, 64, 11, 512)).astype(np.float32))
    ids = selection(rng, 50, 11, width=20, count=3, empty_row=9)
    counters = {}
    out = sa.sparse_decode(q, latent, ids, policy="fused_decode_v1", counters=counters)
    want = sa.sparse_decode(q, latent, ids, policy="gathered_v1")
    assert np.array_equal(bits(out), bits(want))  # grouping does not change a row's result
    assert np.isnan(np.array(out)[0, :, 9]).all()
    assert [c[0] for c in calls] == [8, 3] and all(c[1:] == (50, 0) for c in calls)
    assert counters == {"sparse_attention_fused": 1, "sparse_attention_fused_launches": 2}
    calls.clear()
    sources = [(latent, ids[:, :1]), (tree_view(latent, 47), ids[:, 1:2]), (tree_view(latent, 50), ids[:, 2:3])]
    q_rows = q[:, :, :3].transpose(2, 1, 0, 3)
    rows = sa.rows_attention(q_rows, sources, policy="fused_decode_v1")
    assert calls == [(1, 50, 0), (1, 47, 3), (1, 50, 0)]
    assert np.array_equal(bits(rows), bits(want[:, :, :3].transpose(2, 1, 0, 3)))


@pytest.mark.parametrize("error", [0.0, 1e-3])
def test_fused_first_use_gate_passes_an_exact_kernel_and_fails_closed_otherwise(monkeypatch, error):
    calls = []
    exact = emulated_launch(calls)
    monkeypatch.setattr(sa, "_require_metal", lambda: None)
    monkeypatch.setattr(sa, "_fused_launch", lambda *a: exact(*a) + error)
    monkeypatch.setattr(sa, "_FUSED_GATE", {"passed": None})
    q = mx.zeros((1, 64, 1, 512))
    latent = mx.zeros((1, 1, 2100, 512))
    ids = mx.zeros((1, 1, 2051), dtype=mx.int32)
    if not error:
        assert sa.self_check() is True and sa.route_status()["policies"]["fused_decode_v1"]["first_use_gate"] == "passed"
        assert {c[2] for c in calls} == {0, 5}  # plain source and a tree view were both gated
        sa.fused_decode_attention(q, latent, ids)
        return
    for _ in range(2):  # the verdict is sticky: no retry, no fallback law
        with pytest.raises(RuntimeError, match="first-use gate"):
            sa.fused_decode_attention(q, latent, ids)
    assert sa.route_status()["policies"]["fused_decode_v1"]["first_use_gate"] == "failed"


class Pools:
    """Stand-in for Glm53PoolCache's clone contract (a new descriptor per clone)."""

    def __init__(self, pooled=None, complete=0):
        self.pooled, self.complete = pooled, complete

    def clone(self):
        return Pools(None if self.pooled is None else mx.stop_gradient(self.pooled), self.complete)


def committed_base(rng, tokens):
    base = Glm53MLACache()
    for plane, width in zip(base.caches, (512, 256), strict=True):
        rows = mx.array(rng.normal(size=(1, 1, tokens, width)).astype(np.float32))
        plane.update_and_fetch(rows, mx.zeros((1, 1, tokens, 0)))
    base.pools = Pools(mx.array(rng.normal(size=(1, 64, 128)).astype(np.float32)), tokens // 4)
    mx.eval([c.state for c in base.caches])
    return base


def tail_rows(rng, depth):
    return [(mx.array(rng.normal(size=(1, 1, 1, 512)).astype(np.float32)),
             mx.array(rng.normal(size=(1, 1, 256)).astype(np.float32))) for _ in range(depth)]


def test_frontier_views_equal_the_concatenation():
    rng = np.random.default_rng(7)
    base = committed_base(rng, 300)
    frontier = sa.Glm53MLAFrontier(base, 300)
    assert frontier.pools is not base.pools and frontier.offset == 300
    for latent_row, raw_row in tail_rows(rng, 3):
        frontier.append(latent_row, raw_row)
    assert frontier.offset == 303 and frontier.depth == 3
    latent = mx.concatenate([base.caches[0].keys[0, 0, :300], frontier.tail_latent])
    raw = mx.concatenate([base.caches[1].keys[0, 0, :300], frontier.tail_raw])
    assert np.array_equal(bits(frontier.dense_keys()), bits(latent[None, None]))
    for start, stop in ((0, 303), (296, 303), (296, 300), (300, 302), (12, 12), (303, 303)):
        assert np.array_equal(bits(frontier.raw_rows(start, stop)), bits(raw[start:stop])), (start, stop)
    with pytest.raises(ValueError, match="outside"):
        frontier.raw_rows(0, 304)
    rng_ids = np.random.default_rng(8)
    ids = selection(rng_ids, 303, 1, width=24, count=4)
    got = sa.gather_latent(frontier.latent_source(), ids)[0]
    assert np.array_equal(bits(got), bits(sa.gather_latent(latent[None, None], ids)[0]))
    assert len(frontier.state) == 3 and frontier.state[2] is frontier.pools.pooled
    with pytest.raises(ValueError, match="one row"):
        frontier.append(mx.zeros((1, 1, 2, 512)), mx.zeros((1, 2, 256)))
    with pytest.raises(ValueError, match="integer base frontier"):
        sa.Glm53MLAFrontier(base, mx.array(300))
    with pytest.raises(ValueError, match="differs"):
        sa.Glm53MLAFrontier(base, 299)


def test_frontier_forks_are_private_siblings():
    rng = np.random.default_rng(9)
    base = committed_base(rng, 64)
    parent = sa.Glm53MLAFrontier(base, 64)
    (l0, r0), (l1, r1), (l2, r2) = tail_rows(rng, 3)
    parent.append(l0, r0)
    left, right = parent.fork(), parent.fork()
    assert left.pools is not right.pools and left.pools is not parent.pools
    left.append(l1, r1)
    right.append(l2, r2)
    assert parent.depth == 1 and left.depth == right.depth == 2
    assert np.array_equal(bits(left.tail_latent[1]), bits(l1.reshape(512)))
    assert np.array_equal(bits(right.tail_latent[1]), bits(l2.reshape(512)))
    assert left.base is right.base is base and base.offset == 64


@pytest.mark.parametrize("tokens,depth", [(300, 3), (510, 4), (512, 1), (64, 0)])
def test_materialize_equals_the_clone_and_append_chain(tokens, depth):
    from mlx2.adapters.glm53_ast_target import clone

    rng = np.random.default_rng(10 + depth)
    base = committed_base(rng, tokens)
    rows = tail_rows(rng, depth)
    frontier = sa.Glm53MLAFrontier(base, tokens)
    # Today's layered tree: each row clones its parent's advanced cache and appends.
    chain = base
    for latent_row, raw_row in rows:
        chain = clone([chain])[0]
        chain.caches[0].update_and_fetch(latent_row, mx.zeros((1, 1, 1, 0)))
        chain.caches[1].update_and_fetch(raw_row[:, None], mx.zeros((1, 1, 1, 0)))
        frontier.append(latent_row, raw_row)
    before = [bits(c.keys) for c in base.caches]
    result = frontier.materialize()
    assert type(result) is Glm53MLACache and result is not base
    assert result.offset == chain.offset == tokens + depth
    assert result.meta_state == chain.meta_state
    for got, want in zip(result.caches, chain.caches, strict=True):
        for a, b in zip(got.state, want.state, strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.array_equal(np.array(a).view(np.uint8), np.array(b).view(np.uint8))
    assert result.pools is not frontier.pools and result.pools.complete == frontier.pools.complete
    assert all(np.array_equal(bits(c.keys), b) for c, b in zip(base.caches, before, strict=True))
    assert base.offset == tokens


def test_frontier_attaches_a_pool_cache_to_a_base_without_one():
    indexer = pytest.importorskip("mlx2.runtime.models.glm53_gguf_indexer")
    if not hasattr(indexer, "Glm53PoolCache"):
        pytest.skip("Glm53PoolCache (P1) not landed")
    rng = np.random.default_rng(11)
    base = committed_base(rng, 32)
    base.pools = None
    frontier = sa.Glm53MLAFrontier(base, 32)
    assert isinstance(base.pools, indexer.Glm53PoolCache)
    assert isinstance(frontier.pools, indexer.Glm53PoolCache) and frontier.pools is not base.pools


def test_frontier_siblings_complete_a_boundary_pool_privately():
    indexer = pytest.importorskip("mlx2.runtime.models.glm53_gguf_indexer")
    if not hasattr(indexer, "Glm53PoolCache") or not hasattr(indexer, "pool_fixed_order"):
        pytest.skip("Glm53PoolCache (P1) not landed")
    rng = np.random.default_rng(13)
    ape = mx.array(rng.normal(size=(4, 128)).astype(np.float32))
    base = committed_base(rng, 10)  # pool 2 = base rows 8, 9 + tail rows 10, 11
    base.pools = indexer.Glm53PoolCache()
    base.pools.extend(lambda a, b: base.caches[1].keys[0, 0, a:b], 10, ape)
    parent = sa.Glm53MLAFrontier(base, 10)
    (l0, r0), (l1, r1), (l2, r2) = tail_rows(rng, 3)
    parent.append(l0, r0)
    parent.pools.extend(parent.raw_rows, parent.offset, ape)
    left, right = parent.fork(), parent.fork()
    left.append(l1, r1)
    right.append(l2, r2)
    for child in (left, right):
        got = child.pools.extend(child.raw_rows, child.offset, ape)
        raw = mx.concatenate([base.caches[1].keys[0, 0, :10], child.tail_raw])
        assert np.array_equal(bits(got), bits(indexer.pool_fixed_order(raw[:12], ape)[None]))
        assert np.array_equal(bits(child.materialize().pools.pooled[:, :3]), bits(got))
    assert not np.array_equal(bits(left.pools.pooled[:, 2]), bits(right.pools.pooled[:, 2]))
    assert parent.pools.complete == base.pools.complete == 2
