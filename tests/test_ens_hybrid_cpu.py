"""H40 hybrid pieces on CPU: GDN reference, causality, RoPE, sparse mask, bus targets, bypass, long-context family."""
from __future__ import annotations

import os
import random

os.environ.setdefault("ENS_DEVICE", "cpu")

import mlx.core as mx
import numpy as np
import pytest

from research.ens import common as C  # noqa: F401  (sets the CPU device)
from research.ens import longctx as LC
from research.ens import vocab as V
from research.ens.hybrid import AttnBlock, GDNBlock, HybridConfig, HybridCore, rope
from research.ens.model import BusCall

SMALL = HybridConfig(d=32, heads=2, ffn=64, seq_len=64, pattern="GAGS", site=1, sparse_block=8, sparse_topk=2)


def _tokens(B=3, T=40, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, V.VOCAB_SIZE, (B, T)))


def test_gdn_scan_matches_numpy_reference():
    mx.random.seed(0)
    blk = GDNBlock(SMALL)
    h = mx.random.normal((2, 7, SMALL.d))
    o, S, _ = blk.scan(h)
    q, k, v, al, be, _ = (np.array(t) for t in blk.prepare(h))
    H, hd = SMALL.heads, SMALL.d // SMALL.heads
    St = np.zeros((2, H, hd, hd))
    ref = []
    for t in range(7):
        for b in range(2):
            for hh in range(H):
                kt, vt = k[b, t, hh], v[b, t, hh]
                St[b, hh] = al[b, t, hh] * (St[b, hh] @ (np.eye(hd) - be[b, t, hh] * np.outer(kt, kt))) \
                    + be[b, t, hh] * np.outer(vt, kt)
        ref.append(np.einsum("bhij,bhj->bhi", St, q[:, t]))
    assert np.allclose(np.array(o), np.stack(ref, 1), atol=1e-4)
    assert np.allclose(np.array(S), St, atol=1e-4)
    # published-form gates: decay starts near 0.95, write strength near 0.5
    assert abs(float(al.mean()) - 0.95) < 0.02 and abs(float(be.mean()) - 0.5) < 0.1


def test_gdn_conv_is_causal_and_binds_adjacent_tokens():
    mx.random.seed(9)
    blk = GDNBlock(SMALL)
    h = mx.random.normal((1, 12, SMALL.d))
    h2 = mx.array(np.array(h)); h2[:, 8:] = h2[:, 8:] * -1.0
    q1, k1, v1, *_ = blk.prepare(h)
    q2, k2, v2, *_ = blk.prepare(h2)
    assert mx.allclose(k1[:, :8], k2[:, :8], atol=1e-6).item()
    # the conv mixes the previous token into the current key: changing token 5 alone changes k at 6
    h3 = mx.array(np.array(h)); h3[:, 5] = h3[:, 5] + 3.0
    _, k3, *_ = blk.prepare(h3)
    assert not mx.allclose(k1[:, 6], k3[:, 6], atol=1e-4).item()


@pytest.mark.parametrize("pattern", ["G", "A", "S"])
def test_blocks_are_causal(pattern):
    mx.random.seed(1)
    core = HybridCore(HybridConfig(**{**SMALL.to_json(), "pattern": pattern}))
    t1 = _tokens()
    t2 = mx.array(np.array(t1))
    t2[:, 30:] = (t2[:, 30:] + 7) % V.VOCAB_SIZE
    a, b = core(t1), core(t2)
    assert mx.allclose(a[:, :30], b[:, :30], atol=1e-5).item()
    assert not mx.allclose(a[:, 30:], b[:, 30:], atol=1e-5).item()


def test_rope_dot_depends_on_relative_position_only():
    mx.random.seed(2)
    q = mx.random.normal((1, 1, 1, 16)); k = mx.random.normal((1, 1, 1, 16))
    d1 = (rope(q, offset=5) * rope(k, offset=2)).sum().item()
    d2 = (rope(q, offset=105) * rope(k, offset=102)).sum().item()
    assert abs(d1 - d2) < 1e-3


def test_sparse_mask_causal_own_block_and_budget():
    cfg = HybridConfig(**{**SMALL.to_json(), "pattern": "S", "sparse_block": 8, "sparse_topk": 2})
    blk = AttnBlock(cfg, sparse=True)
    mx.random.seed(3)
    q, k = mx.random.normal((1, 2, 40, 16)), mx.random.normal((1, 2, 40, 16))
    m = np.array(blk.sparse_mask(q, k))
    T = 40
    assert not m[..., np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]].any()       # causal
    for t in range(T):
        assert m[0, :, t, (t // 8) * 8: t + 1].all()                                      # own block, causally
    for hh in range(2):
        for t in range(T):
            assert len({s // 8 for s in range(T) if m[0, hh, t, s]}) <= 3                  # per token: top-2 + own
    # the chosen past blocks are the highest-scoring ones by query . block-mean key
    kb = np.array(k).reshape(1, 2, 5, 8, 16).mean(3)
    sc = np.einsum("bhtd,bhjd->bhtj", np.array(q), kb)
    for hh in range(2):
        for t in range(16, T):
            past = list(range(t // 8))
            top = set(sorted(past, key=lambda j: -sc[0, hh, t, j])[:2])
            seen = {s // 8 for s in range(T) if m[0, hh, t, s]} - {t // 8}
            assert seen == top
    full = HybridConfig(**{**cfg.to_json(), "sparse_topk": 99})
    mf = np.array(AttnBlock(full, sparse=True).sparse_mask(q, k))
    assert (mf == np.tril(np.ones((T, T), bool))[None, None]).all()                      # all past = causal


def _bus(B, T, w, field, scale=1.0):
    mx.random.seed(4)
    return BusCall("x", mx.zeros((B,), dtype=mx.int32), mx.array(field), mx.random.normal((B, w)) * scale)


def test_bus_targets_bypass_zero_message_and_locality():
    mx.random.seed(5)
    core = HybridCore(SMALL)
    t = _tokens(B=2, T=40)
    base = core(t)
    field = [20, 25]
    for target, kw in (("residual", {}), ("gdn_state", {"gdn_layer": 2})):
        zero = BusCall("x", mx.zeros((2,), dtype=mx.int32), mx.array(field), mx.zeros((2, SMALL.bus_width)))
        assert mx.allclose(core(t, zero, target=target, **kw), base, atol=1e-6).item()   # zero writes nothing
        out = core(t, _bus(2, 40, SMALL.bus_width, field), target=target, **kw)
        for b, f in enumerate(field):
            assert mx.allclose(out[b, :f], base[b, :f], atol=1e-5).item()                # nothing before the field
            assert not mx.allclose(out[b, f], base[b, f], atol=1e-5).item()              # seen at the field itself
    with pytest.raises(ValueError):
        core(t, _bus(2, 40, SMALL.bus_width, field), target="gdn_state", gdn_layer=1)    # block 1 is attention


def test_longctx_rows_answers_and_layout():
    for k in (1, 2, 3):
        for ids, fi, ai, ans, chain in LC.make_rows(20, L=256, k=k, n_pairs=40, seed=k):
            assert len(ids) == 256 and ids[fi] == V.NEUTRAL and ids[ai] == V.ANS
            assert LC.answer_from_tokens(ids, k) == ans and len(chain) == k + 1
            srcs = [ids[i + 1] for i in range(len(ids) - 2) if ids[i] == V.tok("GR")]
            assert len(srcs) == len(set(srcs)) == 40                                      # a function
    a = LC.make_rows(5, L=256, k=2, seed=0)
    b = LC.make_rows(5, L=256, k=2, seed=1)
    assert all(x[0] != y[0] for x, y in zip(a, b))


def test_bus_validation_and_state_bound():
    mx.random.seed(6)
    core = HybridCore(SMALL)
    t = _tokens(B=2, T=40)
    bus = _bus(2, 40, SMALL.bus_width, [20, 25])
    for kw in ({"target": "gdn-state"}, {"target": "gdn_state"}, {"target": "gdn_state", "gdn_layer": 99}):
        with pytest.raises(ValueError):
            core(t, bus, **kw)
    bad = HybridCore(HybridConfig(**{**SMALL.to_json(), "site": 9}))
    with pytest.raises(ValueError):
        bad(t, bus)
    with pytest.raises(ValueError):
        core(t, BusCall("x", bus.query_idx, bus.field_idx, lambda tap: tap))
    core.gate = mx.array([10.0])
    big = BusCall("x", bus.query_idx, bus.field_idx, bus.message * 100)
    _, (wk, wv) = core._state_write(big)
    assert float(mx.max(mx.sqrt(mx.sum(wv * wv, -1))).item()) <= SMALL.tau + 1e-5


@pytest.mark.parametrize("C", [4, 8, 16, 32])
def test_chunked_gdn_equals_sequential(C):
    mx.random.seed(11)
    seq = GDNBlock(SMALL)
    chunk = GDNBlock(HybridConfig(**{**SMALL.to_json(), "gdn_chunk": C}))
    chunk.update(seq.parameters())
    h = mx.random.normal((2, 37, SMALL.d))
    o1, S1, _ = seq.scan(h)
    o2, S2, _ = chunk.scan(h)
    assert mx.allclose(o1, o2, atol=1e-4).item() and mx.allclose(S1, S2, atol=1e-4).item()
    H, hd = SMALL.heads, SMALL.d // SMALL.heads
    wk = mx.random.normal((2, H, hd)); wk = wk / mx.linalg.norm(wk, axis=-1, keepdims=True)
    wv = mx.random.normal((2, H, hd)) * 0.3
    for pos in ([17, 17], [0, 0], [5, 22]):
        w = (mx.array(pos), (wk, wv))
        a, Sa, _ = seq.scan(h, w)
        b, Sb, _ = chunk.scan(h, w)
        assert mx.allclose(a, b, atol=1e-4).item() and mx.allclose(Sa, Sb, atol=1e-4).item()


def test_chunked_gdn_gradients_boundaries_and_extreme_decay():
    mx.random.seed(12)
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    seq = GDNBlock(SMALL)
    chunk = GDNBlock(HybridConfig(**{**SMALL.to_json(), "gdn_chunk": 8}))
    chunk.update(seq.parameters())
    h = mx.random.normal((2, 24, SMALL.d))
    H, hd = SMALL.heads, SMALL.d // SMALL.heads
    wk = mx.random.normal((2, H, hd)); wk = wk / mx.linalg.norm(wk, axis=-1, keepdims=True)
    wv = mx.random.normal((2, H, hd)) * 0.3
    for pos in (8, 16, 24, 27):                         # chunk boundaries, T itself, past T
        w = (mx.array([pos, pos]), (wk, wv))
        a, Sa, _ = seq.scan(h, w)
        b, Sb, _ = chunk.scan(h, w)
        assert mx.allclose(a, b, atol=1e-4).item() and mx.allclose(Sa, Sb, atol=1e-4).item(), pos
    ga = nn.value_and_grad(seq, lambda m: m(h).sum())(seq)[1]
    gb = nn.value_and_grad(chunk, lambda m: m(h).sum())(chunk)[1]
    fa, fb = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    assert all(mx.allclose(fa[k], fb[k], atol=1e-3).item() for k in fa)
    # alpha underflowing to exactly 0 must not NaN the chunked path
    big = mx.concatenate([mx.full((H,), 120.0), mx.zeros((H,))])
    chunk.ab.bias = big
    seq.ab.bias = big
    o1, _, _ = seq.scan(h)
    o2, _, _ = chunk.scan(h)
    assert bool(mx.all(mx.isfinite(o2)).item()) and mx.allclose(o1, o2, atol=1e-4).item()
