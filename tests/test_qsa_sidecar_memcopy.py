"""QSA sidecar memory-copy fixes: each one bit-exact against the code it
replaced, plus a check that the copy it removes is really gone."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models import qsa_sidecar as qs  # noqa: E402

HID, H, D, ROPE, BASE = 64, 4, 128, 64, 1e7


def _weights(seed):
    rng = np.random.default_rng(seed)
    return {
        "q_proj.weight": mx.array(rng.standard_normal((H * D, HID)).astype(np.float32) * 0.1),
        "k_proj.weight": mx.array(rng.standard_normal((D, HID)).astype(np.float32) * 0.1),
        "q_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
        "k_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
        "k_block_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
    }


def _indexer(seed=0):
    return qs.LayerIndexer(_weights(seed), heads=H, head_dim=D, rope_dim=ROPE,
                           rope_base=BASE, eps=1e-6)


class _ConcatIndexer:
    """The pre-fix indexer (85f0c607b), verbatim: ``1 + w`` per call and the
    pooled history re-concatenated on every append."""

    def __init__(self, w):
        self.wq, self.wk = w["q_proj.weight"], w["k_proj.weight"]
        self.q_norm, self.k_norm, self.k_block_norm = w["q_norm"], w["k_norm"], w["k_block_norm"]
        self.tokens, self.blocks, self.tail, self.heads = 0, None, None, H

    def queries(self, x, offset):
        q = (x @ self.wq.T).reshape(x.shape[0], H, D)
        q = qs._zc_norm(q, self.q_norm, 1e-6).transpose(1, 0, 2)
        return qs._rope(q, dims=ROPE, base=BASE, offset=offset)

    def append_keys(self, x, offset):
        k = qs._zc_norm(x @ self.wk.T, self.k_norm, 1e-6)
        if self.tail is not None:
            k = mx.concatenate([self.tail, k], axis=0)
        first_block = self.tokens // 4
        complete = k.shape[0] // 4
        if complete:
            pooled = k[: complete * 4].reshape(complete, 4, -1).mean(axis=1)
            pooled = qs._zc_norm(pooled, self.k_block_norm, 1e-6)
            pooled = qs._rope(pooled[None], dims=ROPE, base=BASE,
                              offset=first_block, scale=4.0)[0]
            self.blocks = (pooled if self.blocks is None
                           else mx.concatenate([self.blocks, pooled], axis=0))
        rest = k[complete * 4:]
        self.tail = rest if rest.shape[0] else None
        self.tokens += x.shape[0]


def _chunks():
    # Prefill-shaped chunks (uneven, tail carried), then single-token decode.
    return [37, 5, 64, 2, 130] + [1] * 90


def _ptr(a):
    view = np.asarray(a)
    address = view.__array_interface__["data"][0]
    del view
    return address


@pytest.mark.parametrize("min_capacity", [1, 8, 256])
def test_pooled_keys_queries_and_scores_bit_exact(monkeypatch, min_capacity):
    monkeypatch.setattr(qs, "KEY_BLOCKS_MIN_CAPACITY", min_capacity)
    rng = np.random.default_rng(3)
    new, old = _indexer(), _ConcatIndexer(_weights(0))
    offset = 0
    for size in _chunks():
        x = mx.array(rng.standard_normal((size, HID)).astype(np.float32))
        new.append_keys(x, offset)
        old.append_keys(x, offset)
        q_new, q_old = new.queries(x, offset), old.queries(x, offset)
        assert mx.array_equal(q_new, q_old).item()
        if old.blocks is None:
            assert new.blocks is None and new.n_blocks == 0
        else:
            assert new.n_blocks == old.blocks.shape[0]
            assert mx.array_equal(new.blocks, old.blocks).item()
            want = qs.LayerIndexer.scores(old, q_old)
            assert mx.array_equal(new.scores(q_new), want).item()
        assert (new.tail is None) == (old.tail is None)
        if new.tail is not None:
            assert mx.array_equal(new.tail, old.tail).item()
        offset += size


def test_pooled_keys_append_in_place_with_geometric_growth(monkeypatch):
    """Within capacity an append is a donated slice update (same storage);
    the buffer is reallocated only O(log n) times."""
    monkeypatch.setattr(qs, "KEY_BLOCKS_MIN_CAPACITY", 16)
    rng = np.random.default_rng(5)
    idx = _indexer()
    offset, grows, in_place, capacities = 0, 0, 0, set()
    for size in [9] + [1] * 400:
        before = None if idx._kbuf is None else (_ptr(idx._kbuf), idx._kbuf.shape[0])
        x = mx.array(rng.standard_normal((size, HID)).astype(np.float32))
        idx.append_keys(x, offset)
        offset += size
        if idx.n_blocks:
            # A decode step's use of the pooled keys, evaluated before the next.
            s = idx.scores(idx.queries(x, offset - size))
            mx.eval(s)
            del s
        if before is None or idx._kbuf.shape[0] != before[1]:
            grows += 1
        elif idx.n_blocks and (offset % 4 == 0):
            assert _ptr(idx._kbuf) == before[0], "append re-copied the pooled keys"
            in_place += 1
        capacities.add(idx._kbuf.shape[0] if idx._kbuf is not None else 0)
    assert idx.n_blocks == offset // 4 == 102
    assert grows <= 4  # 16 -> 32 -> 64 -> 128 (102 blocks)
    assert in_place >= 90
    assert idx._kbuf.shape[0] == 128


def test_pooled_keys_concatenate_traffic_is_amortized(monkeypatch):
    """Rows moved by ``mx.concatenate`` inside ``append_keys`` stay linear in
    the number of blocks (the old code moved ~n^2/2 rows over n blocks)."""
    moved = []
    real = mx.concatenate

    def counting(arrays, axis=0, **kw):
        moved.append(sum(a.shape[0] for a in arrays if a.ndim == 2 and a.shape[-1] == D))
        return real(arrays, axis=axis, **kw)

    rng = np.random.default_rng(9)
    idx = _indexer()
    x = rng.standard_normal((2048, HID)).astype(np.float32)
    idx.append_keys(mx.array(x[:1024]), 0)
    monkeypatch.setattr(mx, "concatenate", counting)
    for t in range(1024, 2048):
        idx.append_keys(mx.array(x[t:t + 1]), t)
        if t % 64 == 0:
            mx.eval(idx._kbuf)
    monkeypatch.setattr(mx, "concatenate", real)
    blocks = idx.n_blocks
    assert blocks == 512
    # Tail carries (<= 4 rows per step) plus one doubling of 512 rows.
    assert sum(moved) <= 4 * 1024 + 2 * blocks


# --- the sidecar attention layer -------------------------------------------

NQH, NKV, HD = 6, 2, 32


def _sidecar_attention(tmp_path, *, budget, mode):
    import types

    import mlx.nn as nn  # noqa: F401  (Module.set_dtype)

    from mlx2.runtime.models.qwen38_27b import Qwen3NextAttention
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    mx.random.seed(11)
    args = TextModelArgs(
        hidden_size=HID, num_hidden_layers=1, intermediate_size=128, vocab_size=32,
        num_attention_heads=NQH, num_key_value_heads=NKV, head_dim=HD,
        rope_theta=1e7, partial_rotary_factor=0.25,
    )
    attention = Qwen3NextAttention(args)
    attention.set_dtype(mx.bfloat16)
    prefix = "model.layers.0.self_attn._qsa_indexer."
    mx.save_safetensors(str(tmp_path / "qsa-indexers.safetensors"),
                        {prefix + k: v for k, v in _weights(21).items()})
    sidecar = qs.QSASidecar(tmp_path, budget=budget, mode=mode)
    model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attention)])
    assert qs.install(model, sidecar) == [0]
    return attention, sidecar


def _run(attention, sidecar, xs, *, reserve=None):
    from mlx2.runtime.models.cache import KVCache

    sidecar.reset()
    cache = KVCache()
    if reserve is not None:
        cache.reserve(reserve)
    outs = []
    for x in xs:
        mask = "causal" if x.shape[1] > 1 else None
        out = attention(x, mask=mask, cache=cache)
        mx.eval(out)
        outs.append(out)
    return outs


def _inputs(prefill, decode, seed=17):
    rng = np.random.default_rng(seed)
    xs = [mx.array(rng.standard_normal((1, n, HID)).astype(np.float32)).astype(mx.bfloat16)
          for n in prefill]
    xs += [mx.array(rng.standard_normal((1, 1, HID)).astype(np.float32)).astype(mx.bfloat16)
           for _ in range(decode)]
    return xs


def test_sidecar_gating_bit_exact_with_base_attention(tmp_path):
    """``selfdense`` runs the sidecar's own projections, cache and gating on
    plain causal attention: it must reproduce the base layer bit for bit."""
    attention, sidecar = _sidecar_attention(tmp_path, budget=16, mode="dense")
    xs = _inputs([24, 9], 12)
    want = _run(attention, sidecar, xs)
    sidecar.mode = "selfdense"
    got = _run(attention, sidecar, xs)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype == mx.bfloat16
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("mode", ["mask", "gather"])  # window/nax need Metal
def test_sparse_modes_bit_exact_with_reserved_kv_cache(tmp_path, mode):
    """A reserved KV cache (one allocation, strided views) gives the same
    bits as the default 256-step growth in every CPU-servable sparse mode."""
    attention, sidecar = _sidecar_attention(tmp_path, budget=16, mode=mode)
    # gather serves L == 1 only (L > 1 would route to Metal NAX): keep its
    # prefill dense by construction (<= 4 complete blocks).
    prefill = [16] if mode == "gather" else [24, 40]
    xs = _inputs(prefill, 40)
    qs.status(reset=True)
    want = _run(attention, sidecar, xs)
    rows = qs.status(reset=True)
    assert any(k.startswith("sparse_") for k in rows), rows
    total = sum(x.shape[1] for x in xs)
    got = _run(attention, sidecar, xs, reserve=total)
    for a, b in zip(got, want):
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
@pytest.mark.parametrize("length,valid_rate", [(1, 0.9), (33, 0.5), (8, 0.0)])
def test_compaction_scalar_zero_bit_exact(dtype, length, valid_rate):
    from mlx2.runtime.models.qwen4_exp import _compact_qsa_block_ids

    rng = np.random.default_rng(length)
    n_blocks, width = 300, 64
    ids = np.stack([rng.permutation(n_blocks)[:width] for _ in range(length)])[None]
    valid = rng.random((1, length, width)) < valid_rate
    sel, ok = mx.array(ids).astype(dtype), mx.array(valid)
    (got, counts) = _compact_qsa_block_ids(sel, ok, n_blocks=n_blocks)
    # The pre-fix body, with its materialized zeros.
    keys = mx.where(ok, sel.astype(mx.int32), mx.array(n_blocks, dtype=mx.int32))
    want = mx.take_along_axis(sel, mx.argsort(keys, axis=-1), axis=-1)
    want_counts = mx.sum(ok.astype(mx.int32), axis=-1)
    want = mx.where(mx.arange(width) < want_counts[..., None], want, mx.zeros_like(want))
    assert got.dtype == want.dtype == dtype
    assert mx.array_equal(got, want).item() and mx.array_equal(counts, want_counts).item()
