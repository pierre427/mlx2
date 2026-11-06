"""QSA sidecar: indexer math against a numpy transcription of the reference,
incremental key pooling, and the three sparse attention paths against one
masked-softmax oracle."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models import qsa_sidecar as qs  # noqa: E402

HID, H, D, ROPE, BASE = 64, 4, 128, 64, 1e7


def _weights(rng):
    return {
        "q_proj.weight": mx.array(rng.standard_normal((H * D, HID)).astype(np.float32) * 0.1),
        "k_proj.weight": mx.array(rng.standard_normal((D, HID)).astype(np.float32) * 0.1),
        "q_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
        "k_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
        "k_block_norm": mx.array(rng.standard_normal(D).astype(np.float32) * 0.1),
    }


def _np_rope(x, pos):
    # HF rotate_half convention on the first ROPE channels.
    inv = 1.0 / (BASE ** (np.arange(0, ROPE, 2) / ROPE))
    ang = pos[:, None] * inv[None]
    cos = np.concatenate([np.cos(ang)] * 2, -1)
    sin = np.concatenate([np.sin(ang)] * 2, -1)
    r, p = x[..., :ROPE], x[..., ROPE:]
    half = ROPE // 2
    rot = np.concatenate([-r[..., half:], r[..., :half]], -1)
    return np.concatenate([r * cos + rot * sin, p], -1)


def _np_norm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * (1 + w)


def _np_scores(w, x):
    """Transcription of the reference ``QSAIndexer.forward`` (one sequence)."""
    w = {k: np.array(v) for k, v in w.items()}
    t = x.shape[0]
    q = _np_norm((x @ w["q_proj.weight"].T).reshape(t, H, D), w["q_norm"])
    k = _np_norm(x @ w["k_proj.weight"].T, w["k_norm"])
    pos = np.arange(t, dtype=np.float64)
    q = np.stack([_np_rope(q[:, h], pos) for h in range(H)], 1)
    nb = t // 4
    kb = _np_norm(k[: nb * 4].reshape(nb, 4, D).mean(1), w["k_block_norm"])
    kb = _np_rope(kb, np.arange(nb, dtype=np.float64) * 4)
    s = np.maximum(np.einsum("thd,nd->htn", q, kb), 0).sum(0)
    valid = (np.arange(nb)[None] * 4 + 3) <= np.arange(t)[:, None]
    return np.where(valid, s, -np.inf), valid


def _indexer(rng):
    return qs.LayerIndexer(_weights(rng), heads=H, head_dim=D, rope_dim=ROPE,
                           rope_base=BASE, eps=1e-6)


def test_incremental_scores_match_reference_transcription():
    rng = np.random.default_rng(0)
    idx = _indexer(rng)
    x = rng.standard_normal((61, HID)).astype(np.float32)
    want, valid = _np_scores({k: v for k, v in _weights(np.random.default_rng(0)).items()}, x)
    # Feed the keys in uneven chunks so the tail carry is exercised.
    got_rows = []
    offset = 0
    for size in (5, 1, 13, 2, 40):
        part = mx.array(x[offset:offset + size])
        idx.append_keys(part, offset)
        q = idx.queries(part, offset)
        s = np.array(idx.scores(q))
        pad = np.full((size, want.shape[1]), np.nan, np.float32)
        pad[:, : s.shape[1]] = s
        got_rows.append(pad)
        offset += size
    got = np.concatenate(got_rows)
    got = np.where(valid, got, -np.inf)
    np.testing.assert_allclose(got[valid], want[valid], rtol=2e-4, atol=2e-4)


def test_append_refuses_a_gap():
    idx = _indexer(np.random.default_rng(1))
    idx.append_keys(mx.zeros((3, HID)), 0)
    with pytest.raises(qs.QSASidecarRefused):
        idx.append_keys(mx.zeros((1, HID)), 5)


def _oracle(q, k, v, vis, scale):
    s = np.einsum("hld,htd->hlt", q, k) * scale
    s = np.where(vis[None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hlt,htd->hld", p, v)


def _selection(rng, length, offset, topk):
    total = offset + length
    nb = total // 4
    q_pos = mx.arange(offset, total, dtype=mx.int32)
    scores = mx.array(rng.standard_normal((length, nb)).astype(np.float32))
    starts = mx.arange(nb, dtype=mx.int32) * 4
    valid = (starts[None] + 3) <= q_pos[:, None]
    scores = mx.where(valid, scores, -mx.inf)
    ids = mx.argpartition(-scores, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    return ids, mx.take_along_axis(valid, ids, axis=-1), q_pos, nb, total


@pytest.mark.parametrize("length,offset", [(1, 4099), (1, 4095), (7, 4100), (1, 2050)])
def test_mask_path_matches_oracle(length, offset):
    rng = np.random.default_rng(length + offset)
    ids, pv, q_pos, nb, total = _selection(rng, length, offset, 256)
    vis = np.array(qs._dense_mask(ids, pv, q_pos, nb, total))[0, 0]
    # Reference visibility, built independently.
    want = np.zeros((length, total), bool)
    for row in range(length):
        p = int(q_pos[row])
        for b, ok in zip(np.array(ids[row]), np.array(pv[row])):
            if ok:
                want[row, b * 4: b * 4 + 4] = True
        want[row, (p // 4) * 4: p + 1] = True
        want[row, p + 1:] = False
    np.testing.assert_array_equal(vis, want)


@pytest.mark.parametrize("offset", [4099, 4095, 4096, 2050])
def test_gather_matches_mask(offset):
    rng = np.random.default_rng(offset)
    ids, pv, q_pos, nb, total = _selection(rng, 1, offset, 256)
    q = mx.array(rng.standard_normal((1, 6, 1, 256)).astype(np.float32))
    k = mx.array(rng.standard_normal((1, 1, total, 256)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, 1, total, 256)).astype(np.float32))
    vis = np.array(qs._dense_mask(ids, pv, q_pos, nb, total))[0, 0]
    want = _oracle(np.array(q[0]), np.repeat(np.array(k[0]), 6, 0),
                   np.repeat(np.array(v[0]), 6, 0), vis, 0.0625)
    got = np.array(qs._gather(q, k, v, ids, pv, q_pos, 0.0625))[0]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mx.metal.is_available(), reason="NAX needs Metal")
@pytest.mark.parametrize("length,offset", [(1, 4099), (1, 4095), (16, 4100)])
def test_nax_matches_mask(length, offset):
    from mlx2.runtime.models.qwen4_qsa_nax import nax_kernel_available

    if not nax_kernel_available():
        pytest.skip("NAX kernel unavailable on this device")
    rng = np.random.default_rng(7 + length)
    ids, pv, q_pos, nb, total = _selection(rng, length, offset, 512)
    q = mx.array(rng.standard_normal((1, 24, length, 256)).astype(np.float16))
    k = mx.array(rng.standard_normal((1, 4, total, 256)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, 4, total, 256)).astype(np.float16))
    vis = qs._dense_mask(ids, pv, q_pos, nb, total)
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=vis)
    got = qs._nax(q, k, v, qs._compact(ids, pv, q_pos, total), 0.0625)
    np.testing.assert_allclose(np.array(got, np.float32), np.array(want, np.float32),
                               rtol=2e-2, atol=2e-2)


def _np_quest(q, k, offset, window, sink, budget, page):
    """Brute force: per-head per-channel max(q*min, q*max), summed over heads."""
    (h, length, d) = q.shape
    (hk, total, _) = k.shape
    group = h // hk
    n_pages = total // page
    vis = np.zeros((length, total), bool)
    for row in range(length):
        p = offset + row
        cand = []
        for pg in range(n_pages):
            s0, e0 = pg * page, pg * page + page - 1
            if e0 <= p and e0 >= sink and s0 <= p - window:
                score = 0.0
                for head in range(h):
                    kk = k[head // group, s0:e0 + 1]
                    lo, hi = kk.min(0), kk.max(0)
                    score += np.maximum(q[head, row] * lo, q[head, row] * hi).sum()
                cand.append((score, pg))
        cand.sort(reverse=True)
        for _, pg in cand[: min(n_pages, budget // page)]:
            vis[row, pg * page:(pg + 1) * page] = True
        t = np.arange(total)
        vis[row] |= (t < sink) | (t > p - window)
        vis[row] &= t <= p
    return vis


@pytest.mark.parametrize("length,offset", [(1, 900), (5, 700), (3, 129)])
def test_quest_mask_matches_brute_force(length, offset):
    rng = np.random.default_rng(length * 31 + offset)
    total = offset + length
    q = rng.standard_normal((6, length, 16)).astype(np.float32)
    k = rng.standard_normal((2, total, 16)).astype(np.float32)
    got = np.array(qs._quest_mask(mx.array(q[None]), mx.array(k[None]), offset,
                                  window=64, sink=32, budget=128, page=16))[0, 0]
    want = _np_quest(q, k, offset, 64, 32, 128, 16)
    np.testing.assert_array_equal(got, want)
