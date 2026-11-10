"""Fused QSA selection kernel against an exact numpy model of its rule and
against the unfused argpartition selection (equal above the threshold)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="fused QSA selection is a Metal kernel"
)


def _skip_unsupported(tg):
    from mlx2.runtime.models.qsa_select import _device_threadgroup

    if tg is not None and tg > _device_threadgroup():
        pytest.skip(f"threadgroup {tg} is above this device's selected width")


def _model(dots, q_pos, k):
    """Same rule in numpy: relu head-sum, valid prefix, top-k, ties -> low ids."""
    h, rows, nb = dots.shape
    relu = np.maximum(dots, np.float32(0))
    s = relu[0].copy()
    for head in range(1, h):
        s = (s + relu[head]).astype(np.float32)
    ids = np.zeros((rows, k), np.int64)
    counts = np.zeros(rows, np.int64)
    for r in range(rows):
        nv = int(np.clip((int(q_pos[r]) + 1) // 4, 0, nb))
        if nv <= k:
            ids[r, :nv] = np.arange(nv)
            counts[r] = nv
            continue
        row = s[r, :nv]
        order = np.lexsort((np.arange(nv), -row.astype(np.float64)))
        chosen = np.sort(order[:k])
        ids[r] = chosen
        counts[r] = k
    return ids, counts, s


def _case(seed, heads, rows, nb, offset, ties):
    rng = np.random.default_rng(seed)
    dots = rng.standard_normal((heads, rows, nb)).astype(np.float32)
    if ties:  # coarse values: many exact ties, many zeros after relu
        dots = (np.round(dots * 2) / 2).astype(np.float32)
    q_pos = np.arange(offset, offset + rows, dtype=np.int32)
    return dots, q_pos


def _run(dots, q_pos, k, q_chunk=512, tg=1024):
    from mlx2.runtime.models import qsa_select

    # Feed dots through the kernel directly: identity GEMM via D = nb.
    kernel = qsa_select._kernel()
    (h, rows, nb) = dots.shape
    (ids, counts, _) = kernel(
        inputs=[mx.array(dots), mx.array(q_pos), mx.array([rows, nb, k], dtype=mx.uint32)],
        template=[("H", h), ("TG", tg)],
        grid=(tg * rows, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(rows, k), (rows,), (rows, nb)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    return np.array(ids).astype(np.int64), np.array(counts).astype(np.int64)


@pytest.mark.parametrize(
    "seed,heads,rows,nb,offset,k,ties",
    [
        (0, 4, 1, 65536, 262140, 512, False),   # decode at 256K
        (1, 4, 1, 65536, 262143, 512, True),
        (2, 4, 37, 8192, 32700, 512, True),     # prefill rows, many ties
        (3, 4, 64, 2600, 10000, 512, False),    # rows straddle nv <= k
        (4, 4, 16, 600, 2040, 512, True),       # shortcut and select rows mixed
        (5, 4, 8, 3000, 4100, 512, True),
        (6, 1, 5, 5000, 19990, 7, True),        # tiny k
    ],
)
@pytest.mark.parametrize("tg", [768, 1024])  # device-dependent width (2026-11-06)
def test_kernel_matches_exact_model(seed, heads, rows, nb, offset, k, ties, tg):
    _skip_unsupported(tg)
    dots, q_pos = _case(seed, heads, rows, nb, offset, ties)
    want_ids, want_counts, _ = _model(dots, q_pos, k)
    ids, counts = _run(dots, q_pos, k, tg=tg)
    np.testing.assert_array_equal(counts, want_counts)
    for r in range(rows):
        c = int(counts[r])
        np.testing.assert_array_equal(ids[r, :c], want_ids[r, :c])
        assert (ids[r, c:] == 0).all()


@pytest.mark.parametrize("tg", [None, 768, 1024])
def test_matches_argpartition_above_threshold(tg):
    """Against the unfused rule: same sets except among exact threshold ties."""
    from mlx2.runtime.models.qsa_select import fused_select

    _skip_unsupported(tg)
    rng = np.random.default_rng(11)
    (heads, rows, nb, d, k) = (4, 40, 9000, 128, 512)
    q = mx.array(rng.standard_normal((heads, rows, d)).astype(np.float32))
    kb = mx.array(rng.standard_normal((nb, d)).astype(np.float32))
    q_pos = mx.arange(nb * 4 - rows, nb * 4, dtype=mx.int32)
    ids, counts = fused_select(q, kb, q_pos, k, q_chunk=16, threadgroup=tg)
    scores = sum(mx.maximum(q[h] @ kb.T, 0) for h in range(heads))
    # Blocks are selectable only once complete: 4j + 3 <= q_pos.
    valid = (mx.arange(nb)[None, :] * 4 + 3) <= q_pos[:, None]
    scores = mx.where(valid, scores, -mx.inf)
    ref = np.array(mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k])
    s = np.array(scores)
    ids = np.array(ids)
    for r in range(rows):
        got = set(ids[r, : int(counts[r])].tolist())
        want = set(ref[r].tolist())
        thresh = np.sort(s[r])[-k]
        assert {j for j in got if s[r, j] > thresh} == {j for j in want if s[r, j] > thresh}
        assert len(got) == k and min(s[r, j] for j in got) >= thresh
        assert list(ids[r]) == sorted(ids[r])
