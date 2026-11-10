"""Flash-Next (qwen4_exp) MTP-head cache after accept-one / reject-the-rest.

Rapid-MLX #5646 (merged 2026-11-07): with Qwen4 native MTP at K=2, a rejected
draft trimmed the head's attention KV but left its compressed index cache
ahead of it. mlx2's head cache is ``QSAKVCache`` (B1) or ``BatchQSAKVCache``
(merged lanes), which roll ``index_keys`` and re-bound the pooled summaries
inside ``trim``/``trim_ragged``. This drives the real self-MTP propose/commit
path on Metal with a forced verifier that accepts exactly one draft per cycle
and rejects the rest (with two lanes the second lane alternates between
rejecting everything and accepting K-1, so the batched head cache stays
ragged), then checks on every MTP-head cache that

* the KV cursor, the raw indexer-key ledger and the per-row offsets agree,
* the pooled block summaries cover only closed blocks below the cursor and
  are bit-identical to a fresh pooling of the surviving ledger, and
* the head cache trails the target by its pending accepted pairs plus a
  per-lane constant.

Runs on the GPU: CPU-only runs miss MLX Metal bugs.
"""

import os
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import hybrid_speculative as hs
from mlx2.runtime.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.qwen4_exp import BatchQSAKVCache, QSAKVCache
from mlx2.runtime.sample_utils import LaneRNG

from test_batched_mtp import _tiny_qwen4_model

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (owned GPU window) for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


CYCLES = 6
PROMPTS = ([1, 2, 3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])


def _lane(model, uid, prompt, num_draft, share):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=1000,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(900 + uid),
        num_draft=num_draft,
        sampling_temp=0.8,
        sampling_top_p=1.0,
        sampling_top_k=8,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_index_state=share,
    )[0]


def _rows(cache):
    """(offset, ledger width, left pad) per row, from the live cache."""
    if isinstance(cache, BatchQSAKVCache):
        pads = cache.left_padding.tolist()
        return [
            (int(off), cache._idx - pad, pad)
            for off, pad in zip(cache.offset.tolist(), pads)
        ], cache._idx
    return [(int(cache.offset), int(cache.offset), 0)], int(cache.offset)


def _target_offsets(batch):
    for cache in batch.caches.target:
        if isinstance(cache, (QSAKVCache, BatchQSAKVCache)):
            off = cache.offset
            return [int(v) for v in (off.tolist() if isinstance(off, mx.array) else [off])]
    raise AssertionError("no QSA target cache")


def _check_head_cache(model, cache, where, frozen):
    """Ledger/KV/pooled agreement; returns how many pooled blocks it checked.

    ``frozen`` holds the pooled blocks that existed when the lanes attached
    (prompt prefill). Rollback must never rewrite them, so they are compared
    byte for byte with that snapshot. Every block closed during the draft
    cycles must equal a fresh pooling of the surviving raw-key ledger, byte for
    byte. (Prefill blocks are not re-pooled here: under pytest the conftest
    imports on the CPU device, and one prefill block can differ from a GPU
    re-pool by an ulp, independent of any rollback.)
    """
    assert isinstance(cache, (QSAKVCache, BatchQSAKVCache)), (where, type(cache))
    assert cache.index_keys is not None, where
    rows, cursor = _rows(cache)
    # KV cursor == raw-key ledger width (the #5646 invariant).
    assert cache.index_keys.shape[1] == cursor, (where, cache.index_keys.shape, cursor)
    for off, width, _pad in rows:
        assert off == width, (where, rows)
    assert cache.keys is not None and cache.keys.shape[2] >= cursor, where
    pooled = cache._qsa_pooled_keys
    if pooled is None:
        return 0
    ratio = cache._qsa_pooled_ratio
    assert ratio, where
    max_pad = max(pad for _o, _w, pad in rows)
    closed = (cursor - max_pad) // ratio
    assert pooled.shape[1] <= closed, (where, pooled.shape, closed)
    first = 0
    if frozen is not None and frozen.shape[0] == pooled.shape[0]:
        first = min(frozen.shape[1], pooled.shape[1])
        assert _bytes(pooled[:, :first]) == _bytes(frozen[:, :first]), (
            where, "prefill pooled block rewritten")
    if first == pooled.shape[1]:
        return 0
    indexer = model.mtp.layers[0].self_attn.indexer
    starts = mx.arange(first, pooled.shape[1]) * ratio
    if isinstance(cache, BatchQSAKVCache):
        fresh = indexer._pool_blocks_left_padded(
            cache.index_keys, starts, cache.left_padding.astype(cache.offset.dtype)
        )
    else:
        fresh = indexer._pool_blocks(
            cache.index_keys[:, first * ratio : pooled.shape[1] * ratio], starts
        )
    assert _bytes(pooled[:, first:]) == _bytes(fresh), (
        where, "pooled != repool(ledger)",
        mx.abs(pooled[:, first:].astype(mx.float32) - fresh.astype(mx.float32)).max().item())
    return int(pooled.shape[1] - first)


def _bytes(a):
    mx.eval(a)
    return np.array(a.astype(mx.float32)).tobytes()


@pytest.mark.parametrize("num_draft", [2, 3])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("n_lanes", [1, 2])
def test_mtp_head_cache_after_accept_one_reject_rest(
    monkeypatch, num_draft, share, n_lanes
):
    assert mx.default_device() == mx.gpu
    # Flash-Next serving turns the pooled-key cache on (adapters/flash_next.py).
    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    mx.random.seed(41)
    model = _tiny_qwen4_model()
    prompts = PROMPTS[:n_lanes]
    lanes = [
        _lane(model, uid, prompt, num_draft, share)
        for uid, prompt in enumerate(prompts)
    ]
    batch = attach_self_mtp_lanes(model, None, lanes)

    head_trims = []
    real_trim = hs._trim_self_mtp_cache_group

    def spy_trim(caches, counts, *, validate):
        if caches is batch.caches.draft:  # the MTP head, not the target
            head_trims.append([int(c) for c in counts])
        return real_trim(caches, counts, validate=validate)

    def plan(cycle):
        # Lane 0 accepts exactly one draft and rejects the rest. With two
        # lanes, lane 1 rejects everything on even cycles and accepts K-1 on
        # odd ones, so the rows stay ragged: uneven pending widths, right
        # padding at finalize, and per-row trims that differ.
        if n_lanes == 1:
            return (1,)
        return (1, 0 if cycle % 2 == 0 else num_draft - 1)

    pending = []

    def force(logprobs, _draft_lps, _drafts, _temperature, *, rng=None):
        accepted = pending.pop(0)
        return accepted, int(mx.argmax(logprobs[accepted]).item())

    gap = None
    pooled_seen = 0
    frozen = {}
    for i, c in enumerate(batch.caches.draft):
        p = getattr(c, "_qsa_pooled_keys", None)
        frozen[i] = None if p is None else mx.array(p)
    for cycle in range(CYCLES):
        head_trims.clear()
        accepts = plan(cycle)
        pending[:] = list(accepts)
        with patch.object(hs, "_batched_residual_verify", side_effect=force), \
                patch.object(hs, "_trim_self_mtp_cache_group", side_effect=spy_trim):
            proposal = propose_batched_self_mtp(model, batch)
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False] * n_lanes,
        )
        # Mechanism ran: K drafts per lane, the forced acceptances, the rest
        # rejected, and the head cache was rewound.
        assert not pending, pending
        assert proposal.draft_depths == (num_draft,) * n_lanes, proposal.draft_depths
        assert proposal.accepted_lengths == accepts, (proposal.accepted_lengths, accepts)
        assert proposal.target_drops == tuple(num_draft - a for a in accepts)
        assert any(any(c) for c in head_trims), ("no MTP-head trim ran", head_trims)
        heads = [
            (i, c)
            for i, c in enumerate(batch.caches.draft)
            if isinstance(c, (QSAKVCache, BatchQSAKVCache))
        ]
        assert heads, "no QSA MTP-head cache"
        mx.eval([c.state for _i, c in heads])
        for i, head in heads:
            pooled_seen = max(
                pooled_seen,
                _check_head_cache(model, head, (cycle, num_draft, share), frozen[i]),
            )
            head_offsets = [r[0] for r in _rows(head)[0]]
            target = _target_offsets(batch)
            # The head trails the target by the accepted pairs it has not
            # been fed yet (``pending_ts``) plus a constant per lane.
            pending_pairs = [len(lane.pending_ts) for lane in batch.lanes]
            now = [t - h - p for t, h, p in zip(target, head_offsets, pending_pairs)]
            gap = now if gap is None else gap
            assert now == gap, ("head drifted from target", cycle, now, gap, pending_pairs)
    # The post-prefill pooled-summary check was not vacuous.
    assert pooled_seen > 0
