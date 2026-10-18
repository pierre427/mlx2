"""Greedy native-MTP continuation branches keep cache pairs request-private."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.proposal_path_sources import NativeMTPContinuationSource
from mlx2.adapters.proposal_sources import native_mtp_source
from mlx2.runtime.hybrid_speculative import SelfMTPCachePair
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.native_mtp_continuation_tree import (
    NativeMTPContinuationTreeTransaction,
    NativeMTPSidecar,
    sidecar_from_lane,
)


def _cache(values):
    cache = KVCache()
    values = mx.array(values, mx.float32).reshape(1, 1, -1, 1)
    cache.update_and_fetch(values, values + 100)
    return cache


def _sidecar():
    return NativeMTPSidecar(
        SelfMTPCachePair([_cache([1, 2])], [_cache([3, 4])]),
        mx.array([[[5.0]]]),
        mx.array([[[6.0]]]),
        [7],
        8,
        mx.array([1, 2, 3], mx.uint32),
    )


def _digest(sidecar):
    digest = hashlib.sha256()
    digest.update(str((sidecar.cur, tuple(sidecar.pending_ts))).encode())
    for cache in (*sidecar.caches.target, *sidecar.caches.draft):
        digest.update(str(cache.offset).encode())
        for array in cache.state:
            digest.update(np.asarray(array).tobytes())
    for array in (sidecar.seed_h, sidecar.pending_hs, sidecar.token_prefix):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def _materialize(sidecar, path, ordinal):
    token = mx.array([path[-1]], mx.float32).reshape(1, 1, 1, 1)
    sidecar.caches.target[0].update_and_fetch(token, token + 100)
    sidecar.caches.draft[0].update_and_fetch(token + 10, token + 110)
    sidecar.pending_ts.append(path[-1])
    sidecar.cur = path[-1]
    sidecar.token_prefix = mx.concatenate(
        [sidecar.token_prefix, mx.array(path, mx.uint32)]
    )


def test_selected_branch_publishes_complete_pair_and_discards_siblings():
    original = _sidecar()
    before = _digest(original)
    transaction = NativeMTPContinuationTreeTransaction(
        original, [(9, 10), (9, 11)], _materialize
    )
    branch_digests = tuple(_digest(branch.sidecar) for branch in transaction.branches)

    selected = transaction.commit(1, selected_tokens=(9, 11))
    assert transaction.closed and transaction.branches == ()
    assert _digest(original) == before
    assert _digest(selected) == branch_digests[1]
    assert _digest(selected) != branch_digests[0]
    assert isinstance(selected.caches, SelfMTPCachePair)
    assert selected.cur == 11 and selected.pending_ts == [7, 11]


def test_materialization_fault_leaves_authoritative_cache_digest_unchanged():
    original = _sidecar()
    before = _digest(original)

    def fail(sidecar, path, ordinal):
        _materialize(sidecar, path, ordinal)
        if ordinal == 1:
            raise RuntimeError("branch failed")

    with pytest.raises(RuntimeError, match="branch failed"):
        NativeMTPContinuationTreeTransaction(
            original, [(9, 10), (9, 11)], fail
        )
    assert _digest(original) == before


def test_cancel_discards_private_branches_without_publication():
    original = _sidecar()
    before = _digest(original)
    transaction = NativeMTPContinuationTreeTransaction(
        original, [(9, 10), (9, 11)], _materialize
    )
    transaction.abort()
    transaction.abort()  # cancellation cleanup is idempotent
    assert transaction.closed and transaction.branches == ()
    assert _digest(original) == before
    with pytest.raises(RuntimeError, match="closed"):
        transaction.commit(0, selected_tokens=(9, 10))


@pytest.mark.parametrize(
    "temperature,distribution",
    [(0.7, "deterministic_point_mass"), (0.0, "stochastic")],
)
def test_sampled_tree_fails_closed_without_exact_joint_law(
    temperature, distribution
):
    with pytest.raises(ValueError, match="greedy-only"):
        NativeMTPContinuationTreeTransaction(
            _sidecar(),
            [(9, 10)],
            _materialize,
            sampling_temperature=temperature,
            proposal_distribution=distribution,
        )


def test_native_source_primary_path_and_sidecar_match_existing_greedy_protocol():
    # This is the same tiny loaded-head protocol used by the existing native
    # MTP composition tests.  The new ownership seam cannot alter its B1 path.
    from test_proposal_composition_cpu import _native_qsa_pair
    from mlx2.runtime.proposal_providers import ContinuationContext

    model, _draft = _native_qsa_pair("xpress")
    source = NativeMTPContinuationSource(model)
    history, anchor, depth = (1, 2, 3), 17, 2
    context = ContinuationContext(
        history,
        anchor,
        depth,
        [],
        None,
        (),
    )
    candidates = source(context, 3)
    expected = tuple(native_mtp_source(model)(list(history), anchor, depth))
    assert candidates[0].tokens == expected

    checkpoint = _sidecar()
    before = _digest(checkpoint)
    transaction = source.open_sidecar_tree(
        checkpoint, candidates[:2], _materialize
    )
    selected = transaction.commit(0, selected_tokens=expected)
    assert selected.cur == expected[-1]
    assert _digest(checkpoint) == before


def test_sidecar_from_lane_copies_pending_host_container():
    pair = SelfMTPCachePair([_cache([1])], [_cache([2])])
    lane = SimpleNamespace(
        seed_h=mx.array([[[1.0]]]),
        pending_hs=None,
        pending_ts=[3],
        cur=4,
        token_prefix=mx.array([1, 2], mx.uint32),
    )
    sidecar = sidecar_from_lane(lane, pair)
    sidecar.pending_ts.append(5)
    assert lane.pending_ts == [3]

