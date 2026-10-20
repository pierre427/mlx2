"""Private AST copy elimination, CPU ownership/recovery evidence only."""

import copy

import mlx.core as mx
import pytest
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_ast_tree_cohort_flow_cpu import _cohort

from mlx2.runtime.ast_private_state import PrivateHistory


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_history_construction_never_scans_or_copies_committed_prefix():
    class Prefix(list):
        def __iter__(self):
            raise AssertionError("private history copied the whole prompt")

    prefix = Prefix(range(32768))
    history = PrivateHistory(prefix)
    history.extend((5, 6))
    assert len(history) == 32770
    assert history[-3:] == [32767, 5, 6]
    assert len(prefix) == 32768
    prefix.extend((99, 100))
    assert len(history) == 32770 and history[-2:] == [5, 6]


@pytest.mark.parametrize("step", [None, -3, -1, 1, 2])
def test_history_index_and_slice_semantics_match_private_list(step):
    prefix = [1, 2, 3, 4]
    history = PrivateHistory(prefix)
    history.extend((5, 6))
    expected = [1, 2, 3, 4, 5, 6]
    for start in (None, -9, -3, 0, 2, 4, 6, 9):
        for stop in (None, -9, -2, 0, 3, 4, 5, 9):
            assert (
                history[slice(start, stop, step)] == expected[slice(start, stop, step)]
            )
    assert history == expected and expected == history
    for position in range(-6, 6):
        assert history[position] == expected[position]
    for position in (-7, 6):
        with pytest.raises(IndexError):
            _ = history[position]
    prefix.clear()
    with pytest.raises(RuntimeError, match="truncated"):
        _ = history[0]


def test_private_cache_uses_descriptors_not_deepcopy_and_keeps_live_state(monkeypatch):
    _draft, batch, lanes = _cohort()
    lane = lanes[0]
    before = _committed_digest(lane.cache)

    def no_deepcopy(*args):
        raise AssertionError("AST private cache attempted a generic deep copy")

    try:
        with monkeypatch.context() as local:
            for kind in {type(cache) for cache in lane.cache}:
                local.setattr(kind, "__deepcopy__", no_deepcopy, raising=False)
            private = batch._copy_ast_private_cache(lane.cache)
        assert len(private) == len(lane.cache)
        assert all(
            left is not right for left, right in zip(private, lane.cache, strict=True)
        )
        assert _committed_digest(private) == before
        batch.model.forward_with_taps(
            mx.array([[lane.anchor, 0]]), private, batch.layers
        )
        assert _committed_digest(lane.cache) == before
        assert _committed_digest(private) != before
    finally:
        batch.close()


def test_recovery_aliases_replace_only_ast_metadata_without_recursive_copy():
    _draft, batch, lanes = _cohort()
    lane = lanes[0]

    class Receipt(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("round recovery copied replace-only AST metadata")

    receipt = Receipt(launches=0, measurements=[{"value": i} for i in range(32)])
    lane.ast_tree_tiles_receipt = receipt
    lane.ast_tree_tuning = Receipt(samples=32)
    lane.ast_source_budget = Receipt(depth=7)
    try:
        frozen, mode = batch._freeze_lane(lane)
        assert mode == "descriptor_cow"
        for key in ("ast_tree_tiles_receipt", "ast_tree_tuning", "ast_source_budget"):
            assert frozen[0][key] is getattr(lane, key)
        before = dict(receipt)
        batch._run_round(lanes)
        assert receipt == before
        assert lane.ast_tree_tiles_receipt is not receipt
    finally:
        batch.close()


def test_replaced_metadata_and_private_state_restore_after_failed_publication(
    monkeypatch,
):
    _draft, batch, lanes = _cohort()
    for lane in lanes:
        lane.ast_tree_tiles_receipt = {"launches": 0, "last": {"sentinel": lane.uid}}
        lane.ast_tree_tuning = {"samples": 1}
    before = [
        (
            _committed_digest(l.cache),
            list(l.history),
            copy.deepcopy(l.ast_tree_tiles_receipt),
        )
        for l in lanes
    ]
    commit = batch._commit

    def fail(*args, **kwargs):
        commit(*args, **kwargs)
        raise RuntimeError("injected post-publication repair fault")

    monkeypatch.setattr(batch, "_commit", fail)
    try:
        with pytest.raises(RuntimeError, match="post-publication repair fault"):
            batch._run_round(lanes)
        assert [
            (_committed_digest(l.cache), list(l.history), l.ast_tree_tiles_receipt)
            for l in lanes
        ] == before
        assert not batch._open
    finally:
        batch.close()
