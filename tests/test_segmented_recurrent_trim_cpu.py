"""SegmentedBatchArraysCache trim cost: one replay per accepted length, no
cohort re-join when no row rewound."""
from collections import Counter

import mlx.core as mx

from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache

B = 4
S = 4


def _view():
    notes = Counter()
    rows = []
    for i in range(B):
        row = ArraysCache(2)
        row.cache = [mx.full((1, 3, 8), float(i)), mx.full((1, 2, 4, 4), float(i))]
        row.start_speculation()
        rows.append(row)
    view = SegmentedBatchArraysCache(rows, note=lambda k, n=1: notes.update({k: n}))
    return view, rows, notes


def _verify(view, calls):
    snap = list(view.cache)

    def fn(m, snap=snap):
        calls.append(int(m))
        return [value + 100 * m for value in snap]

    view.record_rollback(S, fn, snap)
    for slot in range(2):
        view[slot] = view.cache[slot] + 1
    return snap


def _assert_view_matches_rows(view, rows):
    for slot in range(2):
        joined = mx.concatenate([row[slot] for row in rows], axis=0)
        assert mx.array_equal(view.cache[slot], joined)


def test_zero_drop_trim_skips_rejoin_and_stays_exact():
    view, rows, notes = _view()
    _verify(view, [])
    notes.clear()
    view.trim_ragged([0] * B)
    assert notes["recurrent_state_materializations"] == 0
    _assert_view_matches_rows(view, rows)


def test_zero_drop_trim_without_forward_still_rejoins():
    view, rows, notes = _view()
    notes.clear()
    view.trim_ragged([0] * B)
    assert notes["recurrent_state_materializations"] == 2


def test_partial_accept_replays_once_per_length():
    view, rows, notes = _view()
    calls = []
    snap = _verify(view, calls)
    drops = [0, 2, 2, 1]
    view.trim_ragged(drops)
    for index, (row, drop) in enumerate(zip(rows, drops)):
        for slot in range(2):
            if drop == 0:
                expected = snap[slot][index : index + 1] + 1
            else:
                expected = snap[slot][index : index + 1] + 100 * (S - drop)
            assert mx.array_equal(row[slot], expected), (index, slot)
    assert sorted(calls) == [2, 3]
    _assert_view_matches_rows(view, rows)
