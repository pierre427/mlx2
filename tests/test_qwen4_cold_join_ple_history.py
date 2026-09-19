"""A cold row joining warm rows keeps the model's empty-PLE-history reading.

``ArraysCache.extend``/``merge`` fill a row whose state slot is still None.
Qwen4 PLE slot 3 (n-gram token history) reads None as all-EOS, so a zero fill
seeded token-0 history into the cold row and it diverged from its solo run
(cf. omlx#3703: joining must not alter a row's state).
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen4_exp import Qwen4ArraysCache
from test_batched_mtp import _tiny_qwen4_model


def _drive(model, schedule):
    gen = BatchGenerator(
        model,
        prefill_step_size=7,
        apc_interior_checkpoints={"count": 0, "min_stride": 4},
        completion_batch_size=4,
        prefill_batch_size=4,
    )
    tokens, uid_of, finished = {}, {}, set()
    pending = list(schedule)
    step = 0
    try:
        while pending or len(finished) < len(uid_of):
            while pending and pending[0][0] <= step:
                _, key, prompt, max_tokens = pending.pop(0)
                uid = gen.insert([prompt], max_tokens=[max_tokens])[0]
                uid_of[uid] = key
                tokens[key] = []
            for response in gen.next()[1]:
                tokens[uid_of[response.uid]].append(int(response.token))
                if response.finish_reason:
                    finished.add(response.uid)
            step += 1
            assert step < 2000
    finally:
        gen.close()
    return tokens


def test_plain_late_join_matches_solo_runs():
    mx.random.seed(41)
    model = _tiny_qwen4_model()
    schedule = [
        (0, 0, [(5 * i + 3) % 60 + 2 for i in range(40)], 30),
        (6, 1, [(7 * i + 1) % 60 + 2 for i in range(37)], 24),
        (9, 2, [(11 * i + 5) % 60 + 2 for i in range(19)], 20),
    ]
    batched = _drive(model, schedule)
    for _, key, prompt, max_tokens in schedule:
        assert batched[key] == _drive(model, [(0, key, prompt, max_tokens)])[key], key


def test_empty_ple_history_is_filled_with_eos_on_extend_and_merge():
    warm = Qwen4ArraysCache(size=4)
    warm.ple_history_fill = 7
    for slot in range(4):
        warm[slot] = mx.ones((1, 2), dtype=mx.int64)
    cold = Qwen4ArraysCache(size=4)

    merged = Qwen4ArraysCache.merge([cold, warm])
    assert merged[3].tolist() == [[7, 7], [1, 1]]
    assert merged[0].tolist() == [[0, 0], [1, 1]]

    warm.extend(cold)
    assert warm[3].tolist() == [[1, 1], [7, 7]]
    assert warm[0].tolist() == [[1, 1], [0, 0]]


def test_unknown_ple_history_fill_fails_closed():
    warm = Qwen4ArraysCache(size=4)
    for slot in range(4):
        warm[slot] = mx.ones((1, 2), dtype=mx.int64)
    with pytest.raises(RuntimeError, match="PLE history fill"):
        Qwen4ArraysCache.merge([Qwen4ArraysCache(size=4), warm])


def test_plain_arrays_cache_still_zero_fills():
    warm = ArraysCache(size=2)
    warm[0] = mx.ones((1, 3))
    warm[1] = mx.ones((1, 3))
    merged = ArraysCache.merge([ArraysCache(size=2), warm])
    assert merged[0].tolist() == [[0, 0, 0], [1, 1, 1]]


def test_merging_restored_warm_rows_needs_no_fill_id():
    # APC disk restores rebuild caches without ple_history_fill; merging rows
    # that are all populated must not consult the empty-slot fill.
    rows = []
    for value in (1, 2):
        cache = Qwen4ArraysCache(size=4)
        for slot in range(4):
            cache[slot] = mx.full((1, 2), value, dtype=mx.int64)
        rows.append(cache)
    merged = Qwen4ArraysCache.merge(rows)
    assert merged[3].tolist() == [[1, 1], [2, 2]]
