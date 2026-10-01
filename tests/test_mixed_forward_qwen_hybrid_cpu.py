"""The mixed prefill-and-decode forward equals separate forwards (Qwen hybrid).

Per-token work runs once on the packed stream; attention and the gated-delta
recurrence run per segment against each segment's own cache.  On CPU the
result must match running each segment through the ordinary forward: logits
for every row and every cache state, including a left-padded decode batch.
"""

import copy

import mlx.core as mx
import pytest

import route_harness as rh
from mlx2.runtime.mixed_step import aligned_prompt_rows

TOL = 1e-4


@pytest.fixture(scope="module")
def lm():
    model, vocab = rh.tiny_qwen38_mtp()
    model.vocab = vocab
    return model


def _ids(n, salt, vocab):
    return [(i * 7 + salt) % (vocab - 1) + 1 for i in range(n)]


def _prefill(lm, ids):
    cache = lm.make_cache()
    for start in range(0, len(ids), 16):
        lm(mx.array([ids[start:start + 16]]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _decode_batch(lm, lengths):
    lanes = [_prefill(lm, _ids(n, 3 * j + 1, lm.vocab)) for j, n in enumerate(lengths)]
    batched = [type(lanes[0][i]).merge([lane[i] for lane in lanes]) for i in range(len(lanes[0]))]
    mx.eval([c.state for c in batched])
    return batched


def _maxdiff(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _assert_states_close(left, right):
    for c1, c2 in zip(left, right):
        for s1, s2 in zip(c1.state, c2.state):
            if s1 is None or not hasattr(s1, "shape") or not s1.size:
                continue
            assert _maxdiff(s1, s2) < TOL


def _separate_and_mixed(lm, segments):
    separate_caches = [copy.deepcopy(caches) for _, caches in segments]
    separate = [lm(tokens, cache=caches) for (tokens, _), caches in zip(segments, separate_caches)]
    mixed_caches = [copy.deepcopy(caches) for _, caches in segments]
    hidden = lm.mixed_forward([(tokens, caches) for (tokens, _), caches in zip(segments, mixed_caches)])
    mixed = [lm.logits(h) for h in hidden]
    mx.eval(separate, mixed, [[c.state for c in cs] for cs in separate_caches + mixed_caches])
    return separate, mixed, separate_caches, mixed_caches


def test_prefill_continuation_plus_left_padded_decode_batch(lm):
    prompt = _prefill(lm, _ids(40, 5, lm.vocab))
    batch = _decode_batch(lm, [30, 33, 36])  # unequal lengths: left padding
    segments = [
        (mx.array([_ids(24, 11, lm.vocab)]), prompt),
        (mx.array([[3], [5], [9]]), batch),
    ]
    separate, mixed, sep_caches, mix_caches = _separate_and_mixed(lm, segments)
    for s, m in zip(separate, mixed):
        assert s.shape == m.shape
        assert _maxdiff(s, m) < TOL
    for left, right in zip(sep_caches, mix_caches):
        _assert_states_close(left, right)


def test_first_slice_into_an_empty_cache_and_two_prompts(lm):
    batch = _decode_batch(lm, [20, 20])
    segments = [
        (mx.array([_ids(32, 2, lm.vocab)]), lm.make_cache()),
        (mx.array([_ids(17, 4, lm.vocab)]), _prefill(lm, _ids(9, 6, lm.vocab))),
        (mx.array([[7], [8]]), batch),
    ]
    separate, mixed, sep_caches, mix_caches = _separate_and_mixed(lm, segments)
    for s, m in zip(separate, mixed):
        assert _maxdiff(s, m) < TOL
    for left, right in zip(sep_caches, mix_caches):
        _assert_states_close(left, right)


def test_decoding_after_a_mixed_step_matches_the_separate_history(lm):
    batch = _decode_batch(lm, [25, 31])
    prompt = _prefill(lm, _ids(18, 9, lm.vocab))
    sep_batch, mix_batch = copy.deepcopy(batch), copy.deepcopy(batch)
    sep_prompt, mix_prompt = copy.deepcopy(prompt), copy.deepcopy(prompt)
    step = mx.array([[4], [6]])
    lm(mx.array([_ids(12, 1, lm.vocab)]), cache=sep_prompt)
    lm(step, cache=sep_batch)
    lm.mixed_forward([(mx.array([_ids(12, 1, lm.vocab)]), mix_prompt), (step, mix_batch)])
    after = mx.array([[10], [12]])
    sep_next = lm(after, cache=sep_batch)
    mix_next = lm(after, cache=mix_batch)
    mx.eval(sep_next, mix_next)
    assert _maxdiff(sep_next, mix_next) < TOL
    _assert_states_close(sep_batch, mix_batch)
    _assert_states_close(sep_prompt, mix_prompt)


def test_ordinary_call_is_unchanged_by_the_shared_cores(lm):
    """A whole prompt through ``__call__`` equals the same prompt as one
    mixed segment: the refactor kept a single code path."""
    tokens = mx.array([_ids(20, 8, lm.vocab)])
    plain = lm(tokens, cache=lm.make_cache())
    (hidden,) = lm.mixed_forward([(tokens, lm.make_cache())])
    mx.eval(plain, hidden)
    assert _maxdiff(plain, lm.logits(hidden)) < TOL


def test_refusals(lm):
    tokens = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="at least one segment"):
        lm.mixed_forward([])
    with pytest.raises(ValueError, match="one cache per layer"):
        lm.mixed_forward([(tokens, lm.make_cache()[:-1])])
    with pytest.raises(ValueError, match="token array"):
        lm.mixed_forward([(mx.array([1, 2, 3]), lm.make_cache())])
    cache = lm.make_cache()
    cache[0].speculating = True
    with pytest.raises(ValueError, match="speculative transaction"):
        lm.mixed_forward([(tokens, cache)])


@pytest.mark.parametrize(
    "budget, decode, expected",
    [(128, 1, 127), (128, 4, 124), (256, 4, 252), (448, 1, 447), (130, 1, 127),
     (64, 64, 0), (64, 70, 58), (512, 0, 512), (63, 1, 63)],
)
def test_aligned_prompt_rows_fill_whole_tiles(budget, decode, expected):
    rows = aligned_prompt_rows(budget, decode)
    assert rows == expected
    assert (rows + decode) % 64 == 0


@pytest.mark.parametrize("bad", [(0, 1), (128, -1), (True, 1), (128, 1.5)])
def test_aligned_prompt_rows_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        aligned_prompt_rows(*bad)
