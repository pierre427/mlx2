"""Opt-in top-p-only fast path (MLX2_SAMPLING_TOPP_TOPK512, CPU).

Where the fit flag is set, the window path must be bitwise ``apply_top_p``,
ties at the cutoff included; flat rows must take the dense fallback; the
sampler and transform must stay bitwise the default path either way.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import sample_utils as su

INKLING_VOCAB = 201024


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _logprobs(rows, vocab, seed, scale, dtype=mx.float32):
    mx.random.seed(seed)
    x = mx.random.normal((rows, vocab)) * scale
    return (x - mx.logsumexp(x, axis=-1, keepdims=True)).astype(dtype)


def _rows_equal(a, b):
    # array_equal treats -inf == -inf as equal and compares dtype-exact values.
    return [bool(mx.array_equal(a[i], b[i]).item()) for i in range(a.shape[0])]


def _check(lp, top_p):
    """Every row that claims to fit is bitwise the dense filter; returns fits."""
    reference = su.apply_top_p(lp, top_p)
    masked, fits = su.apply_top_p_topk512(lp, top_p)
    assert masked.dtype == reference.dtype
    fits = fits.tolist()
    equal = _rows_equal(masked, reference)
    assert all(e for f, e in zip(fits, equal) if f), (fits, equal)
    assert mx.array_equal(su.apply_top_p_fast(lp, top_p), reference).item()
    return fits


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("top_p", [0.5, 0.8, 0.95])
def test_peaked_inkling_vocab_takes_fast_path(dtype, top_p):
    lp = _logprobs(8, INKLING_VOCAB, seed=1, scale=8.0, dtype=dtype)
    assert all(_check(lp, top_p))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_flat_inkling_vocab_falls_back(dtype):
    lp = _logprobs(4, INKLING_VOCAB, seed=2, scale=0.3, dtype=dtype)
    assert not any(_check(lp, 0.9))


def test_mixed_batch_flags_each_row():
    peaked = _logprobs(3, INKLING_VOCAB, seed=3, scale=8.0)
    flat = _logprobs(2, INKLING_VOCAB, seed=4, scale=0.3)
    lp = mx.concatenate([peaked[:2], flat[:1], peaked[2:], flat[1:]], axis=0)
    assert _check(lp, 0.9) == [True, True, False, True, False]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("scale", [4.0, 5.0, 6.0])
def test_random_rows_never_claim_a_wrong_fit(seed, scale):
    # Middling entropies put the threshold near the window edge, where both
    # outcomes occur; only the claimed fits are checked.
    for top_p in (0.5, 0.9, 0.95):
        _check(_logprobs(4, INKLING_VOCAB, seed=seed, scale=scale, dtype=mx.bfloat16), top_p)


@pytest.mark.parametrize("vocab", [1500, 4096, 8192])
def test_small_vocab(vocab):
    assert all(_check(_logprobs(4, vocab, seed=5, scale=4.0), 0.9))
    assert not any(_check(_logprobs(4, vocab, seed=6, scale=0.2), 0.9))


def test_one_dimensional_row():
    row = _logprobs(1, 4096, seed=7, scale=6.0)[0]
    masked, fits = su.apply_top_p_topk512(row, 0.9)
    assert fits.item()
    assert mx.array_equal(masked, su.apply_top_p(row, 0.9)).item()


def test_top_p_near_zero_keeps_the_argmax():
    lp = _logprobs(4, INKLING_VOCAB, seed=8, scale=6.0)
    assert all(_check(lp, 1e-6))
    masked, _ = su.apply_top_p_topk512(lp, 1e-6)
    assert mx.isfinite(masked).sum(axis=-1).tolist() == [1, 1, 1, 1]


def test_top_p_near_one_falls_back():
    lp = _logprobs(4, INKLING_VOCAB, seed=9, scale=6.0)
    assert not any(_check(lp, 0.99999))


def _tied_row(vocab, tied, top_p_cut):
    """A row whose threshold cuts a group of tied tokens.

    Token 7 is the argmax; ``tied`` indices share one logit; everything else
    is far below.  Returns (logprobs, top_p) with the threshold between the
    second and third tied position of the ascending order.
    """
    logits = mx.full((vocab,), -40.0)
    logits[7] = 0.0
    logits[mx.array(tied)] = -1.0
    lp = logits - mx.logsumexp(logits)
    probs = mx.exp(lp)
    tail = mx.sum(mx.where(lp < -2.0, probs, 0.0)).item()
    q = probs[tied[0]].item()
    total = mx.sum(probs).item()
    # keep iff C > (1 - p) * total; C after k tied tokens is tail + k q.
    keep_after = tail + top_p_cut * q
    return lp, 1.0 - keep_after / total


@pytest.mark.parametrize("vocab", [4096, INKLING_VOCAB])
def test_tie_at_cutoff_keeps_highest_index_tokens(vocab):
    tied = [100, 900, 2500, 3000]
    lp, top_p = _tied_row(vocab, tied, 2.5)
    reference = su.apply_top_p(lp[None], top_p)[0]
    kept = [i for i in [7] + tied if mx.isfinite(reference[i]).item()]
    # The dense rule: stable ascending argsort puts equal values in index
    # order, and the threshold keeps a suffix, so the highest-index tied
    # tokens survive.
    assert kept == [7, 2500, 3000]
    assert mx.isfinite(reference).sum().item() == 3
    masked, fits = su.apply_top_p_topk512(lp[None], top_p)
    assert fits.tolist() == [True]
    assert mx.array_equal(masked[0], reference).item()


def test_tie_straddling_the_window_edge_falls_back():
    # 600 tokens share the second-best logit: the window (512) cannot hold
    # the tie, so the fast path must not decide which members survive.
    vocab = 4096
    logits = mx.full((vocab,), -30.0)
    logits[0] = 4.0
    logits[mx.arange(10, 610)] = -2.0
    lp = (logits - mx.logsumexp(logits))[None]
    fits = _check(lp, 0.9)
    assert fits == [False]


def test_negative_infinity_entries():
    lp = _logprobs(3, INKLING_VOCAB, seed=10, scale=8.0)
    dead = mx.arange(INKLING_VOCAB) % 3 == 0
    lp = mx.where(dead[None], -float("inf"), lp)
    assert all(_check(lp, 0.9))
    # Fewer finite tokens than the window: the -inf window entries carry no
    # mass and are dropped on both paths.
    sparse = mx.full((2, INKLING_VOCAB), -float("inf"))
    sparse[:, mx.arange(0, 3000, 10)] = _logprobs(2, 300, seed=11, scale=3.0)
    sparse = sparse - mx.logsumexp(sparse, axis=-1, keepdims=True)
    assert all(_check(sparse, 0.9))
    # An all -inf row is never claimed (the forced last position has the
    # window minimum's value); the dense result is all -inf either way.
    assert _check(mx.full((1, 4096), -float("inf")), 0.9) == [False]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    assert not su._fast_top_p_eligible(0.9, 0, 0.0)
    status = su.sampling_filter_status()["top_p_topk512"]
    assert status["policy"] == "reference" and status["default"] == "0"
    assert su.sampling_filter_receipt({"temperature": 0.7, "top_p": 0.9}) == {"policy": "reference"}


def test_eligibility_is_top_p_only(monkeypatch):
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    assert su._fast_top_p_eligible(0.9, 0, 0.0)
    assert not su._fast_top_p_eligible(0.9, 20, 0.0)
    assert not su._fast_top_p_eligible(0.9, 0, 0.05)
    assert not su._fast_top_p_eligible(0.9, 0, 0.0, xtc=True)
    assert not su._fast_top_p_eligible(1.0, 0, 0.0)
    assert not su._fast_top_p_eligible(0.0, 0, 0.0)
    assert su.sampling_filter_receipt({"temperature": 0.7, "top_p": 0.9}) == {
        "policy": "top_p_topk512_v1", "qualification": "unqualified_opt_in"}
    assert su.sampling_filter_receipt({"temperature": 0, "top_p": 0.9}) == {"policy": "reference"}
    assert su.sampling_filter_receipt({"temperature": 0.7, "top_p": 0.9, "top_k": 20}) == {
        "policy": "reference"}


@pytest.mark.parametrize("top_k", [0, 20])
def test_sampling_filter_receipt_matches_sampler_with_xtc(monkeypatch, top_k):
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    settings = {"top_p": 0.9, "top_k": top_k, "xtc_probability": 0.5}
    counts = dict(su.SAMPLING_FILTER_STATS)
    su.make_sampler(temp=0.7, **settings)
    assert su.SAMPLING_FILTER_STATS == counts
    assert su.sampling_filter_receipt({"temperature": 0.7, **settings}) == {"policy": "reference"}


def _stats():
    return dict(su.TOPP_TOPK512_STATS)


def test_transform_is_bitwise_the_default_and_counts_paths(monkeypatch):
    peaked = _logprobs(4, INKLING_VOCAB, seed=12, scale=8.0)
    flat = _logprobs(4, INKLING_VOCAB, seed=13, scale=0.3)
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    off = su.make_transformed_logprobs(0.7, top_p=0.9)
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    on = su.make_transformed_logprobs(0.7, top_p=0.9)
    assert on is not off
    before = _stats()
    assert mx.array_equal(on(peaked), off(peaked)).item()
    assert _stats()["fast"] == before["fast"] + 1
    assert mx.array_equal(on(flat), off(flat)).item()
    assert _stats()["dense_fallback"] == before["dense_fallback"] + 1
    # Narrow rows skip the window and touch no counter.
    narrow = _logprobs(2, 1024, seed=14, scale=4.0)
    assert mx.array_equal(on(narrow), off(narrow)).item()
    after = _stats()
    assert {k: after[k] - before[k] for k in after} == {
        "fast": 1, "dense_fallback": 1, "eager_checks": 2,
        "deferred_checks": 0, "deferred_redos": 0,
    }


@pytest.mark.parametrize("scale,path", [(8.0, "fast"), (0.3, "dense_fallback")])
def test_sampler_draws_the_default_tokens(monkeypatch, scale, path):
    lp = _logprobs(8, INKLING_VOCAB, seed=15, scale=scale)
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    off = su.make_sampler(temp=0.8, top_p=0.9)
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    on = su.make_sampler(temp=0.8, top_p=0.9)
    for seed in range(4):
        mx.random.seed(seed)
        expected = off(lp)
        follow = mx.random.uniform(shape=(4,))
        before = _stats()
        mx.random.seed(seed)
        got = on(lp)
        # Same tokens and the same random-state advance on both outcomes.
        assert mx.array_equal(got, expected).item()
        assert mx.array_equal(mx.random.uniform(shape=(4,)), follow).item()
        assert _stats()[path] == before[path] + 1


def test_status_reports_constructions(monkeypatch):
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    before = dict(su.SAMPLING_FILTER_STATS)
    su.make_sampler(temp=0.7, top_p=0.9)
    su.make_transformed_logprobs(0.7, top_p=0.9)
    status = su.sampling_filter_status()
    assert status["top_p_topk512"]["policy"] == "top_p_topk512_v1"
    assert status["top_p_topk512"]["window"] == 512
    counts = status["counts"]
    assert counts["sampler_topp_topk512"] == before.get("sampler_topp_topk512", 0) + 1
    assert counts["transform_topp_topk512"] == before.get("transform_topp_topk512", 0) + 1


def test_threshold_within_rounding_of_a_prefix_sum_is_not_decided():
    # Put (1 - top_p) * total exactly on one prefix sum of the dense order:
    # float32 rounding decides that position, so the window must refuse it,
    # and accept a threshold halfway between two prefix sums.
    import numpy as np

    lp = _logprobs(1, INKLING_VOCAB, seed=16, scale=8.0)
    probs = np.sort(np.exp(np.array(lp[0], dtype=np.float64)))
    prefix = np.cumsum(probs)
    total = prefix[-1]
    j = INKLING_VOCAB - 4  # inside the nucleus of a peaked row
    on_sum = 1.0 - prefix[j] / total
    between = 1.0 - (prefix[j] + 0.5 * probs[j + 1]) / total
    assert _check(lp, on_sum) == [False]
    assert _check(lp, between) == [True]
