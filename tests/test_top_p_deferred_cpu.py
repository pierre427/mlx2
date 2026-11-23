"""Deferred fit check of the opt-in top-p window (MLX2_SAMPLING_TOPP_TOPK512, CPU).

The deferred path draws (or builds the law) from the window mask and leaves
the fit flag for the caller's own host read.  What it returns after
``resolve_top_p_deferred`` must be bitwise the default path's, hits and
forced misses alike: same tokens, same laws, same random stream.  It must
record no verify sync of its own.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import sample_utils as su
from mlx2.runtime import verify_sync

INKLING_VOCAB = 201024


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def syncs(monkeypatch):
    """Every ``record_verify_sync`` call, by site."""
    calls = []
    monkeypatch.setattr(verify_sync, "record_verify_sync", calls.append)
    return calls


def _logprobs(rows, vocab, seed, scale):
    mx.random.seed(seed)
    x = mx.random.normal((rows, vocab)) * scale
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _mixed(vocab=INKLING_VOCAB):
    """Rows: peaked, peaked, flat, peaked, flat (fits T, T, F, T, F)."""
    peaked = _logprobs(3, vocab, seed=3, scale=8.0)
    flat = _logprobs(2, vocab, seed=4, scale=0.3)
    return mx.concatenate([peaked[:2], flat[:1], peaked[2:], flat[1:]], axis=0)


INPUTS = {
    "hit": lambda: _logprobs(4, INKLING_VOCAB, seed=15, scale=8.0),
    "miss": lambda: _logprobs(4, INKLING_VOCAB, seed=16, scale=0.3),
    "mixed": _mixed,
}


def _samplers(monkeypatch, **settings):
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    off = su.make_sampler(**settings)
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    on = su.make_sampler(**settings)
    assert not hasattr(off, "deferred") and hasattr(on, "deferred")
    return off, on


def _delta(before):
    after = dict(su.TOPP_TOPK512_STATS)
    return {key: after[key] - before.get(key, 0) for key in after}


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_deferred_sampler_draws_the_default_tokens_and_stream(monkeypatch, syncs, case):
    lp = INPUTS[case]()
    mx.eval(lp)
    off, on = _samplers(monkeypatch, temp=0.8, top_p=0.9)
    hit = case == "hit"
    for seed in range(4):
        mx.random.seed(seed)
        expected = off(lp)
        follow = mx.random.uniform(shape=(4,))
        mx.eval(expected, follow)
        before = dict(su.TOPP_TOPK512_STATS)
        mx.random.seed(seed)
        pending = on.deferred(lp)
        # The caller's one read carries the token and the flag together.
        mx.eval(pending.value, pending.fits)
        assert syncs == []
        token = su.resolve_top_p_deferred(pending)
        assert mx.array_equal(token, expected).item()
        # The random state advanced exactly once, hit or miss; the redo
        # consumed none of it.
        assert mx.array_equal(mx.random.uniform(shape=(4,)), follow).item()
        assert _delta(before) == {
            "fast": int(hit), "dense_fallback": int(not hit), "eager_checks": 0,
            "deferred_checks": 1, "deferred_redos": int(not hit),
        }
    assert syncs == []


def test_mixed_batch_flags_rows_and_redoes_the_whole_draw(monkeypatch):
    lp = _mixed()
    _off, on = _samplers(monkeypatch, temp=0.8, top_p=0.9)
    mx.random.seed(0)
    pending = on.deferred(lp)
    mx.eval(pending.value, pending.fits)
    assert pending.fits.tolist() == [True, True, False, True, False]


def test_redo_uses_the_key_of_the_fast_draw(monkeypatch):
    lp = INPUTS["miss"]()
    off, on = _samplers(monkeypatch, temp=0.8, top_p=0.9)
    mx.random.seed(7)
    state_before = mx.random.state[0]
    pending = on.deferred(lp)
    state_after = mx.random.state[0]
    # A key-less draw takes split(state)[1] and keeps split(state)[0].
    expected_state, key = mx.random.split(state_before)
    assert mx.array_equal(state_after, expected_state).item()
    explicit = mx.random.categorical(
        su._scale_by_temperature(su.apply_top_p(lp, 0.9), 0.8), key=key
    )
    mx.random.seed(7)
    default = off(lp)
    redo_a, redo_b = pending.redo(), pending.redo()
    for got in (redo_a, redo_b):
        assert mx.array_equal(got, default).item()
        assert mx.array_equal(got, explicit).item()
    # Redoing reads no state: the stream is where the fast draw left it.
    assert mx.array_equal(mx.random.state[0], state_after).item()


def test_eager_sampler_still_counts_its_own_sync(monkeypatch, syncs):
    lp = INPUTS["hit"]()
    _off, on = _samplers(monkeypatch, temp=0.8, top_p=0.9)
    before = dict(su.TOPP_TOPK512_STATS)
    mx.eval(on(lp))
    assert syncs == ["sample_utils.top_p_topk512.fit"]
    assert _delta(before)["eager_checks"] == 1
    assert _delta(before)["deferred_checks"] == 0


def test_narrow_rows_leave_nothing_pending(monkeypatch, syncs):
    lp = _logprobs(2, 1024, seed=14, scale=4.0)
    off, on = _samplers(monkeypatch, temp=0.8, top_p=0.9)
    mx.random.seed(1)
    expected = off(lp)
    mx.random.seed(1)
    pending = on.deferred(lp)
    assert pending.fits is None and pending.redo is None
    before = dict(su.TOPP_TOPK512_STATS)
    assert mx.array_equal(su.resolve_top_p_deferred(pending), expected).item()
    assert _delta(before) == {k: 0 for k in before}
    assert syncs == []


def _transforms(monkeypatch, temp=0.7, top_p=0.9):
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    off = su.make_transformed_logprobs(temp, top_p=top_p)
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    on = su.make_transformed_logprobs(temp, top_p=top_p)
    return off, on


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_deferred_transform_is_bitwise_the_default(monkeypatch, syncs, case):
    lp = INPUTS[case]()
    off, on = _transforms(monkeypatch)
    expected = off(lp)
    before = dict(su.TOPP_TOPK512_STATS)
    pending = su.transform_top_p_deferred(on, lp)
    mx.eval(pending.value, pending.fits)
    assert syncs == []
    got = su.resolve_top_p_deferred(pending)
    assert mx.array_equal(got, expected).item()
    hit = case == "hit"
    assert _delta(before) == {
        "fast": int(hit), "dense_fallback": int(not hit), "eager_checks": 0,
        "deferred_checks": 1, "deferred_redos": int(not hit),
    }
    # Hit rows of the window result already equal the default.
    rows = pending.fits.tolist()
    for row, fit in enumerate(rows):
        if fit:
            assert mx.array_equal(pending.value[row], expected[row]).item()


def test_transform_without_the_fast_path_passes_through(monkeypatch):
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    off = su.make_transformed_logprobs(0.7, top_p=0.9)
    lp = INPUTS["hit"]()
    pending = su.transform_top_p_deferred(off, lp)
    assert pending.fits is None
    assert mx.array_equal(su.resolve_top_p_deferred(pending), off(lp)).item()


# ------------------------------------------- external speculative law sites --
from mlx2.runtime.external_speculative import (  # noqa: E402
    ExternalDraftBatchGenerator,
    HostDraftRow,
    TreeDraftRow,
)
from mlx2.runtime.speculative_sampling import FLyVerificationPolicy, RequestRNG  # noqa: E402

VOCAB = 4096


def _generator(batched):
    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.stops = set()
    gen.fly_verification = FLyVerificationPolicy()
    gen.scheduler_stats = {}
    gen.batched_sampled_laws = batched
    gen.tree_gates = {"logprobs_on_request": False}
    return gen


def _lanes(seed):
    return [
        SimpleNamespace(
            uid=uid, anchor=1 + uid, history=[0, uid], processors=[],
            rng=RequestRNG(seed + uid),
            sampling={"emit_logprobs": True, "sampling_temp": temp, "top_p": 0.9},
        )
        for uid, temp in enumerate((0.7, 0.9, 1.0))
    ]


def _chain_logits(seed, depth, scales):
    rng = np.random.default_rng(seed)
    rows = [scale * rng.normal(size=(depth + 1, VOCAB)) for scale in scales]
    return mx.array(np.stack(rows).astype(np.float32))


def _blocks(seed, logits, depth):
    rng = np.random.default_rng(seed + 100)
    blocks = []
    for row in range(logits.shape[0]):
        target = np.asarray(mx.argmax(logits[row], axis=-1)).tolist()
        drafts = [int(t if rng.random() < 0.7 else rng.integers(VOCAB)) for t in target[:depth]]
        laws = rng.dirichlet(np.ones(VOCAB), size=depth)
        for law, d in zip(laws, drafts):
            law[d] += 1.0
        blocks.append(HostDraftRow(drafts, [law / law.sum() for law in laws]))
    return blocks


def _assert_same(ref, got, ref_lanes, lanes):
    for a, b, la, lb in zip(ref, got, ref_lanes, lanes):
        assert (a.accepted, list(a.emitted)) == (b.accepted, list(b.emitted))
        assert la.rng.snapshot() == lb.rng.snapshot()
        assert len(a.target_laws) == len(b.target_laws)
        for x, y in zip(a.target_laws, b.target_laws):
            assert np.array_equal(x, y)


# peaked rows fit the window, flat rows miss it; the last mixes both.
SCALES = {"hit": (8.0, 8.0, 8.0), "miss": (0.3, 0.3, 0.3), "mixed": (8.0, 0.3, 8.0)}


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("case", sorted(SCALES))
def test_chain_verify_laws_match_the_default(monkeypatch, syncs, batched, case):
    depth = 3
    logits = _chain_logits(1, depth, SCALES[case])
    blocks = _blocks(1, logits, depth)
    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    ref_lanes = _lanes(5)
    ref = _generator(batched)._verify(ref_lanes, blocks, logits)
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    lanes = _lanes(5)
    before = dict(su.TOPP_TOPK512_STATS)
    got = _generator(batched)._verify(lanes, blocks, logits)
    _assert_same(ref, got, ref_lanes, lanes)
    delta = _delta(before)
    assert "sample_utils.top_p_topk512.fit" not in syncs
    assert delta["eager_checks"] == 0
    assert delta["deferred_checks"] == delta["fast"] + delta["dense_fallback"] > 0
    assert delta["deferred_redos"] == delta["dense_fallback"]
    if case == "hit":
        assert delta["deferred_redos"] == 0
    else:
        assert delta["deferred_redos"] > 0
    if case == "miss":
        assert delta["fast"] == 0


@pytest.mark.parametrize("case", ["hit", "miss"])
def test_tree_verify_laws_match_the_default(monkeypatch, syncs, case):
    depth = 4
    tokens = [11, 22, 33, 44]
    block = TreeDraftRow(tokens, list(range(-1, depth - 1)))
    scale = 8.0 if case == "hit" else 0.3
    rng = np.random.default_rng(3)
    logits = scale * rng.normal(size=(1, depth + 1, VOCAB)).astype(np.float32)
    for row, token in enumerate(tokens):
        logits[0, row, token] += 30.0 if case == "hit" else 3.0
    logits = mx.array(logits)
    sampling = {"sampling_temp": 0.8, "top_p": 0.9, "emit_logprobs": True}

    def lane():
        return SimpleNamespace(
            uid=0, history=[1, 2], processors=[], rng=RequestRNG(9),
            sampling=dict(sampling), anchor=2,
        )

    def run():
        gen, row_lane = _generator(True), lane()
        fence, state = gen._launch_tree_laws(row_lane, block, logits)
        mx.eval(*fence)
        return gen._verify_tree_batched(row_lane, block, state), row_lane, len(fence)

    monkeypatch.delenv(su.SAMPLING_TOPP_TOPK512_ENV, raising=False)
    ref, ref_lane, ref_fence = run()
    monkeypatch.setenv(su.SAMPLING_TOPP_TOPK512_ENV, "1")
    before = dict(su.TOPP_TOPK512_STATS)
    got, got_lane, got_fence = run()
    assert (got_fence, ref_fence) == (3, 2)  # the fit flag rides the fence
    assert (got.emitted, got.commit_rows) == (ref.emitted, ref.commit_rows)
    assert got_lane.rng.snapshot() == ref_lane.rng.snapshot()
    for x, y in zip(got.target_laws, ref.target_laws):
        assert np.array_equal(x, y)
    hit = case == "hit"
    assert _delta(before) == {
        "fast": int(hit), "dense_fallback": int(not hit), "eager_checks": 0,
        "deferred_checks": 1, "deferred_redos": int(not hit),
    }
    assert syncs == []
