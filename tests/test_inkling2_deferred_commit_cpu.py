"""inkling2_verify_deferred_commit and inkling2_visit_length_check, CPU only.

Deferred commit: a speculating ring forward writes nothing and the commit
writes only the kept rows in place (MTPLX fast/state.py mechanism). Every
test here compares it against the existing record-and-replay rollback path,
bitwise: the K/V each forward attends to, the logits of a tiny Inkling 2
model through real verify transactions, and the committed state of every
plane after each commit, over random accept/reject sequences that wrap the
window several times.
"""

import random

import mlx.core as mx
import pytest
from test_inkling2_cpu import _engage_splitk_reference, _tiny_model

from mlx2.runtime.models.cache import KVCache, RotatingKVCache, deferred_commit_counts


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


# ----- generic cache: RotatingKVCache alone -----


def _window(cache):
    """The rows any later read can see: the newest min(offset, max_size)."""
    if cache.keys is None:
        return None, None
    keys, values = cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    n = min(cache.offset, cache.max_size)
    return keys[..., keys.shape[2] - n :, :], values[..., values.shape[2] - n :, :]


def _same_window(a, b):
    assert a.offset == b.offset
    (ka, va), (kb, vb) = _window(a), _window(b)
    if ka is None or kb is None:
        assert ka is None and kb is None
        return
    assert ka.dtype == kb.dtype and ka.shape == kb.shape
    assert mx.array_equal(ka, kb) and mx.array_equal(va, vb)


def _rows(rng, n, dtype, heads=2, dim=4):
    k = mx.array([[[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]
                   for _ in range(heads)]]).astype(dtype)
    v = mx.array([[[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]
                   for _ in range(heads)]]).astype(dtype)
    return k, v


def _rollback_view(cache, k, v):
    cache.update_and_fetch(k, v)
    return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("window", [3, 5, 8])
@pytest.mark.parametrize("seed", range(6))
def test_random_accept_reject_matches_rollback(dtype, window, seed):
    """Random prefill chunks, plain decode, verify rounds of 1..window-1 rows
    with random accepts (including all rejected and all kept), several
    forwards per epoch, aborts: views and committed windows are bitwise the
    rollback path's, through growth, wrap and many laps of the ring."""
    rng = random.Random(seed * 31 + window)
    ref, dut = RotatingKVCache(window), RotatingKVCache(window)
    for _step in range(60):
        kind = rng.random()
        if kind < 0.15:  # prefill chunk or plain decode, not speculating
            k, v = _rows(rng, rng.choice([1, 1, 2, window + 2]), dtype)
            ref.update_and_fetch(k, v)
            dut.update_and_fetch(k, v)
        else:
            ref.start_speculation()
            dut.start_speculation(deferred_commit=True)
            forwards = 1 if rng.random() < 0.8 else 2
            total = 0
            for _ in range(forwards):
                n = rng.randint(1, window + 2)  # wider than the window too
                k, v = _rows(rng, n, dtype)
                expect = _rollback_view(ref, k, v)
                assert dut.deferred_commit_reason(k, v) is None
                got = dut.update_and_fetch(k, v, deferred=True)
                assert all(mx.array_equal(x, y) for x, y in zip(got, expect))
                assert got[0].shape == expect[0].shape
                total += n
                assert dut.offset == ref.offset and dut.deferred_rows == total
            choice = rng.random()
            if choice < 0.2:
                drop = total  # everything rejected (an abort)
            elif choice < 0.4:
                drop = 0  # everything kept: no trim, stop keeps them
            else:
                drop = rng.randint(0, total)
            if drop:
                assert ref.trim(drop) == drop
                assert dut.trim(drop) == drop
            ref.stop_speculation()
            dut.stop_speculation()
            assert dut.deferred_rows == 0 and "_deferred" not in dut.__dict__
        _same_window(ref, dut)
        # The next read of both layouts is the same as well.
        assert ref.size() == dut.size()


def test_deferred_forward_writes_nothing_until_the_commit():
    rng = random.Random(0)
    cache = RotatingKVCache(4)
    cache.update_and_fetch(*_rows(rng, 6, mx.float32))
    cache.update_and_fetch(*_rows(rng, 1, mx.float32))
    before = [mx.array(cache.keys), mx.array(cache.values), cache._idx, cache.offset]
    cache.start_speculation(deferred_commit=True)
    cache.update_and_fetch(*_rows(rng, 3, mx.float32), deferred=True)
    assert cache.offset == before[3] + 3 and cache._idx == before[2]
    assert mx.array_equal(cache.keys, before[0]) and mx.array_equal(cache.values, before[1])
    assert cache.nbytes > before[0].nbytes + before[1].nbytes  # pending rows counted
    for read in ("state", "meta_state", "keys_and_values"):
        with pytest.raises(RuntimeError, match="uncommitted deferred rows"):
            getattr(cache, read) if read != "keys_and_values" else cache.keys_and_values()
    with pytest.raises(RuntimeError, match="only its 3 uncommitted deferred rows"):
        cache.trim(4)
    assert cache.trim(3) == 3  # all rejected: the ring is untouched
    assert cache.offset == before[3] and cache._idx == before[2]
    assert mx.array_equal(cache.keys, before[0])
    cache.stop_speculation()
    assert cache.state is not None


def test_commit_writes_in_place_and_counts():
    rng = random.Random(1)
    cache = RotatingKVCache(4)
    cache.update_and_fetch(*_rows(rng, 9, mx.float32))  # concatenated chunk, 9 rows
    deferred_commit_counts(reset=True)
    for kept in (2, 3, 1):
        cache.start_speculation(deferred_commit=True)
        cache.update_and_fetch(*_rows(rng, 3, mx.float32), deferred=True)
        cache.trim(3 - kept)
        cache.stop_speculation()
        assert cache.keys.shape[2] == 4  # a ring from the first commit on
    counts = deferred_commit_counts(reset=True)
    assert counts["commit_normalize"] == 1 and counts["commit_in_place"] == 3
    assert counts["rows_kept"] == 6 and counts["rows_dropped"] == 3
    assert counts["forwards"] == 3 and "commit_replay" not in counts


def test_declines_leave_the_rollback_path():
    rng = random.Random(2)
    k, v = _rows(rng, 2, mx.float32)
    cache = RotatingKVCache(4)
    assert cache.deferred_commit_reason(k, v) == "not_speculating"
    cache.start_speculation()
    assert cache.deferred_commit_reason(k, v) == "not_armed"
    assert "_deferred_armed" not in cache.__dict__
    cache.stop_speculation()
    cache.start_speculation(deferred_commit=True)
    assert cache.deferred_commit_reason(k[0], v[0]) == "rows"
    assert cache.deferred_commit_reason(k, v[..., :1, :]) == "rows"
    cache.update_and_fetch(k, v)  # a recorded forward first
    assert cache.deferred_commit_reason(k, v) == "rollback_records"
    cache.stop_speculation()
    assert "_deferred_armed" not in cache.__dict__
    kept = RotatingKVCache(4, keep=1)
    kept.start_speculation(deferred_commit=True)
    assert kept.deferred_commit_reason(k, v) == "keep"
    other = RotatingKVCache(4)
    other.update_and_fetch(k, v)
    other.start_speculation(deferred_commit=True)
    assert other.deferred_commit_reason(k.astype(mx.bfloat16), v.astype(mx.bfloat16)) == "layout"


def _recorded(cache):
    return sum(r[0] for r in cache._rollbacks)


@pytest.mark.parametrize("widths", [(3, 3, 3, 3, 3), (80,), (40, 40), (1, 9, 2, 6, 5)])
def test_pending_rows_follow_the_rollback_window(widths):
    """Exactly the rows the rollback path can rewind stay pending (whole
    forwards, never the newest one), so trim succeeds and refuses alike."""
    rng = random.Random(3)
    ref, dut = RotatingKVCache(8), RotatingKVCache(8)
    ref.start_speculation(rollback_window=8)
    dut.start_speculation(rollback_window=8, deferred_commit=True)
    for width in widths:
        k, v = _rows(rng, width, mx.float32)
        expect = _rollback_view(ref, k, v)
        got = dut.update_and_fetch(k, v, deferred=True)
        assert all(mx.array_equal(x, y) for x, y in zip(got, expect))
        assert dut.deferred_rows == _recorded(ref)
    capacity = _recorded(ref)
    with pytest.raises(RuntimeError):
        ref.trim(capacity + 1)
    with pytest.raises(RuntimeError):
        dut.trim(capacity + 1)
    ref.trim(capacity - 1)
    dut.trim(capacity - 1)
    ref.stop_speculation()
    dut.stop_speculation()
    _same_window(ref, dut)


# ----- the tiny Inkling 2 model through verify transactions -----


def _configure(model, policy, deferred):
    from mlx2.adapters.inkling2 import validate_policy

    kernels, _ = validate_policy({**policy, "inkling2_verify_deferred_commit": deferred})
    for name in ("hard_no_thinking", "dspark_prefill_reclaim", "dspark_predraft"):
        kernels.pop(name)
    kernels["moe_nax_gather"] = False
    model.configure_kernels(**kernels)


def _same_planes(a, b):
    from mlx2.runtime.models.inkling.cache import InklingConvolutionCache

    for x, y in zip(a, b):
        assert type(x) is type(y)
        if type(x) is RotatingKVCache:
            assert x.deferred_rows == y.deferred_rows == 0
            _same_window(x, y)
        elif type(x) is KVCache:
            assert x.offset == y.offset
            assert mx.array_equal(x.keys[..., : x.offset, :], y.keys[..., : y.offset, :])
            assert mx.array_equal(x.values[..., : x.offset, :], y.values[..., : y.offset, :])
        else:
            assert type(x) is InklingConvolutionCache
            assert not x.speculating and not y.speculating
            for s, t in zip(x.cache, y.cache):
                assert (s is None) == (t is None)
                if s is not None:
                    assert s.dtype == t.dtype and mx.array_equal(s, t)


POLICIES = {
    # The shipped defaults: fused attention, sink SDPA, split-K with the ring
    # read in place (declines to the reordered copy on the CPU).
    "defaults": {},
    # Everything the defaults turn on, off: the reference composed path.
    "reference": None,
    # Fused attention with sink SDPA off: the ring + sink concatenation path.
    "fused_ring": {"inkling2_sink_sdpa": False, "inkling2_verify_splitk": False},
}


def _policy(name):
    from mlx2.adapters.inkling2 import REFERENCE_POLICY

    return dict(REFERENCE_POLICY) if POLICIES[name] is None else dict(POLICIES[name])


def _arms(policy, seed, splitk_engaged=False, monkeypatch=None, dtype=None):
    if splitk_engaged:
        _engage_splitk_reference(monkeypatch)
    rng = random.Random(seed)
    arms = []
    for deferred in (False, True):
        model = _tiny_model()
        if dtype is not None:
            model.set_dtype(dtype)
        _configure(model, policy, deferred)
        arms.append([model, model.make_cache()])
    prompt = mx.array([[rng.randrange(500) for _ in range(rng.randint(3, 20))]])
    outs = [m(prompt, cache=c) for m, c in arms]
    assert mx.array_equal(outs[0], outs[1])
    owners = [_engine_owner(m, c) for m, c in arms]
    assert [o._deferred_commit for o in owners] == [False, True]
    return rng, arms, owners


def _engine_owner(model, cache):
    """The verify owner the external-draft engine picks for this lane."""
    from types import SimpleNamespace

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    owner = ExternalDraftBatchGenerator._target_owner(
        SimpleNamespace(model=model, scheduler_stats={}), [cache])
    assert type(owner) is HybridVerifyRows
    return owner


def _round(rng, arms, owners):
    width = rng.randint(1, 8)
    plain = width == 1 and rng.random() < 0.5
    tokens = mx.array([[rng.randrange(500) for _ in range(width)]])
    outcome = rng.random()
    accepted = width if outcome < 0.25 else rng.randint(1, width)
    logits = []
    for (model, _cache), owner in zip(arms, owners):
        tx = owner.begin([width], plain_single_token=plain)
        out = model(tokens, cache=tx.caches)
        mx.eval(out)
        logits.append(out)
        if outcome > 0.9 and not plain:
            tx.abort()
        else:
            tx.commit([accepted])
    assert mx.array_equal(logits[0], logits[1])
    _same_planes(arms[0][1], arms[1][1])
    return width


@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("seed", range(3))
def test_model_rounds_match_rollback_bitwise(policy, seed):
    """Random verify widths (1-8), accepts, plain rounds and aborts on a
    window-8 model, from prompts shorter and longer than the window, for
    enough rounds to lap the ring several times: logits and every plane's
    committed state are bitwise the rollback path's."""
    rng, arms, owners = _arms(_policy(policy), seed)
    rows = sum(_round(rng, arms, owners) for _ in range(14))
    assert rows > 24
    on = arms[1][0].kernel_receipt()["verify_deferred_commit"]
    off = arms[0][0].kernel_receipt()["verify_deferred_commit"]
    assert on["selected"] and on["ring_calls"] > 0 and on["ring_declines"] == {}
    assert on["conv"].get("prefix_records", 0) > 0
    assert not off["selected"] and off["ring_calls"] == 0 and off["conv"] == {}


@pytest.mark.parametrize("seed", range(3))
def test_model_rounds_with_splitk_engaged_match_rollback(monkeypatch, seed):
    """Split-K engaged on the CPU (composed law in place of the kernel): the
    deferred view goes through the ring-read arguments the kernel receives,
    and the result is bitwise the rollback path's."""
    rng, arms, owners = _arms({}, seed, splitk_engaged=True, monkeypatch=monkeypatch)
    for _ in range(10):
        _round(rng, arms, owners)
    receipt = arms[1][0].kernel_receipt()
    assert receipt["verify_splitk_ring"].get("calls", 0) > 0
    assert receipt["verify_deferred_commit"]["ring_calls"] > 0


def test_option_is_default_off_and_inert():
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS, KERNEL_OPTIONS, validate_policy

    assert KERNEL_OPTIONS["inkling2_verify_deferred_commit"] == "verify_deferred_commit"
    assert KERNEL_OPTIONS["inkling2_visit_length_check"] == "visit_length_check"
    kernels = validate_policy(None)[0]
    assert kernels["verify_deferred_commit"] is False and kernels["visit_length_check"] is False
    assert "verify_deferred_commit" not in KERNEL_DEFAULTS
    assert "visit_length_check" not in KERNEL_DEFAULTS
    for bad in (1, "true", None):
        with pytest.raises(TypeError, match="inkling2_verify_deferred_commit must be a boolean"):
            validate_policy({"inkling2_verify_deferred_commit": bad})
    model = _tiny_model()
    _configure(model, {}, False)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), cache=cache)
    deferred_commit_counts(reset=True)
    owner = _engine_owner(model, cache)
    assert owner._deferred_commit is False
    tx = owner.begin([4])
    assert not any("_deferred_armed" in c.__dict__ for c in cache[0::2])
    model(mx.array([[1, 2, 3, 4]]), cache=tx.caches)
    assert all(c.deferred_rows == 0 for c in cache[0::2] if type(c) is RotatingKVCache)
    assert any(c._rollbacks for c in cache[0::2] if type(c) is RotatingKVCache)
    tx.commit([2])
    assert deferred_commit_counts() == {}
    receipt = model.kernel_receipt()
    assert receipt["verify_deferred_commit"]["ring_calls"] == 0
    assert receipt["visit_length_check"] == {"selected": False}


def test_unarmed_owner_declines_and_stays_on_rollback():
    """The switch on but an epoch owner that does not arm its rings (one that
    may rewind by restoring fields): every ring visit declines (counted) and
    the result is the rollback path's."""
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    prompt = mx.array([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]])
    arms = []
    for deferred in (False, True):
        model = _tiny_model()
        _configure(model, {}, deferred)
        cache = model.make_cache()
        model(prompt, cache=cache)
        tx = HybridVerifyRows([cache]).begin([5])
        out = model(mx.array([[7, 8, 9, 10, 11]]), cache=tx.caches)
        tx.commit([2])
        arms.append((model, cache, out))
    assert mx.array_equal(arms[0][2], arms[1][2])
    _same_planes(arms[0][1], arms[1][1])
    receipt = arms[1][0].kernel_receipt()["verify_deferred_commit"]
    assert receipt["ring_calls"] == 0 and receipt["ring_declines"] == {"not_armed": 40}


def test_feature_grid_lists_the_options():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "scripts/build_model_feature_grid.py").read_text()
    assert '"inkling2_verify_deferred_commit"' in text
    assert '"inkling2_visit_length_check"' in text


# ----- inkling2_visit_length_check (MTPLX #710) -----


def test_visit_length_check_refuses_mixed_visit_lengths():
    from mlx2.runtime.models.inkling2.language import visit_length_check

    model = _tiny_model()
    tokens = mx.array([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    cache = model.make_cache()
    base = model(tokens, cache=cache)
    assert visit_length_check(model.model, cache) == 11
    model.model.__dict__.pop("visit_length_counts")
    _configure(model, {"inkling2_visit_length_check": True}, False)
    cache = model.make_cache()
    got = model(tokens, cache=cache)
    assert mx.array_equal(got, base)  # no output change
    step = model(mx.array([[16]]), cache=cache)
    cache[2].offset -= 1  # a restore that landed one ring visit a token short
    with pytest.raises(ValueError, match="disagree on the committed length: \\[11, 12\\]"):
        model(mx.array([[17]]), cache=cache)
    counts = model.kernel_receipt()["visit_length_check"]
    assert counts["selected"] and counts["checks"] == 3 and counts["mismatches"] == 1
    assert step.shape[1] == 1
    # Off: the same state is not checked (today's behaviour).
    _configure(model, {}, False)
    model(mx.array([[17]]), cache=cache)


def test_visit_length_check_skips_batch_planes():
    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.inkling2.language import visit_length_check

    model = _tiny_model()
    cache = model.make_cache()
    cache[0] = BatchKVCache([0])
    assert visit_length_check(model.model, cache) == 0
    assert model.model.visit_length_counts["skipped_planes"] == 1
