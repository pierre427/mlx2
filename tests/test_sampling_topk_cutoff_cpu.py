"""Opt-in fused top-p + top-k cutoff filter (MLX2_SAMPLING_TOPK_CUTOFF, CPU).

The cutoff must never keep more than ``top_k``, keep the reference chain's
count except within float32 rounding of the nucleus threshold, keep the
reference set except inside a tie at the boundary, and stay off unless the
flag is set.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import sample_utils as su


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _logprobs(rows, vocab, seed, dtype=mx.bfloat16, scale=2.5):
    mx.random.seed(seed)
    x = mx.random.normal((rows, vocab)) * scale
    return (x - mx.logsumexp(x, axis=-1, keepdims=True)).astype(dtype)


def _reference(lp, top_p, top_k):
    return su.apply_top_k(su.apply_top_p(lp, top_p), top_k)


@pytest.mark.parametrize("top_p,top_k", [(0.95, 20), (0.5, 40), (0.999, 5), (0.05, 20), (0.9, 1)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_cutoff_matches_reference_count_and_differs_only_inside_a_tie(top_p, top_k, dtype):
    lp = _logprobs(64, 4096, seed=top_k, dtype=dtype)
    ref = mx.isfinite(_reference(lp, top_p, top_k))
    got = mx.isfinite(su.apply_top_p_top_k_cutoff(lp, top_p, top_k))
    mx.eval(ref, got)
    assert mx.array_equal(ref.sum(axis=-1), got.sum(axis=-1)).item()
    assert int(got.sum(axis=-1).max().item()) <= top_k
    values = lp.astype(mx.float32)
    for row in range(lp.shape[0]):
        r, g = ref[row], got[row]
        if mx.array_equal(r, g).item():
            continue
        # Only tokens tied at the smallest kept value may be swapped.
        floor = mx.min(mx.where(r, values[row], mx.inf))
        assert mx.all(mx.where(r != g, values[row] == floor, True)).item(), f"row {row}"


def _fp64_nucleus(row, top_p, top_k):
    """(count, margin of the closest decision to the threshold) in float64."""
    import numpy as np

    values = np.sort(row.astype(np.float64))[::-1]
    probs = np.exp(values)
    total = probs.sum()
    above = np.cumsum(probs) - probs
    count = min(max(1, int((above < top_p * total).sum())), top_k)
    edge = above[max(count - 1, 0): count + 1]
    return count, float(np.min(np.abs(edge - top_p * total)) / total)


@pytest.mark.parametrize("top_k,scale,dtype", [
    (100, 5.0, mx.float32), (100, 6.0, mx.bfloat16), (200, 4.5, mx.bfloat16), (1000, 4.0, mx.float32),
])
def test_nucleus_near_k_at_serving_vocab(top_k, scale, dtype):
    """The regime where top-p binds close to k on a 248K vocabulary: counts
    agree except within float32 rounding, and then this side is right."""
    import numpy as np

    lp = _logprobs(60, 248_320, seed=top_k + int(scale * 10), dtype=dtype, scale=scale)
    ref = mx.isfinite(_reference(lp, 0.9, top_k)).sum(axis=-1)
    got = mx.isfinite(su.apply_top_p_top_k_cutoff(lp, 0.9, top_k)).sum(axis=-1)
    mx.eval(ref, got)
    rows = np.asarray(lp.astype(mx.float32))
    assert int(got.max().item()) <= top_k
    for row in np.nonzero(np.asarray(ref != got))[0]:
        count, margin = _fp64_nucleus(rows[row], 0.9, top_k)
        assert abs(int(ref[row].item()) - int(got[row].item())) == 1
        assert margin < 1e-5, f"row {row}: count differs {margin:.2e} from the threshold"
        assert int(got[row].item()) == count, f"row {row}: fused count is not the float64 one"


def test_fusion_is_capped_at_k_1024(monkeypatch):
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    assert su._fuse_top_p_top_k(0.95, 1024, 0.0)
    assert not su._fuse_top_p_top_k(0.95, 1025, 0.0)


def test_boundary_tie_never_exceeds_k():
    # Ten tokens tie at the 3rd-largest value: the reference keeps 3, so must we.
    row = mx.array([[5.0, 4.0] + [3.0] * 10 + [0.0] * 52])
    lp = row - mx.logsumexp(row, axis=-1, keepdims=True)
    got = mx.isfinite(su.apply_top_p_top_k_cutoff(lp, 0.999, 3))
    assert int(got.sum().item()) == 3
    assert got[0, 0].item() and got[0, 1].item() and got[0, 2].item()


def test_tie_free_rows_are_identical_at_serving_vocab():
    lp = _logprobs(8, 248_320, seed=7, dtype=mx.float32)
    mx.eval(lp)
    assert mx.array_equal(
        mx.isfinite(_reference(lp, 0.95, 20)),
        mx.isfinite(su.apply_top_p_top_k_cutoff(lp, 0.95, 20)),
    ).item()


def test_flat_and_masked_rows_keep_the_most_likely_token():
    lp = mx.where(mx.arange(512) == 3, 0.0, -float("inf"))[None]
    got = mx.isfinite(su.apply_top_p_top_k_cutoff(lp, 0.9, 10))
    assert got[0, 3].item() and int(got.sum().item()) == 1
    # A row whose mass underflows to 0 keeps at least the top-k floor token.
    dead = mx.full((1, 512), -200.0).at[0, 5].add(1.0)
    kept = mx.isfinite(su.apply_top_p_top_k_cutoff(dead, 0.9, 10))
    assert kept[0, 5].item()


def test_invalid_top_k_refuses_like_reference():
    lp = _logprobs(1, 64, seed=1)
    with pytest.raises(ValueError):
        su.apply_top_p_top_k_cutoff(lp, 0.9, 64)
    with pytest.raises(ValueError):
        su.apply_top_p_top_k_cutoff(lp, 0.9, 0)


def test_off_by_default_and_counted_when_on(monkeypatch):
    monkeypatch.delenv(su.SAMPLING_TOPK_CUTOFF_ENV, raising=False)
    assert not su._fuse_top_p_top_k(0.95, 20, 0.0)
    assert su.sampling_filter_status()["policy"] == "reference"
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    assert su._fuse_top_p_top_k(0.95, 20, 0.0)
    # Only the exact top-p + top-k pair fuses; min-p, XTC or a lone filter keep
    # the reference chain.
    assert not su._fuse_top_p_top_k(0.95, 20, 0.05)
    assert not su._fuse_top_p_top_k(0.95, 20, 0.0, xtc=True)
    assert not su._fuse_top_p_top_k(1.0, 20, 0.0)
    assert not su._fuse_top_p_top_k(0.95, 0, 0.0)
    before = dict(su.SAMPLING_FILTER_STATS)
    su.make_sampler(temp=0.7, top_p=0.95, top_k=20)
    su.make_transformed_logprobs(0.7, top_p=0.95, top_k=20)
    status = su.sampling_filter_status()
    assert status["policy"] == "top_p_top_k_cutoff_v1"
    assert status["counts"]["sampler_cutoff"] == before.get("sampler_cutoff", 0) + 1
    assert status["counts"]["transform_cutoff"] == before.get("transform_cutoff", 0) + 1


def test_transform_memo_follows_the_flag(monkeypatch):
    lp = _logprobs(4, 2048, seed=3, dtype=mx.float32)
    monkeypatch.delenv(su.SAMPLING_TOPK_CUTOFF_ENV, raising=False)
    off = su.make_transformed_logprobs(0.7, top_p=0.9, top_k=16)
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    on = su.make_transformed_logprobs(0.7, top_p=0.9, top_k=16)
    assert on is not off
    a, b = off(lp), on(lp)
    assert mx.array_equal(mx.isfinite(a).sum(axis=-1), mx.isfinite(b).sum(axis=-1)).item()
    finite = mx.isfinite(a) & mx.isfinite(b)
    assert mx.allclose(mx.where(finite, a, 0.0), mx.where(finite, b, 0.0)).item()


def test_sampler_draws_only_kept_tokens(monkeypatch):
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    lp = _logprobs(256, 1024, seed=11, dtype=mx.float32, scale=1.0)
    allowed = mx.isfinite(_reference(lp, 0.8, 8))
    tokens = su.make_sampler(temp=1.0, top_p=0.8, top_k=8)(lp)
    picked = mx.take_along_axis(allowed, tokens[:, None], axis=-1)
    assert mx.all(picked).item()


def test_route_receipt_label(monkeypatch):
    monkeypatch.delenv(su.SAMPLING_TOPK_CUTOFF_ENV, raising=False)
    sampled = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    assert su.sampling_filter_receipt(sampled) == {"policy": "reference"}
    monkeypatch.setenv(su.SAMPLING_TOPK_CUTOFF_ENV, "1")
    assert su.sampling_filter_receipt(sampled) == {
        "policy": "top_p_top_k_cutoff_v1", "qualification": "unqualified_opt_in"}
    assert su.sampling_filter_receipt({**sampled, "temperature": 0}) == {"policy": "reference"}
    assert su.sampling_filter_receipt(None) == {"policy": "reference"}
