"""Stale-slot NaN poisoning across mlx2's attention dispatch (llama.cpp#31913 /
flashinfer#9203 bug class).

A masked slot contributes probability exactly 0, but 0 * NaN is NaN.  On MLX
0.32.4 a NaN in a masked *value* slot reaches that row's output on the fp
full-attention path at q_len >= 16, the fp verify kernel at q_len >= 4, and
the quantized path at every q_len (decode included); a NaN in a masked *key*
slot is harmless under a boolean mask.  mlx2 stays safe because

* batch caches only ever pad with zeros (CPU test below), and
* mlx2's boolean causal masks keep a fully masked pad row finite, so the K/V a
  pad token writes into its padded slots at the next layer are finite.

These tests pin those invariants plus the two robustness properties the
kernels do have: NaN in masked keys never leaks, and NaN in one row's masked
values never leaves that row.  The Metal cases run with MLX2_RUN_GPU_TESTS=1
(while holding the GPU lease); the cache invariant runs on CPU.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.models import base
from mlx2.runtime.models.cache import BatchKVCache

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
gpu = pytest.mark.skipif(not GPU, reason="Metal case; set MLX2_RUN_GPU_TESTS=1 with the GPU lease")

B, HQ, HK, D, S, PAD = 2, 16, 4, 128, 2048, 96


@pytest.fixture(autouse=True)
def _metal_device():
    """tests/conftest.py makes the CPU the default device; the Metal cases
    must run on the GPU or the kernels' device gates decline them."""
    if not GPU:
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)

# fp_decode_kernel needs q_len 1 and >= FP_MIN_CONTEXT keys; it has its own
# long-context case below instead of a row in this table.
PATHS = {
    "fp_default": ({}, False),
    "fp_verify_kernel": ({"MLX2_FP_VERIFY_KERNEL": "1"}, False),
    "quantized": ({}, True),
}


def _engagement():
    """Snapshot of the dispatch counters that prove which kernel ran."""
    from mlx2.runtime.models import fp_verify_tensor as fpv
    from mlx2.runtime.models import qsdpa_decode_metal as qdm
    from mlx2.runtime.models import qsdpa_verify_metal as qvm

    return {
        "fp_verify": fpv.COUNTERS["calls"],
        "fp_decode": qdm.STATS["fp_decode_kernel_calls"],
        **{f"q_{k}": v for k, v in qvm.selected_host_branch_snapshot().items()},
    }


def _moved(before):
    after = _engagement()
    return {k for k in after if after[k] != before.get(k)}


def _assert_engaged(path, L, moved):
    """The case must have exercised the path it is named after."""
    if path == "fp_verify_kernel" and 2 <= L <= 16:
        assert "fp_verify" in moved, moved
    if path == "quantized":
        assert any(k.startswith("q_") for k in moved), moved


def _attend(q, k, v, mask, quantized):
    if quantized:
        cache = SimpleNamespace(bits=8, group_size=64)
        k = mx.quantize(k, group_size=64, bits=8)
        v = mx.quantize(v, group_size=64, bits=8)
    else:
        cache = SimpleNamespace()
    out = base.scaled_dot_product_attention(q, k, v, cache, D**-0.5, mask)
    mx.eval(out)
    return out


def _inputs(L, seed):
    mx.random.seed(seed)
    q = mx.random.normal((B, HQ, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, HK, S, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HK, S, D)).astype(mx.bfloat16)
    return q, k, v


def _left_padded_mask(L):
    """Row 0 is left-padded by PAD columns; both rows are causal."""
    col = mx.arange(S)[None, None, None, :]
    row = mx.arange(L)[None, None, :, None] + (S - L)
    pad = mx.array([PAD, 0])[:, None, None, None]
    return (col >= pad) & (col <= row)


def _tree_mask(L, seed):
    """Committed prefix visible to every row; the verify tail is a random tree
    (each row sees itself and a random subset of earlier tail rows)."""
    mx.random.seed(seed + 1)
    tail = mx.tril(mx.random.uniform(shape=(L, L)) < 0.5) | mx.eye(L, dtype=mx.bool_)
    prefix = mx.ones((L, S - L), mx.bool_)
    return mx.broadcast_to(mx.concatenate([prefix, tail], axis=1)[None, None], (B, 1, L, S))


def _dead_slots(mask):
    """Slots no query row of a batch row can see, shaped like K/V (B, 1, S, 1)."""
    return ~mx.any(mask, axis=2)[..., None]


@pytest.fixture
def path_env(request, monkeypatch):
    env, quantized = PATHS[request.param]
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return quantized


@gpu
@pytest.mark.parametrize("path_env", list(PATHS), indirect=True)
@pytest.mark.parametrize("L,shape", [(1, "left_padded"), (4, "left_padded"), (16, "left_padded"),
                                     (4, "tree"), (16, "tree")])
def test_nan_in_masked_keys_never_reaches_the_output(path_env, L, shape, request):
    q, k, v = _inputs(L, seed=L)
    mask = _left_padded_mask(L) if shape == "left_padded" else _tree_mask(L, L)
    dead = _dead_slots(mask)
    if shape == "tree":
        # A tree hides branch rows of the tail, not whole slots: poison every
        # key some query cannot see by giving each row its own copy.
        dead = (~mask[:, :, -1:, :]).transpose(0, 1, 3, 2)
    assert mx.any(dead).item()  # the case really poisons something
    clean_k = mx.where(dead, 0, k)
    ref = _attend(q, clean_k, v, mask, path_env)
    before = _engagement()
    out = _attend(q, mx.where(dead, mx.nan, k), v, mask, path_env)
    _assert_engaged(request.node.callspec.params["path_env"], L, _moved(before))
    if shape == "tree":
        # Only the last row's hidden slots were poisoned; earlier rows may see
        # some of them, so compare the last row (which sees none of them).
        ref, out = ref[:, :, -1:], out[:, :, -1:]
    assert mx.isfinite(out).all().item()
    assert mx.array_equal(out, ref).item()


@gpu
@pytest.mark.parametrize("path_env", list(PATHS), indirect=True)
@pytest.mark.parametrize("L", [1, 4, 16, 64])
def test_nan_in_one_rows_masked_values_stays_in_that_row(path_env, L, request):
    q, k, v = _inputs(L, seed=100 + L)
    mask = _left_padded_mask(L)
    dead = _dead_slots(mask)
    ref = _attend(q, k, mx.where(dead, 0, v), mask, path_env)
    before = _engagement()
    out = _attend(q, k, mx.where(dead, mx.nan, v), mask, path_env)
    _assert_engaged(request.node.callspec.params["path_env"], L, _moved(before))
    # Row 1 has no padding: whatever row 0's kernel does, row 1 is untouched.
    assert mx.isfinite(out[1]).all().item()
    assert mx.array_equal(out[1], ref[1]).item()


@gpu
@pytest.mark.parametrize("path_env", ["fp_default", "fp_verify_kernel", "quantized"], indirect=True)
@pytest.mark.parametrize("L", [1, 4, 16, 64])
def test_fully_masked_pad_rows_stay_finite_under_mlx2_causal_masks(path_env, L):
    """The invariant that keeps padded V slots finite: a pad token's query row
    sees nothing, and its output (hence the next layer's K/V for that slot)
    must not be NaN.  An additive -inf mask breaks this at q_len >= 16."""
    mx.random.seed(200 + L)
    q = mx.random.normal((1, HQ, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, HK, L, D)).astype(mx.bfloat16)
    pad = max(1, L // 2)
    mask = base.create_causal_mask(L, 0, left_padding=mx.array([pad]))
    assert not mx.any(mask[0, 0, 0]).item()  # the first pad row really is empty
    out = _attend(q, k, k, mask, path_env)
    assert mx.isfinite(out).all().item()


@gpu
def test_fp_decode_kernel_engages_and_contains_masked_nan(monkeypatch):
    """The opt-in fp decode kernel at its real geometry (q_len 1, 64K keys)."""
    from mlx2.runtime.models.qsdpa_decode_metal import FP_MIN_CONTEXT

    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    n = FP_MIN_CONTEXT
    mx.random.seed(300)
    q = mx.random.normal((B, HQ, 1, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, HK, n, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HK, n, D)).astype(mx.bfloat16)
    col = mx.arange(n)[None, None, None, :]
    mask = col >= mx.array([PAD, 0])[:, None, None, None]
    dead = _dead_slots(mask)
    ref = _attend(q, mx.where(dead, 0, k), mx.where(dead, 0, v), mask, False)
    before = _engagement()
    out_k = _attend(q, mx.where(dead, mx.nan, k), mx.where(dead, 0, v), mask, False)
    assert "fp_decode" in _moved(before)
    assert mx.isfinite(out_k).all().item() and mx.array_equal(out_k, ref).item()
    out_v = _attend(q, mx.where(dead, 0, k), mx.where(dead, mx.nan, v), mask, False)
    assert mx.isfinite(out_v[1]).all().item() and mx.array_equal(out_v[1], ref[1]).item()


def _keys(batch, heads, length, dim, seed):
    mx.random.seed(seed)
    return mx.random.normal((batch, heads, length, dim)) + 1.0  # no exact zeros


def test_batch_cache_padding_is_zero_filled_cpu():
    """Growth, extend and filter never expose uninitialized or stale slots: every
    slot outside a row's written span is exactly zero."""
    with mx.stream(mx.cpu):
        a = BatchKVCache(left_padding=[0])
        a.update_and_fetch(_keys(1, 2, 5, 8, 1), _keys(1, 2, 5, 8, 2))
        b = BatchKVCache(left_padding=[0])
        b.update_and_fetch(_keys(1, 2, 300, 8, 3), _keys(1, 2, 300, 8, 4))
        a.extend(b)
        keys, values = a.keys_and_values()
        mx.eval(keys, values)
        padding = [int(p) for p in a.left_padding.tolist()]
        assert padding == [295, 0]
        for tensor in (keys, values):
            assert mx.all(tensor[0, :, :295] == 0).item()
            assert mx.all(tensor[0, :, 295:] != 0).item()
            assert mx.all(tensor[1] != 0).item()
        # Capacity beyond the cursor (growth step) is zeros too.
        assert mx.all(a.keys[..., a._idx:, :] == 0).item()
        assert mx.all(a.values[..., a._idx:, :] == 0).item()
        a.filter(mx.array([0]))
        keys, values = a.keys_and_values()
        assert all(mx.isfinite(t).all().item() for t in (keys, values))
