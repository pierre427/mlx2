"""CPU-only launch contract for quantized indexed QSA with independent slots.

Independent decode slots (query length 1 per lane) at B2/B4 must reach the
native quantized path as ONE partition launch, batch folded into grid z,
plus ONE combine launch, never a per-lane loop. These tests pin that
contract on the existing ``qwen4_qsa_indexed`` dispatch using fake kernel
builders that record every launch; nothing here constructs or dispatches a
Metal kernel, queries GPU capability, or claims GPU bit exactness or
serving-route engagement. Geometry: D=256, GQA=12 (2 KV heads, 24 query
heads), q4 and q8 affine K/V, ragged compact rows (different left padding,
query positions, selected counts), and packed K/V sliced from a larger
capacity buffer so the logical KV width is below capacity.
"""

import mlx.core as mx

mx.set_default_device(mx.cpu)  # before any tensor in this module

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlx2.runtime.models import qwen4_qsa_indexed as indexed  # noqa: E402
from mlx2.runtime.models.qwen4_exp import QSACompactBlocks  # noqa: E402
from mlx2.runtime.models.qwen4_qsa_nax import compact_blocks_to_kernel_inputs  # noqa: E402

DIM, KV_HEADS, GQA = 256, 2, 12
Q_HEADS = KV_HEADS * GQA
TOTAL, CAPACITY, SELECTED = 1100, 1536, 263  # u_width 264 -> 1056 tokens, inside (1024, 8192]
GROUP = 64


def _ragged_compact(batch, seed=7):
    rng = np.random.default_rng(seed)
    left = np.array([0, 3, 5, 1][:batch], dtype=np.int32)
    ids = np.zeros((batch, 1, SELECTED), dtype=np.uint32)
    counts = np.zeros((batch, 1), dtype=np.int32)
    stop = np.zeros((batch, 1), dtype=np.int32)
    mask = np.zeros((batch, 1, 1, TOTAL), dtype=bool)
    for b in range(batch):
        logical = TOTAL - int(left[b]) - 4 * b          # each row ends somewhere else
        q_pos = logical - 1
        closed = (q_pos + 1) // 4
        count = SELECTED - 5 * b                        # ragged selected counts
        ids[b, 0, :count] = np.sort(rng.choice(closed, count, replace=False))
        counts[b, 0] = count
        stop[b, 0] = logical
        token = np.arange(TOTAL) - left[b]
        mask[b, 0, 0] = (token >= 0) & (token <= q_pos)
    return QSACompactBlocks(
        block_ids=mx.array(ids), block_counts=mx.array(counts), tail_start=mx.array(stop // 4 * 4),
        tail_stop=mx.array(stop), left_padding=mx.array(left), block_size=4, physical_width=TOTAL,
        causal_mask=mx.array(mask))


def _quantized_slots(batch, bits, seed=11):
    """Distinct rows; packed K/V sliced from a capacity buffer (logical < capacity)."""
    mx.random.seed(seed)
    q = mx.random.normal((batch, Q_HEADS, 1, DIM)).astype(mx.bfloat16)
    full_k = mx.random.normal((batch, KV_HEADS, CAPACITY, DIM)).astype(mx.bfloat16)
    full_v = mx.random.normal((batch, KV_HEADS, CAPACITY, DIM)).astype(mx.bfloat16)
    buffers = (mx.quantize(full_k, group_size=GROUP, bits=bits), mx.quantize(full_v, group_size=GROUP, bits=bits))
    q_keys, q_values = (tuple(x[:, :, :TOTAL, :] for x in triple) for triple in buffers)
    return q, q_keys, q_values, buffers


class FakeKernel:
    """Stands in for an ``mx.fast.metal_kernel`` object: records, returns zeros."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, **kwargs):
        self.log.append((self.name, kwargs))
        return [mx.zeros(shape, dtype=dtype) for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])]


def _forbid(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be reached on the CPU contract path")
    return fail


@pytest.fixture
def launches(monkeypatch):
    """Fake kernel builders + forbidden GPU capability calls; returns the launch log."""
    log = []
    monkeypatch.setattr(indexed, "_quantized_partition_kernel", lambda: FakeKernel("partition", log))
    monkeypatch.setattr(indexed, "_combine_kernel", lambda gated=False: FakeKernel(f"combine:{gated}", log))
    for name in ("_partition_kernel", "_private_delta_partition_kernel",
                 "_private_delta_exact_set_partition_kernel", "_measure_candidates", "combine_indexed_partials"):
        monkeypatch.setattr(indexed, name, _forbid(name))
    monkeypatch.setattr(mx.fast, "metal_kernel", _forbid("mx.fast.metal_kernel"))
    monkeypatch.setattr(mx.metal, "is_available", _forbid("mx.metal.is_available"))
    monkeypatch.setattr(mx, "device_info", _forbid("mx.device_info"))
    monkeypatch.delenv("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE", raising=False)
    assert mx.default_device() == mx.cpu
    return log


def _unpack(log):
    partition = [kw for name, kw in log if name == "partition"]
    combine = [kw for name, kw in log if name.startswith("combine")]
    return partition, combine


def _assert_partition(kw, *, batch, bits, compact, q, q_keys, q_values, buffers, threads, splits, hpt):
    head_slices = GQA // hpt
    assert kw["grid"] == (threads, 1, batch * KV_HEADS * splits * head_slices)  # batch folded into z
    assert kw["threadgroup"] == (threads, 1, 1)
    assert kw["output_shapes"] == [(batch, Q_HEADS, 1, 128), (batch, Q_HEADS, 1, 128),
                                   (batch, Q_HEADS, 1, 128, DIM), (1,)]
    assert kw["output_dtypes"] == [mx.float32, mx.float32, mx.bfloat16, mx.uint32]
    template = dict(kw["template"])
    assert template == {"T": mx.bfloat16, "D": DIM, "NQH": Q_HEADS, "NKVH": KV_HEADS, "GQA": GQA, "BS": 4,
                        "S": splits, "HPT": hpt, "BLOCKS": 128, "HAS_MASK": 1, "GROUP_SIZE": GROUP,
                        "KBITS": bits, "VBITS": bits}
    (iq, kw_, ks, kb, vw, vs, vb, ids, counts, n_sel, qpos, left, mask, scale, dims) = kw["inputs"]
    assert np.array_equal(np.array(iq.astype(mx.float32)), np.array(q.astype(mx.float32)))
    packed = DIM * bits // 32
    # Packed weights, scales and biases keep their geometry and per-row identity
    # after mx.contiguous: each is the logical slice of its own capacity buffer.
    for got, src, buf, width in ((kw_, q_keys[0], buffers[0][0], packed), (ks, q_keys[1], buffers[0][1], DIM // GROUP),
                                 (kb, q_keys[2], buffers[0][2], DIM // GROUP), (vw, q_values[0], buffers[1][0], packed),
                                 (vs, q_values[1], buffers[1][1], DIM // GROUP), (vb, q_values[2], buffers[1][2], DIM // GROUP)):
        assert got.shape == (batch, KV_HEADS, TOTAL, width) and got.dtype == src.dtype
        as_np = (lambda a: np.array(a.astype(mx.float32))) if src.dtype != mx.uint32 else np.array
        for b in range(batch):
            assert np.array_equal(as_np(got[b]), as_np(buf[b, :, :TOTAL, :]))
        if batch > 1:
            assert not np.array_equal(as_np(got[0]), as_np(got[1]))  # rows are distinguishable
    expected = compact_blocks_to_kernel_inputs(compact)
    (e_ids, e_counts, e_sel, u_width, e_qpos, e_left, total) = expected
    assert ids.shape == (batch, 1, u_width) and ids.dtype == mx.uint32
    for got, want in ((ids, e_ids), (counts, e_counts), (n_sel, e_sel), (qpos, e_qpos), (left, e_left)):
        assert np.array_equal(np.array(got), np.array(want))
    assert len({int(c) for c in np.array(n_sel).ravel()}) == batch  # ragged rows stayed ragged, in order
    assert mask.shape == (batch, 1, TOTAL) and mask.dtype == mx.bool_
    assert np.array_equal(np.array(mask), np.array(compact.causal_mask[:, 0]))
    assert np.array(dims).tolist() == [1, total, u_width] and total == TOTAL < CAPACITY
    assert scale.dtype == mx.float32


CASES = [(b, bits) for b in (2, 4) for bits in (4, 8)]


@pytest.mark.parametrize("batch,bits", CASES)
@pytest.mark.parametrize("threads,splits,hpt", [(384, 32, 12), (192, 16, 6)])
def test_partition_dispatch_is_one_launch_with_batch_in_grid_z(launches, batch, bits, threads, splits, hpt):
    compact = _ragged_compact(batch)
    q, q_keys, q_values, buffers = _quantized_slots(batch, bits)
    m, l, o, engaged = indexed._quantized_partition_dispatch(
        q, q_keys, q_values, compact, scale=DIM ** -0.5, threads=threads, splits=splits, hpt=hpt,
        group_size=GROUP, key_bits=bits, value_bits=bits)
    partition, combine = _unpack(launches)
    assert len(partition) == 1 and combine == []
    _assert_partition(partition[0], batch=batch, bits=bits, compact=compact, q=q, q_keys=q_keys,
                      q_values=q_values, buffers=buffers, threads=threads, splits=splits, hpt=hpt)
    assert m.shape == l.shape == (batch, Q_HEADS, 1, 128) and o.shape == (batch, Q_HEADS, 1, 128, DIM)


@pytest.mark.parametrize("batch", [2, 4])
def test_combine_is_one_launch_with_batch_in_grid_z(launches, batch):
    m = mx.zeros((batch, Q_HEADS, 1, 128), dtype=mx.float32)
    o = mx.zeros((batch, Q_HEADS, 1, 128, DIM), dtype=mx.bfloat16)
    engaged = mx.array([5], dtype=mx.uint32)
    out, counter = indexed._combine_sdpa_partials(m, m, o, engaged, output_dtype=mx.bfloat16)
    partition, combine = _unpack(launches)
    assert partition == [] and len(combine) == 1
    kw = combine[0]
    assert kw["grid"] == (1024, 1, batch * Q_HEADS) and kw["threadgroup"] == (1024, 1, 1)
    assert kw["output_shapes"] == [(batch, Q_HEADS, 1, DIM)] and kw["output_dtypes"] == [mx.bfloat16]
    assert kw["inputs"][0] is m and kw["inputs"][2] is o and np.array(kw["inputs"][3]).tolist() == [1]
    assert out.shape == (batch, Q_HEADS, 1, DIM) and counter is engaged


@pytest.mark.parametrize("batch,bits", CASES)
def test_attention_entry_launches_one_partition_and_one_merge(launches, monkeypatch, batch, bits):
    """Full quantized entry with a cached candidate: exactly two launches for B lanes."""
    compact = _ragged_compact(batch)
    q, q_keys, q_values, buffers = _quantized_slots(batch, bits)
    monkeypatch.setenv("MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX", "1")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    monkeypatch.setattr(indexed, "indexed_kernel_available", lambda: True)
    monkeypatch.setattr(indexed, "_sdpa_header_state", lambda: (True, None, None))
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g17s"})  # mocked, not queried
    attested = []
    monkeypatch.setattr(indexed, "_device_attest_output",
                        lambda output, counter, **kw: attested.append((output, counter, kw)) or output)
    candidate = (384, 32, 12)
    u_width = compact_blocks_to_kernel_inputs(compact)[3]
    key = (str(getattr(mx, "__version__", "unknown")), str(q.dtype), DIM, Q_HEADS, KV_HEADS, 4, int(u_width),
           batch, 1, 1, 32, 12, GROUP, bits, bits)
    monkeypatch.setitem(indexed._QUANTIZED_PROBE_RESULTS, key, candidate)
    monkeypatch.setitem(indexed._PROBE_TIMINGS, key, {})
    out = indexed.qwen4_qsa_indexed_quantized_attention(
        q, q_keys, q_values, compact, scale=DIM ** -0.5, group_size=GROUP, key_bits=bits, value_bits=bits,
        splits=32, hpt=12)
    partition, combine = _unpack(launches)
    assert [name for name, _ in launches] == ["partition", "combine:False"]   # no per-lane loop
    _assert_partition(partition[0], batch=batch, bits=bits, compact=compact, q=q, q_keys=q_keys,
                      q_values=q_values, buffers=buffers, threads=384, splits=32, hpt=12)
    assert combine[0]["grid"] == (1024, 1, batch * Q_HEADS)
    assert out.shape == (batch, Q_HEADS, 1, DIM) and out.dtype == mx.bfloat16
    (_, counter, kw), = attested
    assert kw["candidate"] == candidate and kw["length"] == 1 and kw["context"] == TOTAL
    assert kw["geometry_key"] == f"quantized-B{batch}-L1-U{u_width}-dtype{q.dtype}-mask1-g{GROUP}-k{bits}-v{bits}"


def test_launch_count_does_not_scale_with_lanes(launches):
    """B2 and B4 each cost one partition launch; only grid z grows."""
    z = {}
    for batch in (2, 4):
        launches.clear()
        compact = _ragged_compact(batch)
        q, q_keys, q_values, _ = _quantized_slots(batch, 4)
        indexed._quantized_partition_dispatch(q, q_keys, q_values, compact, scale=DIM ** -0.5, threads=384,
                                              splits=32, hpt=12, group_size=GROUP, key_bits=4, value_bits=4)
        partition, _ = _unpack(launches)
        assert len(partition) == 1
        z[batch] = partition[0]["grid"][2]
    assert z[4] == 2 * z[2] == 4 * KV_HEADS * 32


def test_contract_guards_still_fire_on_cpu(launches):
    """The CPU fixture never reaches a real kernel: the declined path stays declined."""
    compact = _ragged_compact(2)
    q, q_keys, q_values, _ = _quantized_slots(2, 4)
    with pytest.raises(ValueError, match="one SIMD group per head"):
        indexed._quantized_partition_dispatch(q, q_keys, q_values, compact, scale=1.0, threads=256, splits=32,
                                              hpt=12, group_size=GROUP, key_bits=4, value_bits=4)
    assert launches == []
