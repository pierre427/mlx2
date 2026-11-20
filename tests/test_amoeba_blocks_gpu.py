"""aMoEba block-sparse kernels on Metal (MLX2_RUN_GPU_TESTS=1; never part of the CPU suite).

Per backend this host has (mpp needs M5 tensor units; simd runs everywhere, M5 included):
* all open: the packed layout gives the unpacked (identity-table) layout's bits;
* gates closed: skipped == masked (SKIP=0, every block computed with zeros) bit for bit, and both
  equal the float64 masked reference within dtype tolerance;
* closed blocks are not read: device counters equal ``layout.predicted_reads`` (gate/up and
  down), and NaN poisoned into closed down groups' scales never reaches the output (a gate/up
  closed row is written as 0 whether or not it was computed, so poison cannot tell for gate/up:
  the counters are that proof);
* short chunks (1, 7, 31 rows) on the MPP extent match the reference;
* a row's output does not depend on the rows beside it (decode row == the same row in a batch);
* MoE: sorted and unsorted routes agree bit for bit; an expert with no open block yields zeros;
* tile gates: the kernel's open bits equal the numpy closure reference (and the union).
"""

import os

import mlx.core as mx
import numpy as np
import pytest

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
pytestmark = pytest.mark.skipif(not GPU or not mx.metal.is_available(),
                                reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")

from mlx2.runtime.amoeba_blocks import layout as L  # noqa: E402
from mlx2.runtime.amoeba_blocks import ops as O  # noqa: E402
from mlx2.runtime.amoeba_blocks import reference as R  # noqa: E402


def _backends():
    from mlx2.runtime.lane import matmul as lane
    return ["mpp", "simd"] if lane._tensor_units() else ["simd"]


@pytest.fixture(autouse=True)
def gpu():
    old = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(old)


def quantized(rng, E, N, Kd, bits):
    w = mx.array((rng.standard_normal((E, N, Kd)) * 0.05).astype(np.float32))
    q = mx.quantize(w, group_size=64, bits=bits)
    return O.QW(q[0], q[1].astype(mx.bfloat16), q[2].astype(mx.bfloat16), bits, 64)


def to_np(q: O.QW) -> R.QArrays:
    return R.QArrays(np.array(q.weight), np.array(q.scales.astype(mx.float32)),
                     np.array(q.biases.astype(mx.float32)), q.bits, q.group_size)


def case(rng, E, F, d, T, bits, core=0.6):
    own = rng.integers(1, T, size=(E, F)).astype(np.int32)
    own[rng.random((E, F)) < core] = 0
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    g, u, dn = quantized(rng, E, F, d, bits), quantized(rng, E, F, d, bits), quantized(rng, E, d, F, bits)
    return own, lb, g, u, dn


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 4, 37])
def test_dense_exactness_and_reads(backend, bits, rows):
    rng = np.random.default_rng(rows * 10 + bits)
    F, d, T = 512, 256, 6
    own, lb, g, u, dn = case(rng, 1, F, d, T, bits)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    ident = O.DeviceBlocks.from_layout(L.identity_layer(1, F, group_size=64, chunk=32, tiles=T))
    x = mx.array(rng.standard_normal((rows, d)).astype(np.float32)).astype(mx.bfloat16)
    ones = mx.ones((rows, T), mx.uint8)
    y_id, _ = O.block_mlp(x, g, u, dn, ident, ones, backend_name=backend)
    y_pk, _ = O.block_mlp(x, gp, up, dn, blocks, ones, backend_name=backend)
    assert mx.array_equal(y_id, y_pk).item(), "packed all-open != unpacked"
    topen_np = (rng.random((rows, T)) < 0.35).astype(np.uint8)
    topen_np[:, 0] = 1
    topen = mx.array(topen_np)
    y_sk, c_sk = O.block_mlp(x, gp, up, dn, blocks, topen, count=True, backend_name=backend)
    y_ms, _ = O.block_mlp(x, gp, up, dn, blocks, topen, skip=False, backend_name=backend)
    assert mx.array_equal(y_sk, y_ms).item(), "skipped != masked"
    ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
    got = np.array(y_sk.astype(mx.float32))
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    per = O.tile_rows(backend, rows, "dense")
    pred = L.predicted_reads(lb, np.array(O.dense_worktiles(rows, per)).reshape(-1, 4), topen_np, rmap=None,
                             per_row=True, out_cols=O.down_cols(backend))
    assert O.counter_reads(c_sk) == pred
    # row invariance: each row alone gives its bits in the batch
    for r in (0, rows - 1):
        y1, _ = O.block_mlp(x[r:r + 1], gp, up, dn, blocks, topen[r:r + 1], backend_name=backend)
        assert mx.array_equal(y1[0], y_sk[r]).item()


@pytest.mark.parametrize("backend", _backends())
def test_closed_blocks_are_not_read_poison(backend):
    rng = np.random.default_rng(11)
    F, d, T = 512, 256, 4
    own, lb, g, u, dn = case(rng, 1, F, d, T, 4, core=0.5)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    topen_np = np.array([[1, 0, 1, 0]], np.uint8)
    topen = mx.array(topen_np)
    x = mx.array(rng.standard_normal((1, d)).astype(np.float32)).astype(mx.bfloat16)
    clean, _ = O.block_mlp(x, gp, up, dn, blocks, topen, backend_name=backend)
    # poison every closed chunk's gate/up scales and every closed down group's scales
    closed_rows = np.zeros(F, bool)
    for off, ln, t in lb.chunks:
        if not topen_np[0, t]:
            closed_rows[off:off + ln] = True
    nan = mx.array(np.float32(np.nan)).astype(mx.bfloat16)
    rows_mask = mx.array(closed_rows)[None, :, None]
    gs = O.QW(gp.weight, mx.where(rows_mask, nan, gp.scales), gp.biases, 4, 64)
    us = O.QW(up.weight, mx.where(rows_mask, nan, up.scales), up.biases, 4, 64)
    G = lb.groups
    closed_groups = np.array([not any(topen_np[0, t] for t in lb.gsig[lb.gsig_ptr[i]:lb.gsig_ptr[i + 1]])
                              for i in range(G)])
    assert closed_rows.any() and closed_groups.any(), "the case must have closed blocks"
    gm = mx.array(closed_groups)[None, None, :]
    ds = O.QW(dn.weight, mx.where(gm, nan, dn.scales), dn.biases, 4, 64)
    poisoned, _ = O.block_mlp(x, gs, us, ds, blocks, topen, backend_name=backend)
    assert mx.array_equal(clean, poisoned).item()
    masked, _ = O.block_mlp(x, gs, us, ds, blocks, topen, skip=False, backend_name=backend)
    assert np.isnan(np.array(masked.astype(mx.float32))).any(), "the masked arm reads them (control)"


@pytest.mark.parametrize("backend", _backends())
def test_short_chunks_match_reference(backend):
    rng = np.random.default_rng(14)
    F, d = 128, 256
    own = np.zeros(F, np.int32)
    own[:1] = 1                     # tile 1: 1 neuron, tile 2: 7, tile 3: 31, the core the rest
    own[1:8] = 2
    own[8:39] = 3
    rng.shuffle(own)
    lb = L.plan_layer(own, tiles=4, group_size=64, chunk=32)
    assert sorted(int(n) for _, n, t in lb.chunks if t) == [1, 7, 31]
    g, u, dn = quantized(rng, 1, F, d, 4), quantized(rng, 1, F, d, 4), quantized(rng, 1, d, F, 4)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.array(rng.standard_normal((3, d)).astype(np.float32)).astype(mx.bfloat16)
    for topen_np in (np.ones((3, 4), np.uint8), np.array([[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0]], np.uint8)):
        y, c = O.block_mlp(x, gp, up, dn, blocks, mx.array(topen_np), count=True, backend_name=backend)
        ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
        got = np.array(y.astype(mx.float32))
        assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
        pred = L.predicted_reads(lb, np.array(O.dense_worktiles(3, O.tile_rows(backend, 3, "dense"))).reshape(-1, 4),
                                 topen_np, rmap=None, per_row=True, out_cols=O.down_cols(backend))
        assert O.counter_reads(c) == pred


@pytest.mark.parametrize("backend", _backends())
def test_moe_sorted_unsorted_and_empty_expert(backend):
    rng = np.random.default_rng(12)
    E, F, d, T, k = 6, 640, 256, 5, 3
    own, lb, g, u, dn = case(rng, E, F, d, T, 4)
    own[2] = rng.integers(1, T, size=F)                 # expert 2: no core
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    tokens = 30
    x = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    inds_np = np.stack([rng.choice(E, k, replace=False) for _ in range(tokens)]).astype(np.int32)
    inds_np[0] = [2, 0, 1]
    inds = mx.array(inds_np)
    topen_np = (rng.random((tokens, T)) < 0.4).astype(np.uint8)
    topen_np[:, 0] = 1
    topen_np[0, 1:] = 0                                 # token 0: core only, so expert 2 is empty
    topen = mx.array(topen_np)
    ys, cs = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=True, count=True, backend_name=backend)
    yu, cu = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=False, count=True, backend_name=backend)
    assert mx.array_equal(ys, yu).item(), "sorted != unsorted routes"
    assert not np.array(ys[0, 0].astype(mx.float32)).any()
    ym, _ = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=True, skip=False, backend_name=backend)
    assert mx.array_equal(ys, ym).item()
    flat = inds_np.reshape(-1)
    ref = R.masked_reference(np.repeat(np.array(x.astype(mx.float32)), k, 0), to_np(g), to_np(u), to_np(dn),
                             own, topen_np, experts=flat, tokens=np.arange(tokens * k) // k)
    got = np.array(ys.astype(mx.float32)).reshape(tokens * k, d)
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    pred = L.predicted_reads(lb, np.array(O.pair_worktiles(inds)[0]).reshape(-1, 4), topen_np,
                             rmap=np.arange(tokens * k) // k, per_row=True, out_cols=O.down_cols(backend))
    assert O.counter_reads(cu) == pred


def test_tile_gates_kernel():
    g = L.TileGraph(["sci", "sci.phys", "lang", "code"],
                    ["core", "box:sci", "box:sci.phys", "box:lang", "pair:lang__code", "joint"],
                    [(), (0,), (1,), (2,), (2, 3), (1, 2, 3)], parent=[-1, 0, -1, -1])
    rng = np.random.default_rng(13)
    d = 256
    pw = rng.standard_normal((4, d)).astype(np.float32)
    pb = rng.standard_normal(4).astype(np.float32)
    tables = O.GateTables.from_graph(g, pw, pb)
    h = mx.array(rng.standard_normal((9, d)).astype(np.float32)).astype(mx.bfloat16)
    topen, un = O.tile_gates(h, tables, union=True)
    logit = np.array(h.astype(mx.float32)).astype(np.float64) @ pw.T.astype(np.float64) + pb
    clear = np.abs(logit) > 1e-3                         # away from the sign boundary
    raw = logit > 0
    ref = g.tile_open(raw)
    got = np.array(topen)
    rows = clear.all(1)
    assert np.array_equal(got[rows], ref[rows])
    assert np.array_equal(np.array(un).astype(bool), got.any(0))
    box = mx.array(raw.astype(np.uint8))
    t2, _ = O.tile_gates(h, tables, box_open=box)
    assert np.array_equal(np.array(t2), ref)
