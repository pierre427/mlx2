# SPDX-License-Identifier: MIT
"""GLM-5.3 fused_v2 Metal gates (docs/design/glm53-speed-round2.md 1.1 tests (a)-(c), 2.1).

GPU only: set MLX2_RUN_GPU_TESTS=1 (run through the GPU lease, e.g.
scripts/research/glm53_sparse_attention_bench.py pytest). Synthetic operands;
the model checkpoint is not loaded.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_sparse_attention as sa
from mlx2.runtime.models import glm53_sparse_attention_v2 as v2

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_GPU_TESTS") != "1" or not mx.metal.is_available(),
    reason="Metal gate: set MLX2_RUN_GPU_TESTS=1 on a Metal host",
)
CONFIGS = [(s, d) for s in v2.SPLITS_SUPPORTED for d in v2.DIM_SPLITS_SUPPORTED]
W = sa.WIDTH


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def bits(a):
    return np.array(a).view(np.uint32)


def same_bits(a, b):
    a, b = bits(a), bits(b)
    return a.shape == b.shape and np.array_equal(a, b)


def selection(rng, total, rows, width=W):
    """Decode-shaped ids: 512 random pools x 4 rows ascending, then the 3-row tail."""
    ids = np.full((rows, width), -1, np.int32)
    for r in range(rows):
        pos = total - 1
        pools = np.sort(rng.choice(pos // 4, size=(width - 3) // 4, replace=False))
        row = (pools[:, None] * 4 + np.arange(4)).reshape(-1)
        ids[r, : len(row)] = row
        ids[r, len(row):] = np.arange(pos - 2, pos + 1)
    return ids


def run(q, base, tails, ids, meta, config=(v2.DEFAULT_SPLIT, v2.DEFAULT_DIM_SPLIT)):
    out = v2.launch(mx.array(q), mx.array(base), mx.array(tails), mx.array(ids), mx.array(meta),
                    split=config[0], dim_split=config[1])
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("config", CONFIGS)
def test_kernel_is_within_the_float64_bound_and_the_gate_passes(config):
    for name, q, base, tails, ids, meta in v2.gate_cases():
        want = v2.oracle(q, base, tails, ids, meta)
        errors = v2.oracle_errors(run(q, base, tails, ids, meta, config), want)
        ops = v2.oracle_errors(v2.ops_reference(q, base, tails, ids, meta), want)
        print("float64_bound", name, config, "kernel", errors, "ops", ops)
        assert errors is not None, name
        assert errors[0] <= 1e-5 and errors[1] <= max(1e-6, ops[1]), (name, config, errors, ops)
    assert v2.self_check(split=config[0], dim_split=config[1]) is True
    print("gate_report", v2.GATE_REPORT)


def test_ops_reference_is_bounded_separately():
    """The gathered_v1 ops path against float64 on the same cases (recorded,
    never read as a kernel failure: 5385578a2's receipt failed on the ops side)."""
    report = {name: v2.oracle_errors(v2.ops_reference(q, base, tails, ids, meta), v2.oracle(q, base, tails, ids, meta))
              for name, q, base, tails, ids, meta in v2.gate_cases()}
    print("ops_reference_vs_float64", report)
    assert all(e is not None and e[0] <= 1e-4 for e in report.values())


@pytest.mark.parametrize("config", CONFIGS)
def test_padded_and_empty_splits_and_the_nan_contract(config):
    rng = np.random.default_rng(31)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    split = config[0]
    ids = selection(rng, total, 4)
    ids[0, split : 2 * split] = -1                  # one whole split without a valid key
    ids[1, 5 * split + 3 : 6 * split] = -1          # a split with 3 valid keys
    ids[2, :] = -1                                  # empty selection -> NaN row
    ids[3, 3:] = -1                                 # only 3 valid keys in the whole row
    meta = np.array([[total, 0, 0, total - 1]] * 4, np.int32)
    q = rng.normal(size=(4, 64, 512)).astype(np.float32)
    got = run(q, base, np.zeros((1, 512), np.float32), ids, meta, config)
    want = v2.oracle(q, base, np.zeros((1, 512), np.float32), ids, meta)
    assert np.isnan(got[2]).all() and np.isfinite(got[[0, 1, 3]]).all()
    errors = v2.oracle_errors(got, want)
    ops = v2.oracle_errors(v2.ops_reference(q, base, np.zeros((1, 512), np.float32), ids, meta), want)
    assert errors is not None and errors[0] <= 1e-5 and errors[1] <= max(1e-6, ops[1]), (errors, ops)


def test_padding_is_inert_when_the_clipped_row_is_poisoned():
    """-1 ids are zero-filled, never read: poisoning latent row 0 (where a clip
    would point) with NaN between two launches leaves every bit unchanged."""
    rng = np.random.default_rng(32)
    total = 2600
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 3)
    ids[ids == 0] = 1  # no valid id names row 0
    ids[0, 100:300] = -1
    ids[1, -40:] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 3, np.int32)
    q = rng.normal(size=(3, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    clean = run(q, base, tails, ids, meta)
    poisoned = base.copy()
    poisoned[0] = np.nan
    assert same_bits(clean, run(q, poisoned, tails, ids, meta))


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 64, 128])
def test_rows_in_one_launch_equal_single_row_launches_bitwise(rows):
    """Mixed rows on one shared base: distinct (T0_r, d_r) tails and dense
    causal rows at pos 0, 3, 2047 share a launch; each equals its own launch."""
    rng = np.random.default_rng(100 + rows)
    t_base = 2600
    base = rng.normal(size=(t_base, 512)).astype(np.float32)
    q = rng.normal(size=(rows, 64, 512)).astype(np.float32)
    ids, meta, tails, offset = [], [], [], 0
    dense = [0, 3, 2047]
    for r in range(rows):
        kind = r % 3
        if kind == 0 and r // 3 < len(dense):
            pos = dense[r // 3]
            ids.append(v2.dense_row_ids_np([pos], W)[0])
            meta.append([pos + 1, 0, 0, pos])
            continue
        t0 = int(rng.integers(2100, t_base + 1))
        d = int(rng.integers(0, 6))
        tail = rng.normal(size=(d, 512)).astype(np.float32)
        row = selection(rng, t0 + d, 1)[0]
        if r % 5 == 4:
            row[rng.random(W) < 0.2] = -1
        ids.append(row)
        meta.append([t0, d, offset, t0 + d - 1])
        tails.append(tail)
        offset += d
    tails = np.concatenate(tails + [np.zeros((1, 512), np.float32)])
    ids, meta = np.stack(ids).astype(np.int32), np.array(meta, np.int32)
    together = run(q, base, tails, ids, meta)
    for r in range(rows):
        t0, d, off, pos = meta[r]
        alone = run(q[r : r + 1], base, np.concatenate([tails[off : off + d], np.zeros((1, 512), np.float32)]),
                    ids[r : r + 1], np.array([[t0, d, 0, pos]], np.int32))
        assert same_bits(together[r : r + 1], alone), r
    want = v2.oracle(q, base, tails, ids, meta)
    errors = v2.oracle_errors(together, want)
    ops = v2.oracle_errors(v2.ops_reference(q, base, tails, ids, meta), want)
    assert errors is not None and errors[0] <= 1e-5 and errors[1] <= max(1e-6, ops[1]), (errors, ops)


def test_prefill_block_size_and_decode_rows_agree_bitwise():
    """block = 4 == block = L == one row per decode launch (rows share nothing)."""
    rng = np.random.default_rng(33)
    total, length = 4100, 70
    latent = mx.array(rng.normal(size=(1, 1, total, 512)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, 64, length, 512)).astype(np.float32))
    ids = np.stack([selection(rng, total - length + r + 1, 1)[0] for r in range(length)])
    ids = mx.array(ids[None])
    small = sa.sparse_prefill(q, latent, ids, block=4, policy="fused_v2")
    whole = sa.sparse_prefill(q, latent, ids, block=length, policy="fused_v2")
    rows = mx.concatenate([sa.sparse_decode(q[:, :, r : r + 1], latent, ids[:, r : r + 1], policy="fused_v2")
                           for r in range(length)], axis=2)
    mx.eval(small, whole, rows)
    assert same_bits(small, whole) and same_bits(small, rows)


def test_dense_rows_cross_the_2048_boundary_in_one_law():
    """Dense causal rows (ids None) and gathered rows: a block whose rows span
    T0 + L across 2048 equals single-row launches of the same rows."""
    rng = np.random.default_rng(34)
    latent = mx.array(rng.normal(size=(1, 1, 2048, 512)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, 64, 6, 512)).astype(np.float32))
    whole = sa.sparse_prefill(q, latent, None, block=6, policy="fused_v2")
    rows = mx.concatenate([sa.sparse_decode(q[:, :, r : r + 1], latent[:, :, : 2043 + r], None, policy="fused_v2")
                           for r in range(6)], axis=2)
    mx.eval(whole, rows)
    assert same_bits(whole, rows)


@pytest.mark.parametrize("split", v2.SPLITS_SUPPORTED)
def test_dim_split_does_not_change_the_bits(split):
    """Scores are one MMA chain over dims 0..511 and the statistics use the
    first 512 threads whatever DIM_SPLIT is, so DIM_SPLIT 2 and 4 agree bitwise."""
    rng = np.random.default_rng(35)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 5)
    ids[4, 500:900] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 5, np.int32)
    q = rng.normal(size=(5, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    assert same_bits(run(q, base, tails, ids, meta, (split, 2)), run(q, base, tails, ids, meta, (split, 4)))


def test_rows_attention_tails_v1_is_one_launch_equal_to_its_rows():
    rng = np.random.default_rng(36)
    latent = mx.array(rng.normal(size=(1, 1, 3010, 512)).astype(np.float32))
    base = latent[0, 0, :3000]
    origin = object()
    depths = (0, 1, 4, 10)
    views = [sa.TreeLatentView(base, latent[0, 0, 3000 : 3000 + d], 3000, origin=origin) for d in depths]
    ids = [mx.array(selection(rng, 3000 + max(d, 1), 1)[None]) for d in depths]
    q_rows = mx.array(rng.normal(size=(len(depths), 64, 1, 512)).astype(np.float32))
    counters = {}
    out = sa.rows_attention(q_rows, list(zip(views, ids, strict=True)), policy="fused_v2", counters=counters)
    singles = [sa.sparse_decode(q_rows[r : r + 1], views[r], ids[r], policy="fused_v2") for r in range(len(depths))]
    mx.eval(out, singles)
    assert counters["sparse_attention_fused_v2"] == 1
    for r, single in enumerate(singles):
        assert same_bits(out[r : r + 1], single)
