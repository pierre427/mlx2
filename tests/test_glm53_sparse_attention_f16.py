# SPDX-License-Identifier: MIT
"""GLM-5.3 fp16-operand gathered attention (``gathered_f16_from_<n>``) Metal gates.

GPU only: set MLX2_RUN_GPU_TESTS=1. Synthetic operands; the checkpoint is not
loaded. The kernel is an fp16-operand law (half q, half staged tiles, half P;
fp32 scores, statistics, accumulators), so it is gated against float64 on
the half-rounded operands, never against the fp32 kernel's bits.
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
    ids = np.full((rows, width), -1, np.int32)
    for r in range(rows):
        pos = total - 1
        pools = np.sort(rng.choice(pos // 4, size=(width - 3) // 4, replace=False))
        row = (pools[:, None] * 4 + np.arange(4)).reshape(-1)
        ids[r, : len(row)] = row
        ids[r, len(row):] = np.arange(pos - 2, pos + 1)
    return ids


def run(q, base, tails, ids, meta, config=(v2.DEFAULT_SPLIT, v2.DEFAULT_DIM_SPLIT), half=True):
    out = v2.launch(mx.array(q), mx.array(base), mx.array(tails), mx.array(ids), mx.array(meta),
                    split=config[0], dim_split=config[1], half=half)
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("config", CONFIGS)
def test_f16_kernel_is_within_the_bound_on_half_rounded_operands_and_the_gate_passes(config):
    for name, q, base, tails, ids, meta in v2.gate_cases():
        got = run(q, base, tails, ids, meta, config)
        q16, base16, tails16 = v2.half_rounded(q, base, tails)
        want = v2.oracle(q16, base16, tails16, ids, meta)
        errors = v2.oracle_errors(got, want)
        # Against the unrounded operands too, for the record: the operand
        # rounding dominates and is the law's own class.
        unrounded = v2.oracle_errors(got, v2.oracle(q, base, tails, ids, meta))
        print("f16_bound", name, config, "kernel_vs_half_rounded", errors, "kernel_vs_fp32_operands", unrounded)
        assert errors is not None, name
        assert errors[0] <= v2.F16_ABS_BOUND and errors[1] <= v2.F16_REL_RMS_BOUND, (name, config, errors)
    assert v2.self_check(split=config[0], dim_split=config[1], half=True) is True
    key = f"split{config[0]}_dsplit{config[1]}_f16"
    assert all(case["passed"] for case in v2.GATE_REPORT[key].values())
    print("gate_report", key, v2.GATE_REPORT[key])


def test_f16_matches_the_fp32_kernel_to_fp16_operand_precision():
    """The two kernels share rows, ids and masks: their outputs differ only by
    the half rounding of q, keys, values and P (relative RMS well under 1e-2
    on unit-scale rows), and NaN placement is identical."""
    rng = np.random.default_rng(41)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 4)
    ids[2, :] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 4, np.int32)
    q = rng.normal(size=(4, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    half = run(q, base, tails, ids, meta)
    full = run(q, base, tails, ids, meta, half=False)
    assert np.array_equal(np.isnan(half), np.isnan(full)) and np.isnan(half[2]).all()
    finite = ~np.isnan(full)
    rel = np.sqrt(np.mean((half[finite] - full[finite]) ** 2)) / np.sqrt(np.mean(full[finite] ** 2))
    print("f16_vs_fp32_kernel_rel_rms", rel)
    assert 0 < rel < 1e-2


@pytest.mark.parametrize("config", CONFIGS)
def test_f16_padded_and_empty_splits_and_the_nan_contract(config):
    rng = np.random.default_rng(42)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    split = config[0]
    ids = selection(rng, total, 4)
    ids[0, split : 2 * split] = -1
    ids[1, 5 * split + 3 : 6 * split] = -1
    ids[2, :] = -1
    ids[3, 3:] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 4, np.int32)
    q = rng.normal(size=(4, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    got = run(q, base, tails, ids, meta, config)
    want = v2.oracle(*v2.half_rounded(q, base, tails), ids, meta)
    assert np.isnan(got[2]).all() and np.isfinite(got[[0, 1, 3]]).all()
    errors = v2.oracle_errors(got, want)
    assert errors is not None and errors[0] <= v2.F16_ABS_BOUND and errors[1] <= v2.F16_REL_RMS_BOUND, errors


def test_f16_padding_is_inert_when_the_clipped_row_is_poisoned():
    rng = np.random.default_rng(43)
    total = 2600
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 3)
    ids[ids == 0] = 1
    ids[0, 100:300] = -1
    ids[1, -40:] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 3, np.int32)
    q = rng.normal(size=(3, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    clean = run(q, base, tails, ids, meta)
    poisoned = base.copy()
    poisoned[0] = np.nan
    assert same_bits(clean, run(q, poisoned, tails, ids, meta))


@pytest.mark.parametrize("rows", [1, 2, 8, 64, 128])
def test_f16_rows_in_one_launch_equal_single_row_launches_bitwise(rows):
    rng = np.random.default_rng(200 + rows)
    t_base = 2600
    base = rng.normal(size=(t_base, 512)).astype(np.float32)
    depths = [int(d) for d in rng.integers(0, 6, size=rows)]
    tails = rng.normal(size=(max(1, sum(depths)), 512)).astype(np.float32)
    ids = np.full((rows, W), -1, np.int32)
    meta = np.zeros((rows, 4), np.int32)
    offset = 0
    for r, d in enumerate(depths):
        ids[r] = selection(rng, t_base + d, 1)[0]
        meta[r] = [t_base, d, offset, t_base + d - 1]
        offset += d
    q = rng.normal(size=(rows, 64, 512)).astype(np.float32)
    together = run(q, base, tails, ids, meta)
    for r in range(rows):
        single = run(q[r : r + 1], base, tails, ids[r : r + 1], meta[r : r + 1])
        assert same_bits(together[r : r + 1], single), r


def test_f16_prefill_blocks_agree_bitwise_and_cast_once():
    """block = 4 == block = L through sparse_prefill(policy="fused_f16"); the
    chunk's source is cast to half once and every block reads the same copy."""
    rng = np.random.default_rng(44)
    total, length = 4100, 70
    latent = mx.array(rng.normal(size=(1, 1, total, 512)).astype(np.float32))
    q = mx.array(rng.normal(size=(1, 64, length, 512)).astype(np.float32))
    ids = np.stack([selection(rng, total - length + r + 1, 1)[0] for r in range(length)])
    ids = mx.array(ids[None])
    counters = {}
    small = sa.sparse_prefill(q, latent, ids, block=4, policy="fused_f16", counters=counters)
    whole = sa.sparse_prefill(q, latent, ids, block=length, policy="fused_f16")
    mx.eval(small, whole)
    assert same_bits(small, whole)
    assert counters["sparse_prefill_fused_f16_blocks"] == 18 and counters["sparse_attention_fused_f16_rows"] == length
    assert "sparse_attention_fused_v2" not in counters
    # The fp32 law on the same rows is a different law (operands not rounded).
    fp32 = sa.sparse_prefill(q, latent, ids, block=length, policy="fused_v2")
    mx.eval(fp32)
    assert not same_bits(whole, fp32)
    rel = float(mx.sqrt(mx.mean((whole - fp32) ** 2) / mx.mean(fp32**2)).item())
    print("prefill_f16_vs_fp32_rel_rms", rel)
    assert rel < 1e-2


@pytest.mark.parametrize("split", v2.SPLITS_SUPPORTED)
def test_f16_dim_split_does_not_change_the_bits(split):
    rng = np.random.default_rng(45)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 5)
    ids[4, 500:900] = -1
    meta = np.array([[total, 0, 0, total - 1]] * 5, np.int32)
    q = rng.normal(size=(5, 64, 512)).astype(np.float32)
    tails = np.zeros((1, 512), np.float32)
    assert same_bits(run(q, base, tails, ids, meta, (split, 2)), run(q, base, tails, ids, meta, (split, 4)))


@pytest.mark.parametrize("poison", [np.nan, np.inf])
def test_f16_poisoned_head_stays_in_its_head_across_inner_splits(poison):
    """A NaN (or +inf: exp(inf - inf) is NaN too) q value for one head, seen
    by every inner split, gives that head a NaN output and leaves the other
    63 heads finite and within the bound; a diag(a) MMA rescale over an O that
    held the NaN would have spread 0 * NaN to its seven group mates."""
    rng = np.random.default_rng(46)
    total = 3000
    base = rng.normal(size=(total, 512)).astype(np.float32)
    ids = selection(rng, total, 2)
    meta = np.array([[total, 0, 0, total - 1]] * 2, np.int32)
    q = rng.normal(size=(2, 64, 512)).astype(np.float32)
    q[0, 13, 7] = poison  # head 13 of row 0 (group 1, row 5 of the fragment)
    tails = np.zeros((1, 512), np.float32)
    got = run(q, base, tails, ids, meta)
    assert np.isnan(got[0, 13]).all()
    others = np.delete(got[0], 13, axis=0)
    assert np.isfinite(others).all() and np.isfinite(got[1]).all()
    q16, base16, tails16 = v2.half_rounded(q, base, tails)
    want = v2.oracle(q16, base16, tails16, ids, meta)
    assert np.isnan(want[0, 13]).all()
    errors = v2.oracle_errors(np.delete(got, 13, axis=1), np.delete(want, 13, axis=1))
    assert errors is not None and errors[0] <= v2.F16_ABS_BOUND and errors[1] <= v2.F16_REL_RMS_BOUND, errors


def test_f16_nan_flagging_is_inert_on_finite_rows_bitwise():
    """The per-head NaN flags (BAD, the NaN-dropping max and the (NaN, 1)
    partial) are the only thing added to the 127cd63a6 kernel the receipts
    f16-breadth-16k-32k and f16-ttft-16k-32k ran: with them removed by source
    substitution the partials are bitwise the same on finite rows."""
    spec = v2.kernel_spec()["partial_flash"]
    source = spec["source"]
    plain = source
    for old, new in (("ms = max(ms, poison ? -INFINITY : x);", "ms = max(ms, x);"),
                     ("(valid && !empty && !poison)", "(valid && !empty)"),
                     ("partial_m[slot] = BAD[tid] ? NAN : (empty ? -INFINITY : M[tid]);",
                      "partial_m[slot] = empty ? -INFINITY : M[tid];"),
                     ("partial_l[slot] = BAD[tid] ? 1.0f : (empty ? 0.0f : L[tid]);",
                      "partial_l[slot] = empty ? 0.0f : L[tid];"),
                     ("            BAD[h] |= bad;\n", "")):
        assert plain.count(old) == 1, old
        plain = plain.replace(old, new)
    legacy = mx.fast.metal_kernel(**{**spec, "name": "test_flash_without_nan_flags", "source": plain},
                                  ensure_row_contiguous=True)
    rng = np.random.default_rng(47)
    total, rows = 3000, 6
    base = mx.array(rng.normal(size=(total, 512)).astype(np.float32)).astype(mx.float16)
    ids = selection(rng, total, rows)
    ids[3, 200:1500] = -1
    ids[4, :] = -1  # an empty row: (-inf, 0) slots in both
    ids_mx = mx.array(ids)
    meta = mx.array([[total, 0, 0, total - 1]] * rows, dtype=mx.int32)
    q = mx.array(rng.normal(size=(rows, 64, 512)).astype(np.float32)).astype(mx.float16)
    tails = mx.zeros((1, 512), mx.float16)
    plan = v2.launch_plan(rows, W, kind="flash_f16")
    launch = dict(template=plan["template"], grid=plan["partial"]["grid"], threadgroup=plan["partial"]["threadgroup"],
                  output_shapes=plan["partial"]["output_shapes"], output_dtypes=[mx.float32] * 3)
    m, l, o = legacy(inputs=[q, base, tails, ids_mx, meta], **launch)
    current = v2._kernel("partial_flash")(inputs=[q, base, tails, ids_mx, meta], **launch)
    mx.eval(m, l, o, current)
    assert same_bits(m, current[0]) and same_bits(l, current[1]) and same_bits(o, current[2])
