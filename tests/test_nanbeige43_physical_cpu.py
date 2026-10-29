"""Synthetic four-row/COW contracts; no physical GPU or speed qualification."""

import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters import nanbeige43_kernels as kernels
from mlx2.adapters import nanbeige43_physical as physical


@pytest.fixture(autouse=True)
def receipt(record_property):
    mx.set_default_device(mx.cpu)
    root = Path(__file__).resolve().parents[1]
    record_property(
        "source_revision",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    )
    record_property("working_source_sha256", physical.source_identity())


def q8(record_property):
    mx.random.seed(4276)
    module = nn.QuantizedLinear(128, 68, bias=False, bits=8, group_size=64)
    module.set_dtype(mx.bfloat16)
    mx.eval(module.parameters())
    digest = hashlib.sha256()
    for name in ("weight", "scales", "biases"):
        value = module[name]
        if value.dtype == mx.bfloat16:
            value = value.view(mx.uint16)
        digest.update(np.asarray(value).tobytes())
    record_property(
        "checkpoint", "synthetic-nanbeige-four-row-q8g64@sha256:" + digest.hexdigest()
    )
    return module


def rope_fixture(record_property, rows=4):
    mx.random.seed(4276)
    q = mx.random.normal((1, 56, rows, 128)).astype(mx.bfloat16)
    k, v = (mx.random.normal((1, 8, rows, 128)).astype(mx.bfloat16) for _ in range(2))
    keys, values = (
        mx.random.normal((1, 8, 16, 128)).astype(mx.bfloat16) for _ in range(2)
    )
    positions = mx.array([7, 8, 8, 9][:rows], mx.int32)
    record_property(
        "checkpoint", "synthetic-nanbeige-physical-rope@seed4276-d128-q56-kv8-v1"
    )
    return q, k, v, keys, values, positions


@pytest.mark.parametrize("key", ["q8_four_row", "rope_kv_fused"])
def test_flags_are_default_off_and_strict_boolean(key, record_property):
    record_property("checkpoint", "synthetic-nanbeige-physical-policy@v1")
    assert kernels.validate_kernel_policy()[key] is False
    for bad in (1, None, "true"):
        with pytest.raises(ValueError):
            kernels.validate_kernel_policy({key: bad})
    assert (
        kernels.kernel_contract({key: True})["physical_kernel_source_sha256"]
        == physical.source_identity()
    )


def test_four_rows_reference_and_tail_rejection(record_property):
    module = q8(record_property)
    x = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
    expected = module(x)
    actual = physical.q8_four_row(x, module)
    mx.eval(expected, actual)
    assert bool(mx.all(expected == actual).item())
    assert physical.four_row_eligible(x, module)
    assert not physical.four_row_eligible(x[:, :3], module)
    with pytest.raises(ValueError):
        physical.q8_four_row(x[:, :3], module)


def test_four_row_project_hook_is_not_global_crossover(record_property):
    module = q8(record_property)
    policy = kernels.validate_kernel_policy({"q8_four_row": True})
    counts = {"reference_projection_calls": 0}
    x = mx.random.normal((4, 128)).astype(mx.bfloat16)
    output = kernels.project(module, x, policy, counts)
    mx.eval(output)
    assert counts["four_row_calls"] == 1 and counts["reference_projection_calls"] == 0
    output = kernels.project(module, x[:3], policy, counts)
    mx.eval(output)
    assert counts["four_row_calls"] == 1 and counts["reference_projection_calls"] == 1
    assert policy["projection_min_rows"] == 0


def test_four_row_word_load_precedes_four_accumulators(record_property):
    record_property(
        "checkpoint",
        "synthetic-nanbeige-four-row-shader@"
        + hashlib.sha256(physical._FOUR_ROW_SOURCE.encode()).hexdigest(),
    )
    source = physical._FOUR_ROW_SOURCE
    assert source.count("const uint word = W[") == 1
    assert source.index("const uint word = W[") < source.index(
        "for (uint row = 0; row < 4; ++row)"
    )
    assert "float acc[4]" in source and "first / 64" in source


def test_physical_rope_preserves_cow_prefix_suffix_and_branches(record_property):
    q, k, v, keys, values, positions = rope_fixture(record_property)
    qr, newkeys, newvalues, end = physical.fused_rope_append_arrays(
        q, k, v, keys, values, 7, positions
    )
    assert end == 11
    expectedq = physical._reference_rotate(q, positions, 120000000)
    expectedk = physical._reference_rotate(k, positions, 120000000)
    mx.eval(qr, newkeys, newvalues, expectedq, expectedk)
    assert bool(mx.all(qr == expectedq).item())
    assert bool(mx.all(newkeys[..., :7, :] == keys[..., :7, :]).item())
    assert bool(mx.all(newkeys[..., 11:, :] == keys[..., 11:, :]).item())
    assert bool(mx.all(newvalues[..., :7, :] == values[..., :7, :]).item())
    assert bool(mx.all(newvalues[..., 11:, :] == values[..., 11:, :]).item())
    assert bool(mx.all(newkeys[..., 7:11, :] == expectedk).item())
    assert bool(mx.all(newvalues[..., 7:11, :] == v).item())
    # A second logical owner borrows the same immutable original buffers;
    # committing its different successor must not mutate this arm or old state.
    other = physical.fused_rope_append_arrays(q, k, v, keys, values, 5, positions - 2)
    mx.eval(other[:3])
    assert other[-1] == 9
    assert bool(mx.all(newkeys[..., 7:11, :] == expectedk).item())
    assert newkeys is not keys and other[1] is not newkeys


def test_rope_refuses_capacity_and_wrong_positions_without_input_mutation(
    record_property,
):
    q, k, v, keys, values, positions = rope_fixture(record_property)
    beforekeys = keys
    beforevalues = values
    for offset, pos in (
        (14, positions),
        (7, positions.astype(mx.float32)),
        (7, positions[:3]),
    ):
        assert (
            physical.fused_rope_append_arrays(q, k, v, keys, values, offset, pos)
            is None
        )
    assert keys is beforekeys and values is beforevalues
    assert (
        physical.fused_rope_append_arrays(
            q, k, v, keys, values, 7, positions, base=10000
        )
        is None
    )


def test_physical_rope_source_never_writes_inputs(record_property):
    record_property(
        "checkpoint",
        "synthetic-nanbeige-physical-rope-shader@"
        + hashlib.sha256(physical._ROPE_APPEND_SOURCE.encode()).hexdigest(),
    )
    source = physical._ROPE_APPEND_SOURCE
    assert "OLDK[j] =" not in source and "OLDV[j] =" not in source
    assert "KOUT[j] = OLDK[j]" in source and "VOUT[j] = OLDV[j]" in source
    assert "P[row]" in source


@pytest.mark.parametrize("segmented", [False, True])
def test_actual_packed_lane_hook_preserves_owner_rollback(segmented, record_property):
    from mlx2.adapters.nanbeige43_tree import (
        _LaneKV,
        _PackedRope,
        _Plan,
        _SegmentedLaneKV,
    )
    from mlx2.runtime.models.cache import KVCache

    q, k, v, keys, values, _ = rope_fixture(record_property)
    plan = _Plan([[-1, 0, 0, 1], [-1, 0, 0, 1]], [7, 5], segmented)
    rope = _PackedRope(nn.RoPE(128, base=120000000, traditional=False), plan)
    owners = []
    for start in (7, 5):
        item = KVCache()
        item.keys = mx.stop_gradient(keys)
        item.values = mx.stop_gradient(values)
        item.offset = start
        owners.append(item)
    snapshots = [(o.keys, o.values, o.offset) for o in owners]
    cls = _SegmentedLaneKV if segmented else _LaneKV
    cache = cls(owners, plan, rope_kv_fused=True)
    qq, kk, vv = mx.concatenate([q, q]), mx.concatenate([k, k]), mx.concatenate([v, v])
    result = cache.try_fused_rope_append(qq, kk, vv, rope)
    mx.eval(result[0], [o.state for o in owners])
    assert bool(mx.all(result[0] == rope(qq)).item())
    assert [o.offset for o in owners] == [11, 9]
    if segmented:
        assert result[1:] == (None, None)
        assert [p[0].shape[2] for p in cache._segments] == [11, 9]
    else:
        assert result[1].shape == (2, 8, 11, 128)
    for owner, (oldk, oldv, start) in zip(owners, snapshots, strict=True):
        assert bool(mx.all(owner.keys[..., :start, :] == oldk[..., :start, :]).item())
        owner.keys, owner.values, owner.offset = oldk, oldv, start
    assert [o.offset for o in owners] == [7, 5]


def test_packed_lane_late_refusal_leaves_all_owners_unchanged(record_property):
    from mlx2.adapters.nanbeige43_tree import _LaneKV, _PackedRope, _Plan
    from mlx2.runtime.models.cache import KVCache

    q, k, v, keys, values, _ = rope_fixture(record_property)
    plan = _Plan([[-1, 0, 0, 1], [-1, 0, 0, 1]], [7, 5], False)
    owners = []
    for start, capacity in [(7, 16), (5, 8)]:
        item = KVCache()
        item.keys = mx.stop_gradient(keys[..., :capacity, :])
        item.values = mx.stop_gradient(values[..., :capacity, :])
        item.offset = start
        owners.append(item)
    snapshots = [(o.keys, o.values, o.offset) for o in owners]
    cache = _LaneKV(owners, plan, rope_kv_fused=True)
    rope = _PackedRope(nn.RoPE(128, base=120000000, traditional=False), plan)
    result = cache.try_fused_rope_append(
        mx.concatenate([q, q]), mx.concatenate([k, k]), mx.concatenate([v, v]), rope
    )
    assert result is None
    for owner, (oldk, oldv, start) in zip(owners, snapshots, strict=True):
        assert owner.keys is oldk and owner.values is oldv and owner.offset == start


def test_physical_rope_offset_is_runtime_operand_not_pipeline_template(
    monkeypatch, record_property
):
    q, k, v, keys, values, positions = rope_fixture(record_property)
    dispatches = []

    def dispatch(**kwargs):
        dispatches.append(kwargs)
        # Dispatch-contract stub only, not an arithmetic or GPU execution oracle.
        return [kwargs["inputs"][0], kwargs["inputs"][3], kwargs["inputs"][4]]

    monkeypatch.setattr(physical, "_rope_append_kernel", lambda: dispatch)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    for offset in (7, 8):
        result = physical.fused_rope_append_arrays(
            q, k, v, keys, values, offset, positions + (offset - 7)
        )
        assert result[-1] == offset + 4
    templates = [dict(row["template"]) for row in dispatches]
    assert templates[0] == templates[1]
    assert "OFFSET" not in templates[0]
    assert [row["inputs"][-1].tolist() for row in dispatches] == [[7], [8]]
    assert all(row["inputs"][-1].dtype == mx.int32 for row in dispatches)
    assert "const uint offset = uint(OFF[0])" in physical._ROPE_APPEND_SOURCE
