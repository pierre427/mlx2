"""Real tiny dense-Qwen3 flat execution and plain-KV lifecycle, CPU only."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_standard_xpress_serving_cpu import tiny

from mlx2.adapters.dense_qwen3_flat import DenseQwen3FlatSegmentTarget
from mlx2.runtime import flat_segmented_kv as flat_kv
from mlx2.runtime.flat_segmented_ast import (
    FlatSegmentDescriptor,
    query_flat_segment_admission,
    query_real_flat_segment_admission,
)
from mlx2.runtime.flat_segmented_kv import (
    FlatSegmentedKVRows,
    canonical_kv_state,
)
from mlx2.runtime.models.cache import KVCache, RotatingKVCache

IDENTITY = {
    "source_revision": "dcc5273a22a732a7ea145262de88a683b192129a",
    "build": f"mlx-{mx.__version__}",
    "device": "mlx-cpu",
    # SHA-256 of the canonical fixture path, seed, tied/dtype parameterization,
    # and ModelArgs manifest recorded in provenance.
    "artifact": (
        "tiny-dense-qwen3-test-fixture@sha256:"
        "8c33d8410274bbef931109847ecdcd46c2694f7d60a8a5664a064731b9b5f617"
    ),
}
EXPECTED_OWNER_SEMANTIC_SHA256 = (
    "4bcee4100b641fd1eafbe4a3578ccf939687330ddd19cc62bc42c8fb258821b4"
)
EXPECTED_STATE_FINGERPRINTS = (
    "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
)
OWNER_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)
TARGET_SOURCE_PATHS = (
    "src/mlx2/adapters/dense_qwen3_flat.py",
    "src/mlx2/runtime/models/standard_decoder.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/rope_utils.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
)
_ROOT = Path(__file__).resolve().parents[1]


def _expected_sources(paths):
    return {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _equal(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )


def _rows(model, lanes):
    result = []
    for lane in range(lanes):
        caches = model.make_cache()
        prefix = tuple((lane + position + 1) % 8 for position in range(lane % 3))
        if prefix:
            mx.eval(model(mx.array([prefix]), cache=caches))
        result.append(caches)
    return result


def _serial_outputs(model, base_rows, descriptor, tokens, capture=(0, 2)):
    outputs = []
    final = []
    for lane in range(descriptor.lane_count):
        caches = copy.deepcopy(base_rows[lane])
        lane_outputs = []
        segment = tokens[descriptor.lane_slice(lane)]
        for token in segment:
            logits, features = model.forward_with_taps(
                mx.array([[token]]), caches, capture
            )
            mx.eval(logits, features)
            lane_outputs.append((logits[0, 0], features[0, 0]))
        outputs.extend(lane_outputs)
        final.append(caches)
    return tuple(outputs), final


def _accepted_state(model, base_rows, descriptor, tokens, accepted):
    rows = copy.deepcopy(base_rows)
    for lane, count in enumerate(accepted):
        segment = tokens[descriptor.lane_slice(lane)]
        for token in segment[:count]:
            mx.eval(model(mx.array([[token]]), cache=rows[lane]))
    return rows


def _admission(owner, target, descriptor):
    return query_real_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=EXPECTED_OWNER_SEMANTIC_SHA256,
        expected_state_contract_fingerprints=EXPECTED_STATE_FINGERPRINTS,
        expected_owner_source_hashes=_expected_sources(OWNER_SOURCE_PATHS),
        expected_target_source_hashes=_expected_sources(TARGET_SOURCE_PATHS),
    )


def _pair(model, rows, descriptor, capture=(0, 2)):
    owner = FlatSegmentedKVRows(rows, identity=IDENTITY)
    target = DenseQwen3FlatSegmentTarget(
        model, capture_layers=capture, identity=IDENTITY
    )
    admission = _admission(owner, target, descriptor)
    assert admission.available, admission.decline_reasons
    assert admission.receipt()["execution_scope"] == "adapter_owned_real_cpu_contract"
    assert admission.receipt()["selected"] is False
    return owner, target, admission


@pytest.mark.parametrize(
    ("lengths", "accepted"),
    [
        ((1,), (1,)),
        ((1, 3), (1, 2)),
        ((1, 2, 3, 4), (1, 1, 2, 4)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
    ],
)
def test_real_flat_full_logits_taps_kv_and_continuation(lengths, accepted):
    model, _ = tiny(tied=False)
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    tokens = tuple((index * 3 + 1) % 8 for index in range(descriptor.logical_rows))
    rows = _rows(model, len(lengths))
    base = copy.deepcopy(rows)
    expected_outputs, _ = _serial_outputs(model, base, descriptor, tokens)
    expected_rows = _accepted_state(model, base, descriptor, tokens, accepted)
    expected_state = canonical_kv_state(expected_rows)
    owner, target, admission = _pair(model, rows, descriptor)
    before = canonical_kv_state(rows)

    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    assert canonical_kv_state(rows) == before
    execution = target.execute_flat_segments(tokens, descriptor, transaction)
    assert execution.invocation_count == 1
    assert target.invocations == 1
    assert canonical_kv_state(rows) == before
    assert len(execution.outputs) == len(expected_outputs)
    for (actual_logits, actual_features), (expected_logits, expected_features) in zip(
        execution.outputs, expected_outputs, strict=True
    ):
        _equal(actual_logits, expected_logits)
        _equal(actual_features, expected_features)

    prepared = transaction.prepare_commit(accepted)
    assert prepared.state == expected_state
    assert canonical_kv_state(rows) == before
    assert transaction.publish_commit(prepared) == expected_state
    assert canonical_kv_state(rows) == expected_state
    assert target.receipt() == {
        "route": "dense-qwen3-flat-segment-cpu-direct",
        "execution_scope": "adapter_owned_real_cpu_contract",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": True,
        "target_invocations": 1,
        "logical_rows": descriptor.logical_rows,
        "packed_matmul_authority": False,
        "projection_fidelity": "ordinary_b1s1_row_exact",
        "serving_wired": False,
    }
    for actual, expected in zip(rows, expected_rows, strict=True):
        actual_next = model(mx.array([[8]]), cache=copy.deepcopy(actual))
        expected_next = model(mx.array([[8]]), cache=copy.deepcopy(expected))
        _equal(actual_next, expected_next)


def _flat_outputs(tokens):
    model, _ = tiny(tied=True)
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 3, 1, 2))
    rows = _rows(model, 4)
    owner, target, admission = _pair(model, rows, descriptor)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    execution = target.execute_flat_segments(tokens, descriptor, transaction)
    transaction.abort()
    return execution


def test_reorder_and_neighbor_isolation_keep_each_segment_independent():
    first = (1, 2, 3, 4, 5, 6, 7, 1)
    changed = (1, 2, 7, 7, 7, 6, 4, 4)
    a = _flat_outputs(first)
    b = _flat_outputs(changed)
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 3, 1, 2))
    for actual, expected in zip(
        a.outputs[descriptor.lane_slice(0)],
        b.outputs[descriptor.lane_slice(0)],
        strict=True,
    ):
        _equal(actual[0], expected[0])
        _equal(actual[1], expected[1])
    assert not np.array_equal(
        np.asarray(a.outputs[descriptor.lane_slice(1)][0][0]),
        np.asarray(b.outputs[descriptor.lane_slice(1)][0][0]),
    )

    # Reordering lanes and their state changes only output order.
    model, _ = tiny(tied=True)
    lengths = (1, 2, 3, 4)
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    tokens = tuple((index + 2) % 8 for index in range(sum(lengths)))
    rows = _rows(model, 4)
    expected, _ = _serial_outputs(model, copy.deepcopy(rows), descriptor, tokens)
    order = (3, 1, 0, 2)
    permuted_lengths = tuple(lengths[index] for index in order)
    permuted = FlatSegmentDescriptor.from_query_lengths(permuted_lengths)
    permuted_tokens = tuple(
        token
        for lane in order
        for token in tokens[descriptor.lane_slice(lane)]
    )
    permuted_rows = [copy.deepcopy(rows[index]) for index in order]
    owner, target, admission = _pair(model, permuted_rows, permuted)
    transaction = owner.begin_flat(
        permuted,
        target_executor=target,
        admission=admission,
    )
    actual = target.execute_flat_segments(permuted_tokens, permuted, transaction)
    transaction.abort()
    expected_permuted = tuple(
        value
        for lane in order
        for value in expected[descriptor.lane_slice(lane)]
    )
    for actual_row, expected_row in zip(actual.outputs, expected_permuted, strict=True):
        _equal(actual_row[0], expected_row[0])
        _equal(actual_row[1], expected_row[1])


def test_execution_failure_and_abort_never_mutate_live_rows(monkeypatch):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 3))
    rows = _rows(model, 2)
    owner, target, admission = _pair(model, rows, descriptor)
    before = canonical_kv_state(rows)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    original = transaction.attend_flat_layer
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected after later private layer")
        return result

    monkeypatch.setattr(transaction, "attend_flat_layer", fail_after_first)
    with pytest.raises(RuntimeError, match="later private layer"):
        target.execute_flat_segments((1, 2, 3, 4, 5), descriptor, transaction)
    assert canonical_kv_state(rows) == before
    transaction.abort()
    assert canonical_kv_state(rows) == before


def test_mid_publish_failure_restores_exact_original_references_and_state(monkeypatch):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 2))
    rows = _rows(model, 2)
    owner, target, admission = _pair(model, rows, descriptor)
    before = canonical_kv_state(rows)
    references = tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in rows
        for cache in row
    )
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    target.execute_flat_segments((1, 2, 3, 4), descriptor, transaction)
    prepared = transaction.prepare_commit((1, 2))
    original = transaction._assign_cache
    calls = 0

    def fail(live, successor):
        nonlocal calls
        original(live, successor)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected mid publish")

    monkeypatch.setattr(transaction, "_assign_cache", fail)
    with pytest.raises(RuntimeError, match="mid publish"):
        transaction.publish_commit(prepared)
    assert canonical_kv_state(rows) == before
    assert references == tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in rows
        for cache in row
    )
    assert owner._active is None


def test_post_assignment_verification_failure_restores_and_closes(monkeypatch):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((2,))
    rows = _rows(model, 1)
    owner, target, admission = _pair(model, rows, descriptor)
    before = canonical_kv_state(rows)
    references = tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in rows
        for cache in row
    )
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    target.execute_flat_segments((1, 2), descriptor, transaction)
    prepared = transaction.prepare_commit((2,))
    original = flat_kv.canonical_kv_state
    calls = 0

    def fail_after_assignment(values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected post assignment verification")
        return original(values)

    monkeypatch.setattr(flat_kv, "canonical_kv_state", fail_after_assignment)
    with pytest.raises(RuntimeError, match="post assignment verification"):
        transaction.publish_commit(prepared)
    assert original(rows) == before
    assert references == tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in rows
        for cache in row
    )
    assert owner._active is None


def test_owner_membership_is_immutable_and_bound_to_admission_and_transaction():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    replacement = tuple(tuple(row) for row in _rows(model, 1))
    with pytest.raises(AttributeError):
        owner.rows = replacement

    original = owner._rows
    owner._rows = replacement
    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None

    owner._rows = original
    admission = _admission(owner, target, descriptor)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    target.execute_flat_segments((1,), descriptor, transaction)
    replacement_before = canonical_kv_state(replacement)
    owner._rows = replacement
    with pytest.raises(RuntimeError, match="membership changed"):
        transaction.prepare_commit((1,))
    assert canonical_kv_state(replacement) == replacement_before
    owner._rows = original
    transaction.abort()


def test_target_model_and_capture_schema_are_immutable_and_ticket_bound():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    with pytest.raises(AttributeError):
        target.capture_layers = (1,)
    with pytest.raises(AttributeError):
        target.model = tiny()[0]

    original_capture = target._capture_layers
    target._capture_layers = (1,)
    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None

    target._capture_layers = original_capture
    admission = _admission(owner, target, descriptor)
    original_model = target._model
    target._model = tiny()[0]
    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None
    target._model = original_model


def test_published_transaction_is_irrevocable_to_stale_abort():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((2,))
    rows = _rows(model, 1)
    owner, target, admission = _pair(model, rows, descriptor)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    target.execute_flat_segments((1, 2), descriptor, transaction)
    prepared = transaction.prepare_commit((2,))
    published = transaction.publish_commit(prepared)
    revision = owner.revision

    with pytest.raises(RuntimeError, match="irrevocable"):
        transaction.abort()
    assert canonical_kv_state(rows) == published
    assert owner.revision == revision


def test_real_begin_requires_a_live_externally_bound_admission_ticket():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    with pytest.raises(RuntimeError, match="ticket is unavailable"):
        owner.begin_flat(descriptor, target_executor=target, admission=None)
    assert owner._active is None

    other_target = DenseQwen3FlatSegmentTarget(
        model,
        capture_layers=(0, 2),
        identity=IDENTITY,
    )
    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=other_target,
            admission=admission,
        )
    assert owner._active is None

    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    transaction.abort()
    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None


def test_real_admission_ticket_binds_exact_pre_begin_state():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    mx.eval(model(mx.array([[1]]), cache=owner.rows[0]))

    with pytest.raises(RuntimeError, match="stale or unbound"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None


def test_execute_rejects_a_transaction_whose_admission_binding_was_removed():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    transaction.admission = None
    with pytest.raises(RuntimeError, match="ticket is unavailable"):
        target.execute_flat_segments((1,), descriptor, transaction)
    assert target.invocations == 0
    transaction.abort()


@pytest.mark.parametrize(
    "mutation",
    [
        "layer_count",
        "head_geometry",
        "cache_dtype",
        "position_capacity",
    ],
)
def test_owner_target_geometry_declines_before_begin(monkeypatch, mutation):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 1))
    rows = _rows(model, 2)
    if mutation == "layer_count":
        rows = [row[:-1] for row in rows]
    elif mutation == "head_geometry":
        cache = rows[1][0]
        cache.keys = mx.concatenate([cache.keys, cache.keys], axis=1)
        cache.values = mx.concatenate([cache.values, cache.values], axis=1)
    elif mutation == "cache_dtype":
        cache = rows[1][0]
        cache.keys = cache.keys.astype(mx.float16)
        cache.values = cache.values.astype(mx.float16)
    else:
        model.args.max_position_embeddings = 1
    owner = FlatSegmentedKVRows(rows, identity=IDENTITY)
    target = DenseQwen3FlatSegmentTarget(
        model, capture_layers=(0, 2), identity=IDENTITY
    )
    begins = 0

    def begin_flat(_descriptor):
        nonlocal begins
        begins += 1
        raise AssertionError("incompatible owner must not begin")

    monkeypatch.setattr(owner, "begin_flat", begin_flat)
    admission = _admission(owner, target, descriptor)
    assert admission.available is False
    assert "target_owner_compatibility_invalid" in admission.decline_reasons
    assert begins == 0
    assert owner._active is None


@pytest.mark.parametrize(
    "mutation",
    ["norm", "norm_weight", "rope", "projector"],
)
def test_mutated_target_math_declines_before_begin(monkeypatch, mutation):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner = FlatSegmentedKVRows(_rows(model, 1), identity=IDENTITY)
    target = DenseQwen3FlatSegmentTarget(
        model, capture_layers=(0, 2), identity=IDENTITY
    )
    if mutation == "norm":
        model.layers[0].input_layernorm.eps *= 2
    elif mutation == "norm_weight":
        model.layers[0].input_layernorm.weight = mx.ones((7,))
    elif mutation == "rope":
        model.layers[0].self_attn.rope.base *= 2
    else:
        model.layers[0].self_attn.q_proj = nn.Linear(8, 7, bias=False)
    begins = 0

    def begin_flat(_descriptor):
        nonlocal begins
        begins += 1
        raise AssertionError("mutated target must not begin")

    monkeypatch.setattr(owner, "begin_flat", begin_flat)
    admission = _admission(owner, target, descriptor)
    assert admission.available is False
    assert "target_capability_invalid" in admission.decline_reasons
    assert "target_owner_compatibility_invalid" in admission.decline_reasons
    assert begins == 0
    assert owner._active is None


def test_begin_revalidates_a_stale_pair_admission_before_opening_transaction():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner, target, admission = _pair(model, _rows(model, 1), descriptor)
    model.layers[0].self_attn.q_norm.eps *= 2

    with pytest.raises(RuntimeError, match="capability changed after review"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None
    assert owner.revision == 0


def test_capability_identity_source_and_structure_fail_closed_before_begin():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    rows = _rows(model, 2)
    owner, target, _ = _pair(model, rows, descriptor)
    mismatch = query_real_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity={**IDENTITY, "build": "other"},
        expected_owner_semantic_contract_sha256=owner.semantic_contract_sha256,
        expected_state_contract_fingerprints=owner.state_contract_fingerprints,
        expected_owner_source_hashes=owner.source_hashes,
        expected_target_source_hashes=target.source_hashes,
    )
    assert mismatch.available is False
    assert mismatch.decline_reasons == (
        "target_identity_mismatch",
        "owner_identity_mismatch",
        "device_identity_mismatch",
    )
    bad_sources = dict(target.source_hashes)
    bad_sources[next(iter(bad_sources))] = "0" * 64
    mismatch = query_real_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=owner.semantic_contract_sha256,
        expected_state_contract_fingerprints=owner.state_contract_fingerprints,
        expected_owner_source_hashes=owner.source_hashes,
        expected_target_source_hashes=bad_sources,
    )
    assert mismatch.decline_reasons == ("target_source_hashes_mismatch",)
    assert owner._active is None and target.invocations == 0

    model.args.num_experts = 1
    with pytest.raises(ValueError, match="dense full-attention"):
        target.ast_flat_segment_target_contract()


def test_rotating_and_shared_cache_owners_are_rejected():
    with pytest.raises(TypeError, match="exact KVCache only"):
        FlatSegmentedKVRows([[RotatingKVCache(max_size=8)]], identity=IDENTITY)
    model, _ = tiny()
    row = model.make_cache()
    with pytest.raises(ValueError, match="independent"):
        FlatSegmentedKVRows([row, row], identity=IDENTITY)

    mismatched = model.make_cache()
    mx.eval(model(mx.array([[1]]), cache=mismatched))
    mismatched[0].values = mismatched[0].values[..., :1, :]
    with pytest.raises(ValueError, match="storage geometry"):
        FlatSegmentedKVRows([mismatched], identity=IDENTITY)


def test_bfloat16_state_receipt_preserves_exact_storage_bits():
    cache = KVCache()
    keys = mx.arange(8, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 1, 4, 2)
    values = (keys + 1).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    original = canonical_kv_state([[cache]])
    clone = copy.deepcopy(cache)
    assert canonical_kv_state([[clone]]) == original

    changed = copy.deepcopy(cache)
    changed.keys = (changed.keys.view(mx.uint16) ^ 1).view(mx.bfloat16)
    assert canonical_kv_state([[changed]]) != original

    reshaped = copy.deepcopy(cache)
    reshaped.keys = reshaped.keys.reshape(1, 1, 128, 4)
    reshaped.values = reshaped.values.reshape(1, 1, 128, 4)
    assert canonical_kv_state([[reshaped]]) != original


def test_bfloat16_b4_mixed_depth_full_oracle_is_bit_exact():
    model, _ = tiny(tied=False)
    model.set_dtype(mx.bfloat16)
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    tokens = tuple((index * 5 + 1) % 8 for index in range(descriptor.logical_rows))
    accepted = (1, 1, 2, 4)
    rows = _rows(model, 4)
    base = copy.deepcopy(rows)
    expected_outputs, _ = _serial_outputs(model, base, descriptor, tokens)
    expected_rows = _accepted_state(model, base, descriptor, tokens, accepted)
    expected_state = canonical_kv_state(expected_rows)
    owner, target, admission = _pair(model, rows, descriptor)
    transaction = owner.begin_flat(
        descriptor,
        target_executor=target,
        admission=admission,
    )
    execution = target.execute_flat_segments(tokens, descriptor, transaction)
    for actual, expected in zip(execution.outputs, expected_outputs, strict=True):
        _equal(actual[0], expected[0])
        _equal(actual[1], expected[1])
    prepared = transaction.prepare_commit(accepted)
    assert prepared.state == expected_state
    assert transaction.publish_commit(prepared) == expected_state
    for actual, expected in zip(rows, expected_rows, strict=True):
        actual_next = model(mx.array([[8]]), cache=copy.deepcopy(actual))
        expected_next = model(mx.array([[8]]), cache=copy.deepcopy(expected))
        _equal(actual_next, expected_next)


def test_real_contract_is_not_admitted_by_the_synthetic_scope():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner = FlatSegmentedKVRows(_rows(model, 1), identity=IDENTITY)
    target = DenseQwen3FlatSegmentTarget(
        model, capture_layers=(0, 2), identity=IDENTITY
    )
    admission = query_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=owner.semantic_contract_sha256,
        expected_state_contract_fingerprints=owner.state_contract_fingerprints,
        expected_owner_source_hashes=owner.source_hashes,
        expected_target_source_hashes=target.source_hashes,
    )
    assert admission.available is False
    assert admission.decline_reasons == (
        "owner_capability_invalid",
        "target_capability_invalid",
    )
