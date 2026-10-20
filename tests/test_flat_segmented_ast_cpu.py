"""CPU-only contract and lifecycle tests for one-invocation flat AST."""

from __future__ import annotations

import copy

import pytest

from mlx2.runtime.flat_segmented_ast import (
    ABORT_LAW,
    BEGIN_LAW,
    COMMIT_FAILURE_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    EXECUTION_LAW,
    EXECUTION_SCOPE,
    FIDELITY_EXACT,
    LAYOUT_CLASS,
    OWNER_CAPABILITY_SCHEMA,
    OWNER_PROTOCOL,
    POSITION_BASE_AUTHORITY,
    STATE_WRITE_LAW,
    TARGET_CAPABILITY_SCHEMA,
    TARGET_PROTOCOL,
    TERMINAL_LAW,
    FlatSegmentDescriptor,
    FlatSegmentError,
    FlatSegmentExecution,
    FlatSegmentUnsupported,
    PreparedFlatSegmentCommit,
    execute_synthetic_flat_segment_round,
    query_flat_segment_admission,
)
from mlx2.runtime.ragged_verify_layout import (
    RaggedBackendMode,
    RaggedVerifyLayout,
    padded_capabilities,
)

IDENTITY = {
    "source_revision": "66e6d0ec36afafd9e4d62b90fabb92e693497cef",
    "build": "cpu-contract-test",
    "device": "host-only",
    "artifact": "synthetic-flat-segment@sha256:" + "b" * 64,
}
LIMITS = {"max_lanes": 8, "max_segment_rows": 8, "max_total_rows": 32}
KV_FINGERPRINT = "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e"
RECURRENT_FINGERPRINT = (
    "8491ce6cf23f5a337f6ec7a052bdfb8cd036c01ecf2b6e51e773d6798a764cbe"
)
OWNER_DIGESTS = {"attention": "c" * 64, "hybrid": "d" * 64}
OWNER_SOURCES = {"tests/synthetic-flat-owner.py": "e" * 64}
TARGET_SOURCES = {"tests/synthetic-flat-target.py": "a" * 64}


def _owner_contract(kind="attention"):
    hybrid = kind == "hybrid"
    fingerprints = (
        [KV_FINGERPRINT, RECURRENT_FINGERPRINT] if hybrid else [KV_FINGERPRINT]
    )
    return {
        "schema": OWNER_CAPABILITY_SCHEMA,
        "protocol": OWNER_PROTOCOL,
        "execution_scope": EXECUTION_SCOPE,
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "layout_classes": [LAYOUT_CLASS],
        "state_ownership": "independent_request_private_b1_lanes",
        "state_plane_kinds": (
            ["attention_kv", "gdn_recurrent"] if hybrid else ["attention_kv"]
        ),
        "state_families": ["kv", "recurrent"] if hybrid else ["kv"],
        "state_contract_fingerprints": fingerprints,
        "owner_semantic_contract_sha256": OWNER_DIGESTS[kind],
        "transaction_operations": {
            "begin_flat": True,
            "prepare_commit": True,
            "publish_commit": True,
            "abort": True,
        },
        "begin_law": BEGIN_LAW,
        "commit_law": COMMIT_LAW,
        "commit_failure_law": COMMIT_FAILURE_LAW,
        "abort_law": ABORT_LAW,
        "rollback_law": (
            "private_kv_and_recurrent_snapshot_restore"
            if hybrid
            else "private_kv_snapshot_restore"
        ),
        "position_base_authority": POSITION_BASE_AUTHORITY,
        "hard_limits": dict(LIMITS),
        "source_hashes": dict(OWNER_SOURCES),
        "identity": dict(IDENTITY),
    }


def _target_contract(**updates):
    contract = {
        "schema": TARGET_CAPABILITY_SCHEMA,
        "protocol": TARGET_PROTOCOL,
        "execution_scope": EXECUTION_SCOPE,
        "fidelity": FIDELITY_EXACT,
        "execution_law": EXECUTION_LAW,
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "layout_classes": [LAYOUT_CLASS],
        "input_storage": "flat_logical_rows",
        "output_storage": "flat_logical_rows_in_segment_order",
        "position_law": "per_segment_base_plus_local_offset",
        "causal_isolation": "strict_no_cross_segment_reads",
        "terminal_law": TERMINAL_LAW,
        "state_write_law": STATE_WRITE_LAW,
        "commit_law": COMMIT_LAW,
        "abort_law": ABORT_LAW,
        "hard_limits": dict(LIMITS),
        "source_hashes": dict(TARGET_SOURCES),
        "identity": dict(IDENTITY),
    }
    contract.update(updates)
    return contract


def _initial_state(lanes, kind):
    return [
        {
            "kv": [lane + 1],
            "recurrent": 10 + lane if kind == "hybrid" else None,
        }
        for lane in range(lanes)
    ]


def _step(state, token, kind, *, stop_after_attention=False):
    attention = int(token) + sum(state["kv"])
    state["kv"].append(int(token))
    if stop_after_attention:
        return None
    if kind == "hybrid":
        state["recurrent"] = state["recurrent"] * 3 + int(token)
        return (attention, state["recurrent"])
    return (attention,)


def _serial_oracle(base, kind):
    def run(lane, inputs):
        state = copy.deepcopy(base[lane])
        return tuple(_step(state, token, kind) for token in inputs)

    return run


def _replay(base, kind, flat_inputs, descriptor, consumed):
    result = copy.deepcopy(base)
    for lane, count in enumerate(consumed):
        segment = flat_inputs[descriptor.lane_slice(lane)]
        for token in segment[:count]:
            _step(result[lane], token, kind)
    return result


def _commit_oracle(base, kind, flat_inputs, descriptor):
    return lambda consumed: _replay(base, kind, flat_inputs, descriptor, consumed)


def _expected_fingerprints(kind):
    return (
        (KV_FINGERPRINT, RECURRENT_FINGERPRINT)
        if kind == "hybrid"
        else (KV_FINGERPRINT,)
    )


def _admission(
    owner,
    target,
    descriptor,
    *,
    expected_identity=IDENTITY,
    expected_owner_digest=None,
    expected_fingerprints=None,
):
    kind = owner.kind
    return query_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            OWNER_DIGESTS[kind]
            if expected_owner_digest is None
            else expected_owner_digest
        ),
        expected_state_contract_fingerprints=(
            _expected_fingerprints(kind)
            if expected_fingerprints is None
            else expected_fingerprints
        ),
        expected_owner_source_hashes=OWNER_SOURCES,
        expected_target_source_hashes=TARGET_SOURCES,
    )


def _execute(owner, target, descriptor, flat_inputs, accepted, serial_oracle):
    base = copy.deepcopy(owner.state)
    return execute_synthetic_flat_segment_round(
        owner,
        target,
        descriptor,
        flat_inputs,
        accepted,
        serial_oracle,
        _commit_oracle(base, owner.kind, flat_inputs, descriptor),
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=OWNER_DIGESTS[owner.kind],
        expected_state_contract_fingerprints=_expected_fingerprints(owner.kind),
        expected_owner_source_hashes=OWNER_SOURCES,
        expected_target_source_hashes=TARGET_SOURCES,
    )


class _SyntheticTransaction:
    def __init__(self, owner, descriptor):
        self.owner = owner
        self.descriptor = descriptor
        self.base = copy.deepcopy(owner.state)
        self.private = copy.deepcopy(owner.state)
        self.checkpoints = [[] for _ in range(descriptor.lane_count)]
        self.closed = False
        self.abort_calls = 0

    def apply(self, lane, token, *, stop_after_attention=False):
        output = _step(
            self.private[lane],
            token,
            self.owner.kind,
            stop_after_attention=stop_after_attention,
        )
        if not stop_after_attention:
            self.checkpoints[lane].append(copy.deepcopy(self.private[lane]))
        return output

    def prepare_commit(self, accepted_lengths):
        accepted = tuple(accepted_lengths)
        next_state = [
            copy.deepcopy(self.checkpoints[lane][count - 1])
            for lane, count in enumerate(accepted)
        ]
        if self.owner.failure == "wrong_prepared_state":
            next_state[0]["kv"].append(999)
        return PreparedFlatSegmentCommit(
            state=next_state,
            accepted_lengths=accepted,
            geometry_sha256=self.descriptor.geometry_sha256,
        )

    def publish_commit(self, prepared):
        self.owner.state = copy.deepcopy(prepared.state)
        if self.owner.failure == "wrong_published_state":
            self.owner.state[0]["kv"].append(998)
            return copy.deepcopy(self.owner.state)
        if self.owner.failure == "commit_close_then_raise":
            self.closed = True
            self.owner.state = copy.deepcopy(self.base)
            raise RuntimeError("synthetic commit closed then failed")
        self.closed = True
        self.owner.active = None
        return copy.deepcopy(self.owner.state)

    def abort(self):
        self.abort_calls += 1
        self.owner.state = copy.deepcopy(self.base)
        self.closed = True
        self.owner.active = None


class _SyntheticOwner:
    def __init__(self, lanes, kind="attention", *, failure=None):
        self.kind = kind
        self.failure = failure
        self.state = _initial_state(lanes, kind)
        self.begin_calls = 0
        self.active = None

    def ast_flat_segment_transaction_contract(self):
        return _owner_contract(self.kind)

    def begin_flat(self, descriptor):
        if self.active is not None and not self.active.closed:
            raise RuntimeError("synthetic owner already leased")
        self.begin_calls += 1
        if self.failure == "begin_failure":
            before = copy.deepcopy(self.state)
            self.state[0]["kv"].append(997)
            self.state = before
            raise RuntimeError("synthetic atomic begin failed")
        self.active = _SyntheticTransaction(self, descriptor)
        return self.active


class _SyntheticTarget:
    def __init__(self, *, failure=None, output_mode="exact", contract=None):
        self.failure = failure
        self.output_mode = output_mode
        self.contract = _target_contract() if contract is None else contract
        self.invocations = 0

    def ast_flat_segment_target_contract(self):
        return copy.deepcopy(self.contract)

    def execute_flat_segments(self, flat_inputs, descriptor, transaction):
        self.invocations += 1
        if self.failure == "before_first_layer":
            raise RuntimeError("synthetic failure before first layer")
        outputs = []
        for lane in range(descriptor.lane_count):
            segment = flat_inputs[descriptor.lane_slice(lane)]
            for position, token in enumerate(segment):
                if self.failure == "mid_attention" and lane == 1 and position == 0:
                    transaction.apply(lane, token, stop_after_attention=True)
                    raise RuntimeError("synthetic failure mid attention")
                output = transaction.apply(lane, token)
                outputs.append(output)
                if self.failure == "mid_recurrence" and lane == 1 and position == 0:
                    raise RuntimeError("synthetic failure mid recurrence")
        if self.failure == "after_outputs":
            raise RuntimeError("synthetic failure after outputs")
        if self.output_mode == "short":
            outputs.pop()
        elif self.output_mode == "reversed":
            outputs.reverse()
        return FlatSegmentExecution(
            outputs=tuple(outputs),
            geometry_sha256=descriptor.geometry_sha256,
            invocation_count=True if self.output_mode == "bool_count" else 1,
        )


def _run(kind="attention", accepted=(0, 1, 2), flat_inputs=(4, 5, 6, 7, 8, 9)):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3))
    owner = _SyntheticOwner(3, kind)
    target = _SyntheticTarget()
    base = copy.deepcopy(owner.state)
    result = execute_synthetic_flat_segment_round(
        owner,
        target,
        descriptor,
        flat_inputs,
        accepted,
        _serial_oracle(base, kind),
        _commit_oracle(base, kind, flat_inputs, descriptor),
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=OWNER_DIGESTS[kind],
        expected_state_contract_fingerprints=_expected_fingerprints(kind),
        expected_owner_source_hashes=OWNER_SOURCES,
        expected_target_source_hashes=TARGET_SOURCES,
    )
    return descriptor, owner, target, base, result


def test_descriptor_is_strict_immutable_token_free_geometry():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 4))
    assert descriptor.query_lengths == (1, 2, 4)
    assert descriptor.cu_query_lengths == (0, 1, 3, 7)
    assert descriptor.token_to_lane == (0, 1, 1, 2, 2, 2, 2)
    assert descriptor.local_positions == (0, 0, 1, 0, 1, 2, 3)
    assert descriptor.draft_depths == (0, 1, 3)
    receipt = descriptor.receipt()
    assert FlatSegmentDescriptor.from_mapping(receipt) == descriptor
    assert "token" not in receipt
    assert len(descriptor.geometry_sha256) == 64
    with pytest.raises(AttributeError):
        descriptor.query_lengths = (1,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (((1, 2), (0, 2, 3), (0, 1, 1), (0, 0, 1)), "cumulative"),
        (((1, 2), (0, 1, 3), (0, 0, 1), (0, 0, 1)), "segment order"),
        (((1, 2), (0, 1, 3), (0, 1, 1), (0, 1, 0)), "restart"),
        (((0,), (0, 0), (), ()), ">= 1"),
    ],
)
def test_descriptor_rejects_invalid_offsets_ownership_positions_and_zero(
    fields, message
):
    with pytest.raises(FlatSegmentError, match=message):
        FlatSegmentDescriptor(*fields)


def test_descriptor_mapping_rejects_unknown_schema_and_digest():
    receipt = FlatSegmentDescriptor.from_query_lengths((2, 1)).receipt()
    bad = dict(receipt, schema="mlx2.ast-flat-segment-layout.v0")
    with pytest.raises(FlatSegmentError, match="schema differs"):
        FlatSegmentDescriptor.from_mapping(bad)
    bad = dict(receipt, geometry_sha256="0" * 64)
    with pytest.raises(FlatSegmentError, match="digest differs"):
        FlatSegmentDescriptor.from_mapping(bad)
    bad = dict(receipt, tokens=[1, 2, 3])
    with pytest.raises(FlatSegmentError, match="fields differ"):
        FlatSegmentDescriptor.from_mapping(bad)
    bad = dict(receipt, query_lengths=3)
    with pytest.raises(FlatSegmentError, match="sequence"):
        FlatSegmentDescriptor.from_mapping(bad)


def test_strict_owner_target_intersection_admits_only_exact_bounded_pair():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3))
    owner = _SyntheticOwner(3, "hybrid")
    target = _SyntheticTarget()
    admission = _admission(owner, target, descriptor)
    receipt = admission.receipt()
    assert admission.available
    assert receipt["owner_contract"]["state_plane_kinds"] == (
        "attention_kv",
        "gdn_recurrent",
    )
    assert receipt["target_contract"]["causal_isolation"] == (
        "strict_no_cross_segment_reads"
    )
    assert receipt["owner_contract"]["position_base_authority"] == (
        "transaction_owner_state"
    )
    assert receipt["owner_contract"]["commit_failure_law"] == (
        "restore_exact_pre_begin_state"
    )
    assert receipt["effective_hard_limits"] == LIMITS
    assert receipt["one_target_executor_invocation_required"] is True
    assert receipt["depthwise_shrinking_loop_permitted"] is False
    assert receipt["implemented"] is True
    assert receipt["qualified"] is False
    assert receipt["selected"] is False
    assert receipt["observed_used"] is False
    assert "preferred" not in str(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "mlx2.ast-flat-segment-target-capability.v0"),
        ("protocol", "depthwise-segmented-v1"),
        ("execution_scope", "production"),
        ("fidelity", "near_tie_bounded"),
        ("execution_law", "one_depth_at_a_time"),
        ("layout_classes", ["depthwise_segmented"]),
        ("input_storage", "padded_rectangular"),
        ("causal_isolation", "unattested"),
        ("state_write_law", "eager_publication"),
        ("source_hashes", {"fixture.py": "not-sha256"}),
    ],
)
def test_malformed_target_capability_fails_closed_before_begin(field, value):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2)
    contract = _target_contract()
    contract[field] = value
    target = _SyntheticTarget(contract=contract)
    admission = _admission(owner, target, descriptor)
    assert admission.available is False
    assert admission.decline_reasons == ("target_capability_invalid",)
    with pytest.raises(FlatSegmentUnsupported, match="target_capability_invalid"):
        _execute(
            owner,
            target,
            descriptor,
            (1, 2, 3),
            (0, 1),
            lambda _lane, values: tuple((value,) for value in values),
        )
    assert owner.begin_calls == 0
    assert target.invocations == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "mlx2.ast-flat-segment-owner-capability.v0"),
        ("execution_scope", "production"),
        ("layout_classes", ["depthwise_segmented"]),
        ("begin_law", "may_leave_partial_state"),
        ("commit_failure_law", "state_unknown"),
        ("position_base_authority", "target_guess"),
        ("transaction_operations", {"begin_flat": True, "commit": True}),
        ("state_ownership", "shared_rows"),
        ("state_plane_kinds", ["fictional_state"]),
        ("state_families", ["recurrent"]),
        ("state_contract_fingerprints", ["synthetic-kv-v1"]),
        ("state_contract_fingerprints", [KV_FINGERPRINT, KV_FINGERPRINT]),
        ("owner_semantic_contract_sha256", "not-sha256"),
        ("source_hashes", {"fixture.py": "not-sha256"}),
        ("identity", {**IDENTITY, "extra": "field"}),
    ],
)
def test_malformed_owner_capability_fails_closed(field, value):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2)
    contract = _owner_contract()
    contract[field] = value
    owner.ast_flat_segment_transaction_contract = lambda: contract
    target = _SyntheticTarget()
    admission = _admission(owner, target, descriptor)
    assert admission.available is False
    assert admission.decline_reasons == ("owner_capability_invalid",)
    assert owner.begin_calls == target.invocations == 0


def test_capability_provider_failure_declines_without_execution():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    owner = _SyntheticOwner(1)
    target = _SyntheticTarget()

    def unavailable():
        raise OSError("installed provider unavailable")

    target.ast_flat_segment_target_contract = unavailable
    admission = _admission(owner, target, descriptor)
    assert admission.decline_reasons == ("target_capability_invalid",)
    assert owner.begin_calls == target.invocations == 0


def test_identity_and_hard_limit_mismatch_decline_before_state_or_rng_use():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3))
    owner = _SyntheticOwner(3)
    target = _SyntheticTarget()
    target.contract["identity"] = dict(IDENTITY, build="different-build")
    mismatch = _admission(owner, target, descriptor)
    assert mismatch.decline_reasons == ("target_identity_mismatch",)
    target.contract["identity"] = dict(IDENTITY)

    owner_contract = _owner_contract()
    owner_contract["identity"] = dict(IDENTITY, build="different-build")
    owner.ast_flat_segment_transaction_contract = lambda: owner_contract
    mismatch = _admission(owner, target, descriptor)
    assert mismatch.decline_reasons == ("owner_identity_mismatch",)
    owner.ast_flat_segment_transaction_contract = lambda: _owner_contract()

    mismatch = _admission(owner, target, descriptor, expected_owner_digest="f" * 64)
    assert mismatch.decline_reasons == ("owner_semantic_contract_mismatch",)
    mismatch = _admission(
        owner,
        target,
        descriptor,
        expected_fingerprints=(RECURRENT_FINGERPRINT,),
    )
    assert mismatch.decline_reasons == ("owner_state_contract_fingerprints_mismatch",)
    owner_contract = _owner_contract()
    owner_contract["source_hashes"] = {"tests/synthetic-flat-owner.py": "f" * 64}
    owner.ast_flat_segment_transaction_contract = lambda: owner_contract
    mismatch = _admission(owner, target, descriptor)
    assert mismatch.decline_reasons == ("owner_source_hashes_mismatch",)
    owner.ast_flat_segment_transaction_contract = lambda: _owner_contract()
    target.contract["source_hashes"] = {"tests/synthetic-flat-target.py": "f" * 64}
    mismatch = _admission(owner, target, descriptor)
    assert mismatch.decline_reasons == ("target_source_hashes_mismatch",)
    target.contract["source_hashes"] = dict(TARGET_SOURCES)
    target.contract["hard_limits"] = {
        "max_lanes": 2,
        "max_segment_rows": 8,
        "max_total_rows": 32,
    }
    limited = _admission(owner, target, descriptor)
    assert limited.decline_reasons == ("descriptor_exceeds_hard_limits",)
    assert owner.begin_calls == target.invocations == 0


def test_one_invocation_mixed_terminal_depths_matches_serial_attention():
    descriptor, owner, target, base, result = _run()
    assert target.invocations == result.target_invocations == 1
    assert owner.begin_calls == 1
    assert descriptor.draft_depths == (0, 1, 2)
    assert result.consumed_lengths == (1, 2, 3)
    assert result.outputs == tuple(
        value
        for lane in range(descriptor.lane_count)
        for value in _serial_oracle(base, "attention")(
            lane, (4, 5, 6, 7, 8, 9)[descriptor.lane_slice(lane)]
        )
    )
    assert result.receipt()["target_invocations"] == 1
    assert result.receipt()["observed_used"] is True


def test_both_ordinary_oracles_complete_before_transaction_lease():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    flat_inputs = (4, 5, 6)
    owner = _SyntheticOwner(2)
    target = _SyntheticTarget()
    base = copy.deepcopy(owner.state)
    serial = _serial_oracle(base, "attention")
    commit_oracle_calls = 0

    def committed(consumed):
        nonlocal commit_oracle_calls
        commit_oracle_calls += 1
        assert owner.active is None
        return _replay(base, "attention", flat_inputs, descriptor, consumed)

    result = execute_synthetic_flat_segment_round(
        owner,
        target,
        descriptor,
        flat_inputs,
        (0, 1),
        serial,
        committed,
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=OWNER_DIGESTS["attention"],
        expected_state_contract_fingerprints=_expected_fingerprints("attention"),
        expected_owner_source_hashes=OWNER_SOURCES,
        expected_target_source_hashes=TARGET_SOURCES,
    )
    assert commit_oracle_calls == 1
    assert result.target_invocations == 1


def test_neighbor_tokens_cannot_change_another_segment_outputs():
    first = _run(flat_inputs=(4, 5, 6, 7, 8, 9))[4]
    second = _run(flat_inputs=(4, 50, 60, 70, 80, 90))[4]
    assert first.segment_outputs[0] == second.segment_outputs[0]
    assert first.segment_outputs[1:] != second.segment_outputs[1:]


@pytest.mark.parametrize("accepted", [(0, 0, 0), (0, 1, 1), (0, 1, 2)])
@pytest.mark.parametrize("kind", ["attention", "hybrid"])
def test_commit_matches_ordinary_consumed_prefix_for_terminal_and_rejection_cases(
    kind, accepted
):
    flat_inputs = (4, 5, 6, 7, 8, 9)
    descriptor, owner, _target, base, result = _run(
        kind=kind, accepted=accepted, flat_inputs=flat_inputs
    )
    consumed = tuple(value + 1 for value in accepted)
    assert result.consumed_lengths == consumed
    assert owner.state == _replay(base, kind, flat_inputs, descriptor, consumed)


@pytest.mark.parametrize(
    "failure",
    ["before_first_layer", "mid_attention", "mid_recurrence", "after_outputs"],
)
@pytest.mark.parametrize("kind", ["attention", "hybrid"])
def test_execution_failures_abort_exact_pre_begin_state(kind, failure):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3))
    owner = _SyntheticOwner(3, kind)
    target = _SyntheticTarget(failure=failure)
    before = copy.deepcopy(owner.state)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6, 7, 8, 9),
            (0, 1, 2),
            _serial_oracle(before, kind),
        )
    assert owner.state == before
    assert owner.active is None
    assert target.invocations == 1


def test_begin_failure_is_atomic_and_does_not_invoke_target():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2, failure="begin_failure")
    target = _SyntheticTarget()
    before = copy.deepcopy(owner.state)
    with pytest.raises(RuntimeError, match="atomic begin failed"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "attention"),
        )
    assert owner.state == before
    assert owner.active is None
    assert target.invocations == 0


def test_commit_close_then_raise_restores_exact_pre_begin_state():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2, "hybrid", failure="commit_close_then_raise")
    target = _SyntheticTarget()
    before = copy.deepcopy(owner.state)
    with pytest.raises(RuntimeError, match="commit closed then failed"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "hybrid"),
        )
    assert owner.state == before
    assert owner.active is None


def test_wrong_prepared_state_is_rejected_before_publication():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2, "hybrid", failure="wrong_prepared_state")
    target = _SyntheticTarget()
    before = copy.deepcopy(owner.state)
    with pytest.raises(FlatSegmentError, match="accepted-prefix oracle"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "hybrid"),
        )
    assert owner.state == before
    assert owner.active is None


def test_wrong_published_state_is_aborted_back_to_pre_begin_state():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2, "hybrid", failure="wrong_published_state")
    target = _SyntheticTarget()
    before = copy.deepcopy(owner.state)
    with pytest.raises(FlatSegmentError, match="published state differs"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "hybrid"),
        )
    assert owner.state == before
    assert owner.active is None


@pytest.mark.parametrize("mode", ["short", "reversed"])
def test_wrong_output_shape_or_order_aborts_without_publication(mode):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2)
    target = _SyntheticTarget(output_mode=mode)
    before = copy.deepcopy(owner.state)
    message = "row count differs" if mode == "short" else "serial oracle"
    with pytest.raises(FlatSegmentError, match=message):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "attention"),
        )
    assert owner.state == before


def test_boolean_invocation_attestation_is_not_integer_one():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2)
    target = _SyntheticTarget(output_mode="bool_count")
    before = copy.deepcopy(owner.state)
    with pytest.raises(FlatSegmentError, match="exactly one invocation"):
        _execute(
            owner,
            target,
            descriptor,
            (4, 5, 6),
            (0, 1),
            _serial_oracle(before, "attention"),
        )
    assert owner.state == before


def test_invalid_acceptance_and_input_shape_fail_before_transaction():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    owner = _SyntheticOwner(2)
    target = _SyntheticTarget()
    oracle = _serial_oracle(copy.deepcopy(owner.state), "attention")
    with pytest.raises(FlatSegmentError, match="logical row"):
        _execute(
            owner,
            target,
            descriptor,
            (1, 2),
            (0, 1),
            oracle,
        )
    with pytest.raises(FlatSegmentError, match="proposal depth"):
        _execute(
            owner,
            target,
            descriptor,
            (1, 2, 3),
            (1, 1),
            oracle,
        )
    assert owner.begin_calls == 0
    assert target.invocations == 0


def test_existing_padded_backend_remains_the_only_established_fallback():
    layout = RaggedVerifyLayout.from_draft_depths((10, 20, 30), (0, 1, 2))
    plan = layout.select_backend(padded_capabilities())
    assert plan.mode is RaggedBackendMode.PADDED
    assert plan.logical_rows == 6
    assert plan.physical_rows == 9
    assert plan.padding_rows == 3
