"""CPU checks for the default-off Qwen3.8 native-flat serving bridge."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest
from test_qwen38_hybrid_native_flat_cpu import _pair

from mlx2.adapters.qwen38_hybrid_metal_probe import (
    QWEN38_27B_OQ4E_MTP_CHECKPOINT,
)
from mlx2.adapters.qwen38_hybrid_native_flat import (
    SERVING_SOURCE_PATHS,
    SOURCE_PATHS,
    Qwen38NativeFlatPhysicalTargetProvider,
    Qwen38NativeFlatServingBinding,
    _ordinary_serial_oracles,
    install_qwen38_native_flat_physical_target_provider,
)
from mlx2.runtime.ast_physical_target import ast_serving_admission_receipt
from mlx2.runtime.flat_segmented_hybrid import (
    canonical_hybrid_state,
    successor_bound_hybrid_state,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows


def _admission(descriptor):
    discovery = {
        "schema": "mlx2.ast-physical-target-candidate.v1",
        "candidate_implemented": True,
        "implemented": False,
        "eligible": False,
        "manifest_sha256": "2" * 64,
        "provider_binding_sha256": "3" * 64,
        "geometry_sha256": descriptor.geometry_sha256,
    }
    admission = ast_serving_admission_receipt(
        discovery=discovery,
        session_scope_sha256="1" * 64,
        source_manifest_sha256="4" * 64,
        build_manifest_sha256="5" * 64,
        device_manifest_sha256="6" * 64,
        owner_capability_manifest_sha256="7" * 64,
        target_capability_manifest_sha256="8" * 64,
    )
    return discovery, admission


def _manifest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _Provider:
    def __init__(self, target):
        self.target = target
        self.drifted = False
        self.hard_limits = dict(target.hard_limits)
        self.source_identity = tuple(sorted(target.source_hashes.items()))

    def _target(self, descriptor, capture_layers):
        assert descriptor == self.target.descriptor
        assert tuple(capture_layers) == self.target.capture_layers
        return self.target

    def validate_live_serving_admission(self, **_kwargs):
        if self.drifted:
            raise RuntimeError("injected post-execute provider drift")
        self.target._validate_model()


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _binding(lengths=(1, 2, 3, 4)):
    descriptor, owner, direct_target = _pair(lengths)
    direct_target.model.configure_target_verify_row_exact(True)
    target = type(direct_target)(
        direct_target.model,
        descriptor=descriptor,
        capture_layers=direct_target.capture_layers,
        identity=direct_target.identity,
        expected_model_manifest_sha256=direct_target._expected_model_manifest_sha256,
        construction_role="serving",
    )
    tokens = tuple(range(17, 17 + descriptor.logical_rows))
    provider = _Provider(target)
    discovery, admission = _admission(descriptor)
    binding = Qwen38NativeFlatServingBinding(
        provider,
        rows=owner.rows,
        descriptor=descriptor,
        capture_layers=target.capture_layers,
        flat_inputs=tokens,
        session_scope_sha256="1" * 64,
        discovery=discovery,
        admission=admission,
    )
    return descriptor, owner, target, tokens, binding


def test_live_bridge_executes_flat_rows_and_transfers_exact_successors():
    descriptor, owner, target, tokens, binding = _binding()
    accepted = (1, 1, 2, 3)
    continuation = (31, 32, 33, 34)
    expected_outputs, _state, _continuation = _ordinary_serial_oracles(
        target, owner, descriptor, tokens, accepted, continuation
    )
    original = canonical_hybrid_state(owner.rows)
    initial_target_receipt = target.receipt()
    assert initial_target_receipt["serving_wired"] is True
    assert initial_target_receipt["selected"] is False
    assert initial_target_receipt["direct_probe_only"] is False
    assert initial_target_receipt["target_row_exact_installed"] is True
    assert initial_target_receipt["observed_used_cpu_fixture"] is False

    result = binding.execute()
    actual_outputs = tuple(
        result.row(lane, position)
        for lane, length in enumerate(descriptor.query_lengths)
        for position in range(length)
    )
    assert all(
        mx.array_equal(actual_logits, expected_logits)
        and mx.array_equal(actual_features, expected_features)
        for (actual_logits, actual_features), (
            expected_logits,
            expected_features,
        ) in zip(actual_outputs, expected_outputs, strict=True)
    )
    assert canonical_hybrid_state(owner.rows) == original

    published = binding.commit(accepted)
    expected_rows = copy.deepcopy(owner.rows)
    for lane, count in enumerate(accepted):
        lane_tokens = tokens[descriptor.lane_slice(lane)][:count]
        for token in lane_tokens:
            mx.eval(
                target.model(
                    mx.array([[token]], dtype=mx.int32),
                    cache=expected_rows[lane],
                )
            )
    assert canonical_hybrid_state(published) == canonical_hybrid_state(expected_rows)
    assert canonical_hybrid_state(owner.rows) == original
    receipt = binding.receipt()
    assert receipt["selected"] is True
    assert receipt["observed_used"] is True
    assert receipt["default"] is receipt["qualified"] is False
    assert receipt["registry_epoch"] == 1
    assert receipt["result"]["padding_rows"] == 0
    assert receipt["execution_mode"] == "native_flat_b1s1_rows"
    assert len(binding.delegate.transactions) == 4
    assert all(
        transaction.batched is False for transaction in binding.delegate.transactions
    )
    target_receipt = target.receipt()
    assert target_receipt["observed_execution_cpu_fixture"] is True
    assert target_receipt["observed_used_cpu_fixture"] is False
    assert target_receipt["row_arithmetic"] == "native_flat_b1s1_rows"


def test_uniform_live_bridge_uses_one_segmented_traversal_and_commits_exactly():
    descriptor, owner, target, tokens, binding = _binding((2, 2, 2, 2))
    accepted = (1, 2, 1, 2)
    continuation = (31, 32, 33, 34)
    expected_outputs, expected_state, _continuation = _ordinary_serial_oracles(
        target, owner, descriptor, tokens, accepted, continuation
    )
    original = canonical_hybrid_state(owner.rows)
    row_exact_route = target.model.__dict__["_mlx2_qwen_hybrid_row_exact"]
    counts_before = row_exact_route.status()["counts"]

    result = binding.execute()

    actual_outputs = tuple(
        result.row(lane, position)
        for lane, length in enumerate(descriptor.query_lengths)
        for position in range(length)
    )
    assert all(
        mx.array_equal(actual_logits, expected_logits)
        and mx.array_equal(actual_features, expected_features)
        for (actual_logits, actual_features), (
            expected_logits,
            expected_features,
        ) in zip(actual_outputs, expected_outputs, strict=True)
    )
    counts_after = row_exact_route.status()["counts"]
    assert counts_after["launches"] == counts_before["launches"] + 1
    assert counts_after["physical_width_hist"] == {"4": 1}
    assert canonical_hybrid_state(owner.rows) == original
    assert len(binding.delegate.transactions) == 1
    assert binding.delegate.transactions[0].batched is True
    assert binding.receipt()["execution_mode"] == "segmented_compact_valid_rows"
    assert target.receipt()["row_arithmetic"] == "segmented_compact_valid_rows"

    published = binding.commit(accepted)
    assert (
        successor_bound_hybrid_state(published, owner.boundaries, accepted)
        == expected_state
    )
    assert canonical_hybrid_state(owner.rows) == original


def test_session_binding_runs_full_model_audit_once_then_uses_hot_guard(
    monkeypatch,
):
    _descriptor, _owner, target, _tokens, _binding_value = _binding((2, 2))
    calls = {"full": 0, "hot": 0}
    full = target._validate_model
    hot = target._validate_hot_model_binding

    def counted_full():
        calls["full"] += 1
        return full()

    def counted_hot():
        calls["hot"] += 1
        return hot()

    monkeypatch.setattr(target, "_validate_model", counted_full)
    monkeypatch.setattr(target, "_validate_hot_model_binding", counted_hot)
    scope = "a" * 64

    target.validate_session_model_binding(scope)
    target.validate_session_model_binding(scope)
    target.ast_live_serving_target_contract()
    target.ast_live_serving_target_contract()

    assert calls == {"full": 1, "hot": 1}

    target.validate_session_model_binding("b" * 64)
    assert calls == {"full": 2, "hot": 1}


def test_hot_model_guard_rejects_mutable_row_exact_hook_drift(monkeypatch):
    _descriptor, _owner, target, _tokens, _binding_value = _binding((2, 2))
    monkeypatch.setattr(target.model, "progressive_external_compact_exact_rows", False)

    with pytest.raises(ValueError, match="model binding drifted"):
        target._validate_hot_model_binding()


def test_live_bridge_failure_aborts_every_b1_lane_without_base_mutation(monkeypatch):
    _descriptor, owner, target, _tokens, binding = _binding((2, 3))
    original = canonical_hybrid_state(owner.rows)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected native target failure")

    monkeypatch.setattr(target, "execute_live_flat_segments", fail)
    with pytest.raises(RuntimeError, match="injected native target failure"):
        binding.execute()
    assert canonical_hybrid_state(owner.rows) == original
    assert all(transaction.closed for transaction in binding.delegate.transactions)
    assert binding.registry.epoch == 0
    assert binding.registry.receipt()["poisoned"] is False


def test_uniform_live_bridge_failure_aborts_aggregate_transaction(monkeypatch):
    _descriptor, owner, target, _tokens, binding = _binding((2, 2))
    original = canonical_hybrid_state(owner.rows)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected segmented target failure")

    monkeypatch.setattr(
        "mlx2.runtime.progressive_external_verify.execute_progressive_target_rows",
        fail,
    )
    with pytest.raises(RuntimeError, match="injected segmented target failure"):
        binding.execute()
    assert canonical_hybrid_state(owner.rows) == original
    assert len(binding.delegate.transactions) == 1
    assert binding.delegate.transactions[0].closed is True
    target_receipt = target.receipt()
    assert target_receipt["target_invocations"] == 1
    assert target_receipt["successful_target_invocations"] == 0
    assert binding.registry.epoch == 0
    assert binding.registry.receipt()["poisoned"] is False


def test_uniform_live_bridge_explicit_abort_restores_every_private_lane():
    _descriptor, owner, _target, _tokens, binding = _binding((2, 2))
    original = canonical_hybrid_state(owner.rows)

    binding.execute()
    binding.abort()

    assert canonical_hybrid_state(owner.rows) == original
    assert all(transaction.closed for transaction in binding.delegate.transactions)
    assert binding.registry.epoch == 0
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_binding_construction_does_not_acquire_live_lease():
    _descriptor, _owner, _target, _tokens, binding = _binding((2, 3))
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_lane_begin_failure_aborts_every_open_b1_transaction(monkeypatch):
    _descriptor, owner, _target, _tokens, binding = _binding((2, 3, 4))
    original_state = canonical_hybrid_state(owner.rows)
    original_begin = HybridVerifyRows.begin
    opened = []

    def fail_second(self, *args, **kwargs):
        if len(opened) == 1:
            raise RuntimeError("injected lane begin failure")
        transaction = original_begin(self, *args, **kwargs)
        opened.append(transaction)
        return transaction

    monkeypatch.setattr(HybridVerifyRows, "begin", fail_second)
    with pytest.raises(RuntimeError, match="injected lane begin failure"):
        binding.execute()
    assert len(opened) == 1
    assert opened[0].closed is True
    assert canonical_hybrid_state(owner.rows) == original_state
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_prepare_failure_aborts_lease_without_base_mutation(monkeypatch):
    _descriptor, owner, _target, _tokens, binding = _binding((2, 2))
    original_state = canonical_hybrid_state(owner.rows)
    binding.execute()
    transaction_type = type(binding.delegate.transactions[0])

    def fail_commit(self, *args, **kwargs):
        raise RuntimeError("injected prepare failure")

    monkeypatch.setattr(transaction_type, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="injected prepare failure"):
        binding.commit((1, 1))
    assert canonical_hybrid_state(owner.rows) == original_state
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_publish_failure_aborts_lease_without_base_mutation(monkeypatch):
    _descriptor, owner, _target, _tokens, binding = _binding((2, 2))
    original_state = canonical_hybrid_state(owner.rows)
    binding.execute()

    def fail_publish(_prepared):
        raise RuntimeError("injected publish failure")

    monkeypatch.setattr(binding.delegate, "publish_commit", fail_publish)
    with pytest.raises(RuntimeError, match="injected publish failure"):
        binding.commit((1, 1))
    assert canonical_hybrid_state(owner.rows) == original_state
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("progressive_external_target_rows", lambda *_args, **_kwargs: None),
        ("progressive_external_compact_exact_rows", False),
        ("progressive_external_verify_protocol", "drifted-protocol"),
    ],
)
def test_uniform_post_execute_hook_drift_refuses_publication(
    monkeypatch, attribute, replacement
):
    _descriptor, owner, target, _tokens, binding = _binding((2, 2))
    original = canonical_hybrid_state(owner.rows)
    binding.execute()
    monkeypatch.setattr(target.model, attribute, replacement)

    with pytest.raises(ValueError, match="model binding drifted"):
        binding.commit((1, 1))

    assert binding.published is None
    assert canonical_hybrid_state(owner.rows) == original
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_post_execute_provider_drift_refuses_publication_without_base_mutation():
    _descriptor, owner, _target, _tokens, binding = _binding((2, 3))
    original_state = canonical_hybrid_state(owner.rows)
    binding.execute()
    binding.provider.drifted = True
    with pytest.raises(RuntimeError, match="post-execute provider drift"):
        binding.commit((1, 1))
    assert binding.published is None
    assert canonical_hybrid_state(owner.rows) == original_state
    receipt = binding.registry.receipt()
    assert receipt["active_lease"] is False
    assert receipt["poisoned"] is False


def test_live_bridge_device_drift_fails_before_invocation_or_base_mutation(
    monkeypatch,
):
    _descriptor, owner, target, _tokens, binding = _binding((2, 3))
    original = canonical_hybrid_state(owner.rows)
    monkeypatch.setattr(
        target,
        "ast_native_flat_device_attestation",
        lambda: {
            "schema": "fixture",
            "backend": "metal",
            "build": target.identity["build"],
            "device": "mlx-metal:drifted",
            "ready": True,
        },
    )
    with pytest.raises(ValueError, match="device attestation drifted"):
        binding.execute()
    assert target.invocations == 0
    assert canonical_hybrid_state(owner.rows) == original


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("progressive_external_target_rows", lambda *_args, **_kwargs: None),
        ("progressive_external_compact_exact_rows", False),
        ("progressive_external_verify_protocol", "drifted-protocol"),
    ],
)
def test_live_bridge_rejects_row_exact_hook_drift_before_mutation(
    monkeypatch, attribute, replacement
):
    _descriptor, owner, target, _tokens, binding = _binding((2, 2))
    original = canonical_hybrid_state(owner.rows)
    monkeypatch.setattr(target.model, attribute, replacement)

    with pytest.raises(ValueError, match="model binding drifted"):
        binding.execute()

    assert target.invocations == 0
    assert canonical_hybrid_state(owner.rows) == original
    assert binding.registry.receipt()["active_lease"] is False


def test_serving_source_closure_includes_segmented_executor_modules():
    serving_only = (
        "src/mlx2/adapters/external_draft_policy.py",
        "src/mlx2/adapters/qwen38_27b.py",
        "src/mlx2/runtime/ast_live_state.py",
        "src/mlx2/runtime/ast_physical_target.py",
        "src/mlx2/runtime/external_speculative.py",
        "src/mlx2/runtime/progressive_external_verify.py",
        "src/mlx2/runtime/row_exact_verify.py",
    )
    assert SERVING_SOURCE_PATHS == SOURCE_PATHS + serving_only
    provenance = json.loads(
        (
            Path(__file__).parents[1]
            / "provenance/ast-qwen38-hybrid-native-flat-target-20261020.json"
        ).read_text()
    )
    assert set(SERVING_SOURCE_PATHS).issubset(provenance["source_paths"])


def test_real_artifact_unready_metal_fails_before_binding_or_base_mutation():
    descriptor, owner, direct_target = _pair((2,))
    direct_target.model.configure_target_verify_row_exact(True)
    identity = {
        **direct_target.identity,
        "artifact": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
    }
    target = type(direct_target)(
        direct_target.model,
        descriptor=descriptor,
        capture_layers=direct_target.capture_layers,
        identity=identity,
        expected_model_manifest_sha256=direct_target._expected_model_manifest_sha256,
        construction_role="serving",
    )
    original = canonical_hybrid_state(owner.rows)
    provider = _Provider(target)
    discovery, admission = _admission(descriptor)
    with pytest.raises(ValueError, match="real artifact requires ready Metal"):
        Qwen38NativeFlatServingBinding(
            provider,
            rows=owner.rows,
            descriptor=descriptor,
            capture_layers=target.capture_layers,
            flat_inputs=(17, 18),
            session_scope_sha256="1" * 64,
            discovery=discovery,
            admission=admission,
        )
    assert target.invocations == 0
    assert canonical_hybrid_state(owner.rows) == original


def test_provider_installation_is_unavailable_without_exact_git_provenance(
    monkeypatch,
):
    _descriptor, _owner, target = _pair((1,))
    monkeypatch.setattr(
        "mlx2.adapters.qwen38_hybrid_native_flat._git_source_revision",
        lambda: (_ for _ in ()).throw(RuntimeError("not a git checkout")),
    )
    assert install_qwen38_native_flat_physical_target_provider(target.model) is None


def test_provider_live_revision_check_rejects_source_generation_drift(monkeypatch):
    provider = object.__new__(Qwen38NativeFlatPhysicalTargetProvider)
    provider.protocol = "ast-physical-target-provider-v1"
    provider._protocol_binding = provider.protocol
    provider.identity = {"artifact": "fixture"}
    provider._identity_binding = tuple(sorted(provider.identity.items()))
    provider.hard_limits = {"max_lanes": 4}
    provider._hard_limits_binding = tuple(sorted(provider.hard_limits.items()))
    provider.source_revision_token = "a" * 40
    provider._source_revision_binding = "a" * 40
    provider.source_identity = (("src/mlx2/fixture.py", "b" * 64),)
    provider._source_identity_binding = provider.source_identity
    provider._source_generation_binding = (("fixture", 1),)
    monkeypatch.setattr(
        "mlx2.adapters.qwen38_hybrid_native_flat._serving_source_generation",
        lambda: (("fixture", 2),),
    )
    with pytest.raises(RuntimeError, match="binding drifted"):
        provider.validate_revision()


@pytest.mark.parametrize("mutation", ["protocol", "identity", "hard_limits"])
def test_provider_live_revision_check_rejects_public_binding_drift(
    monkeypatch, mutation
):
    provider = object.__new__(Qwen38NativeFlatPhysicalTargetProvider)
    provider.protocol = "ast-physical-target-provider-v1"
    provider._protocol_binding = provider.protocol
    provider.identity = {"artifact": "fixture"}
    provider._identity_binding = tuple(sorted(provider.identity.items()))
    provider.hard_limits = {"max_lanes": 4}
    provider._hard_limits_binding = tuple(sorted(provider.hard_limits.items()))
    provider.source_revision_token = "a" * 40
    provider._source_revision_binding = "a" * 40
    provider.source_identity = (("src/mlx2/fixture.py", "b" * 64),)
    provider._source_identity_binding = provider.source_identity
    provider._source_generation_binding = (("fixture", 1),)
    monkeypatch.setattr(
        "mlx2.adapters.qwen38_hybrid_native_flat._serving_source_generation",
        lambda: (("fixture", 1),),
    )
    if mutation == "protocol":
        provider.protocol = "ast-physical-target-provider-v2"
    elif mutation == "identity":
        provider.identity["artifact"] = "drifted"
    else:
        provider.hard_limits["max_lanes"] = 8
    with pytest.raises(RuntimeError, match="binding drifted"):
        provider.validate_revision()


def test_binding_rejects_foreign_session_admission():
    descriptor, owner, target, tokens, _binding_value = _binding((1, 2))
    discovery, admission = _admission(descriptor)
    with pytest.raises(ValueError, match="live admission differs"):
        Qwen38NativeFlatServingBinding(
            _Provider(target),
            rows=owner.rows,
            descriptor=descriptor,
            capture_layers=target.capture_layers,
            flat_inputs=tokens,
            session_scope_sha256="9" * 64,
            discovery=discovery,
            admission=admission,
        )


@pytest.mark.parametrize(
    "field",
    [
        "source_manifest_sha256",
        "build_manifest_sha256",
        "device_manifest_sha256",
        "owner_capability_manifest_sha256",
        "target_capability_manifest_sha256",
    ],
)
def test_provider_bind_rejects_live_authority_digest_drift(field):
    descriptor, owner, target, tokens, _binding_value = _binding((1, 2))
    provider = object.__new__(Qwen38NativeFlatPhysicalTargetProvider)
    provider.validate_revision = lambda: None
    provider._target = lambda _descriptor, _layers: target
    provider._session_source_manifest = lambda _scope: "4" * 64
    provider.hard_limits = dict(target.hard_limits)
    target_contract = target.ast_native_flat_target_contract()
    discovery, _unused = _admission(descriptor)
    expected = {
        "source_manifest_sha256": "4" * 64,
        "build_manifest_sha256": _manifest(target.identity["build"]),
        "device_manifest_sha256": _manifest(target.identity["device"]),
        "owner_capability_manifest_sha256": _manifest(
            provider.serving_admission_contract()
        ),
        "target_capability_manifest_sha256": _manifest(target_contract),
    }
    drifted = {**expected, field: "f" * 64}
    admission = ast_serving_admission_receipt(
        discovery=discovery,
        session_scope_sha256="1" * 64,
        **drifted,
    )
    with pytest.raises(RuntimeError, match="live admission authority drifted"):
        provider.bind_serving_physical_target(
            rows=owner.rows,
            descriptor=descriptor,
            capture_layers=target.capture_layers,
            flat_inputs=tokens,
            session_scope_sha256="1" * 64,
            discovery=discovery,
            admission=admission,
        )
