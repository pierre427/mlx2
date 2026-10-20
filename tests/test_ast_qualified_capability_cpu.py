from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mlx2.runtime.ast_qualified_capability import (
    DIRECT_PROBE_SCHEMA,
    QUALIFICATION_DOMAIN_SCHEMA,
    ASTQualifiedPhysicalTargetCapability,
    discover_ast_qualified_capability,
    load_ast_qualified_capability_evidence,
    mint_ast_qualified_capability,
    normalize_ast_qualified_capability_discovery,
    seal_ast_direct_probe_receipt,
    unavailable_ast_qualified_capability_discovery,
    validate_ast_qualified_session,
)


def _digest(character: str) -> str:
    return character * 64


def _binding(**changes):
    result = {
        "source_revision": "a" * 40,
        "source_manifest_sha256": _digest("b"),
        "build": "mlx-0.32.4",
        "device": "mlx-metal:applegpu_g17s:Apple M5 Max",
        "artifact": f"artifact@sha256:{_digest('c')}",
        "profile_sha256": _digest("d"),
        "provider_binding_sha256": _digest("e"),
        "owner_capability_manifest_sha256": _digest("f"),
        "target_capability_manifest_sha256": _digest("1"),
        "executor_protocol": "whole-layer-ragged-v1",
    }
    result.update(changes)
    return result


def _probe(**changes):
    body = {
        "schema": DIRECT_PROBE_SCHEMA,
        "execution_scope": "whole_transformer_layer_ragged_native_flat",
        "fidelity": "exact",
        "oracle": "ordinary_serial_b1s1",
        "numerical_equivalence": "bit_exact",
        "session_binding": _binding(),
        "layout_classes": ["flat_segmented"],
        "hard_limits": {
            "max_lanes": 4,
            "max_segment_rows": 4,
            "max_total_rows": 12,
        },
        "qualification_domain": {
            "schema": QUALIFICATION_DOMAIN_SCHEMA,
            "request_classes": [
                {
                    "query_lengths": [1],
                    "base_offsets": [0],
                    "capacities": [0],
                    "allocation_steps": [8],
                    "capture_schema_sha256": _digest("2"),
                    "state_schema_sha256": _digest("3"),
                },
                {
                    "query_lengths": [4, 3, 2, 1],
                    "base_offsets": [0, 8, 9, 10],
                    "capacities": [0, 16, 12, 20],
                    "allocation_steps": [8, 8, 8, 8],
                    "capture_schema_sha256": _digest("2"),
                    "state_schema_sha256": _digest("3"),
                },
            ],
        },
        "coverage": {
            "outputs": True,
            "features": True,
            "prepared_state": True,
            "continuation": True,
            "zero_commit": True,
            "partial_commit": True,
            "full_commit": True,
            "abort_rollback": True,
            "lane_isolation": True,
            "ordinary_fallback": True,
        },
        "passed": True,
        "performance_claim": False,
    }
    body.update(changes)
    return seal_ast_direct_probe_receipt(body)


class _Provider:
    def __init__(self, capability):
        self.capability = capability
        self.revision = 0
        self.validations = 0
        self.capability_reads = 0

    def validate_revision(self):
        self.validations += 1
        if self.revision:
            raise RuntimeError("stale")

    def ast_qualified_physical_target_capability(self):
        self.capability_reads += 1
        return self.capability


def test_unavailable_discovery_preserves_only_bounded_decline_reasons():
    bounded = unavailable_ast_qualified_capability_discovery(
        "qualification_warmup_authority_failed"
    )
    assert normalize_ast_qualified_capability_discovery(bounded) == bounded

    foreign = {**bounded, "decline_reason": "private exception contents"}
    assert normalize_ast_qualified_capability_discovery(foreign) == (
        unavailable_ast_qualified_capability_discovery()
    )
    with pytest.raises(ValueError, match="decline reason"):
        unavailable_ast_qualified_capability_discovery("private exception contents")


def _discover(provider, **changes):
    arguments = {
        "provider_getter": lambda: provider,
        "session_binding": _binding(),
        "layout_class": "flat_segmented",
        "query_lengths": (4, 3, 2, 1),
        "base_offsets": (0, 8, 9, 10),
        "capacities": (0, 16, 12, 20),
        "allocation_steps": (8, 8, 8, 8),
        "capture_schema_sha256": _digest("2"),
        "state_schema_sha256": _digest("3"),
    }
    arguments.update(changes)
    return discover_ast_qualified_capability(provider, **arguments)


def test_direct_probe_mints_immutable_capability_and_discovers_without_probe():
    capability = mint_ast_qualified_capability(_probe())
    provider = _Provider(capability)

    receipt = _discover(provider)

    assert receipt["available"] is True
    assert receipt["qualified"] is True
    assert receipt["eligible"] is False
    assert receipt["selected"] is False
    assert receipt["default"] is False
    assert receipt["observed_used"] is False
    assert receipt["performance_claim"] is False
    assert receipt["apcv2_authoritative"] is False
    assert receipt["capability_manifest_sha256"] == capability.manifest_sha256
    assert provider.validations == 2
    assert provider.capability_reads == 1
    with pytest.raises(FrozenInstanceError):
        capability.coverage_sha256 = _digest("2")


def test_runner_envelope_loads_only_when_embedded_capability_matches():
    direct = _probe()
    capability = mint_ast_qualified_capability(direct)
    envelope = {
        **direct,
        "verdict": "pass_exact_direct_probe",
        "cleanup_completed": True,
        "source_revision": "a" * 40,
        "qualified_capability": capability.receipt(),
    }
    assert load_ast_qualified_capability_evidence(envelope) == capability
    envelope["qualified_capability"] = {
        **capability.receipt(),
        "qualified": False,
    }
    with pytest.raises(ValueError, match="embedded capability differs"):
        load_ast_qualified_capability_evidence(envelope)


def test_runner_envelope_missing_direct_field_fails_as_value_error():
    direct = _probe()
    capability = mint_ast_qualified_capability(direct)
    envelope = {
        **direct,
        "verdict": "pass_exact_direct_probe",
        "cleanup_completed": True,
        "source_revision": "a" * 40,
        "qualified_capability": capability.receipt(),
    }
    envelope.pop("coverage")
    with pytest.raises(ValueError, match="runner evidence is incomplete"):
        load_ast_qualified_capability_evidence(envelope)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_revision", "2" * 40),
        ("source_manifest_sha256", _digest("2")),
        ("build", "mlx-other"),
        ("device", "mlx-metal:other"),
        ("artifact", f"other@sha256:{_digest('2')}"),
        ("profile_sha256", _digest("2")),
        ("provider_binding_sha256", _digest("2")),
        ("owner_capability_manifest_sha256", _digest("2")),
        ("target_capability_manifest_sha256", _digest("2")),
        ("executor_protocol", "other-v1"),
    ],
)
def test_session_binding_drift_fails_closed(field, value):
    provider = _Provider(mint_ast_qualified_capability(_probe()))
    receipt = _discover(provider, session_binding=_binding(**{field: value}))
    assert receipt["available"] is False
    assert receipt["qualified"] is False
    assert receipt["capability_manifest_sha256"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"layout_class": "padded_rectangular"},
        {"query_lengths": (4, 3, 2, 2)},
        {"base_offsets": (0, 8, 9, 11)},
        {"capacities": (0, 16, 12, 21)},
        {"allocation_steps": (8, 8, 8, 16)},
        {"capture_schema_sha256": _digest("4")},
        {"state_schema_sha256": _digest("4")},
    ],
)
def test_request_geometry_outside_qualified_domain_fails_closed(changes):
    provider = _Provider(mint_ast_qualified_capability(_probe()))
    assert _discover(provider, **changes)["available"] is False


def test_provider_replacement_and_revision_drift_fail_closed():
    capability = mint_ast_qualified_capability(_probe())
    provider = _Provider(capability)
    replacement = _Provider(capability)
    assert (
        _discover(provider, provider_getter=lambda: replacement)["available"] is False
    )

    provider.revision = 1
    assert _discover(provider)["available"] is False


def test_tampered_or_incomplete_probe_cannot_mint_capability():
    tampered = _probe()
    tampered["passed"] = False
    with pytest.raises(ValueError, match="integrity"):
        mint_ast_qualified_capability(tampered)

    incomplete = _probe()
    body = dict(incomplete)
    body.pop("receipt_sha256")
    body["coverage"] = dict(body["coverage"])
    body["coverage"].pop("continuation")
    with pytest.raises(ValueError, match="coverage"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(body))


def test_session_validator_is_metadata_only_and_rejects_unqualified_geometry():
    capability = mint_ast_qualified_capability(_probe())
    validate_ast_qualified_session(
        capability,
        _binding(),
        layout_class="flat_segmented",
        query_lengths=(1,),
        base_offsets=(0,),
        capacities=(0,),
        allocation_steps=(8,),
        capture_schema_sha256=_digest("2"),
        state_schema_sha256=_digest("3"),
    )
    with pytest.raises(ValueError, match="request class"):
        validate_ast_qualified_session(
            capability,
            _binding(),
            layout_class="flat_segmented",
            query_lengths=(4, 4, 4, 4),
            base_offsets=(0, 8, 9, 10),
            capacities=(0, 16, 12, 20),
            allocation_steps=(8, 8, 8, 8),
            capture_schema_sha256=_digest("2"),
            state_schema_sha256=_digest("3"),
        )

    body = dict(_probe())
    body.pop("receipt_sha256")
    body["hard_limits"] = {
        "max_lanes": 4,
        "max_segment_rows": 4,
        "max_total_rows": 9,
    }
    with pytest.raises(ValueError, match="hard limits"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(body))


def test_probe_schema_and_authority_are_strict():
    body = _probe()
    raw = dict(body)
    raw.pop("receipt_sha256")
    raw["oracle"] = "self_reference"
    with pytest.raises(ValueError, match="authority"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(raw))

    raw = dict(_probe())
    raw.pop("receipt_sha256")
    raw["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(raw))

    raw = dict(_probe())
    raw.pop("receipt_sha256")
    raw["layout_classes"] = ["flat_segmented", "other_layout"]
    with pytest.raises(ValueError, match="layouts"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(raw))


def test_direct_construction_cannot_bypass_minting_authority():
    capability = mint_ast_qualified_capability(_probe())
    with pytest.raises(ValueError, match="authority-minted"):
        ASTQualifiedPhysicalTargetCapability(
            session_binding=capability.session_binding,
            layout_classes=capability.layout_classes,
            hard_limits=capability.hard_limits,
            qualified_request_classes=capability.qualified_request_classes,
            coverage_sha256=_digest("4"),
            direct_probe_receipt_sha256=_digest("5"),
            _seal=object(),
        )

    raw = dict(_probe())
    raw.pop("receipt_sha256")
    raw["performance_claim"] = True
    with pytest.raises(ValueError, match="authority"):
        mint_ast_qualified_capability(seal_ast_direct_probe_receipt(raw))
