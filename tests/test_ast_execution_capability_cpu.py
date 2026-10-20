"""CPU-only semantic discovery for self-configuring AST execution."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.ast_execution_capability import (
    NON_CAPABILITY_TUNING,
    OWNER_SCHEMA,
    PADDED_RECTANGULAR,
    query_ast_execution_capability,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.row_exact_components import (
    RowExactQuantizedLinearMixin,
    row_exact_subclass,
)
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    ProgressiveTargetRows,
    execute_progressive_target_rows,
)
from mlx2.runtime.ragged_verify_layout import (
    RaggedVerifyLayout,
    padded_capabilities,
)
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows


def _binding():
    return {
        "schema": "mlx2.progressive-external-target-binding.v1",
        "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
        "target_protocol": PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        "fidelity": PROGRESSIVE_FIDELITY_EXACT,
        "source_hashes": {"src/mlx2/runtime/test-fixture.py": "a" * 64},
        "qualification_gate_schema": None,
    }


def _profile():
    return SimpleNamespace(
        binding="profile-identity",
        build="cpu-test-build",
        device="cpu",
        max_position_rows=32,
    )


class _Target:
    progressive_external_target_modules = None

    def named_modules(self):
        return ()

    def forward_with_taps(self, tokens, caches, capture_layers):
        del caches, capture_layers
        return tokens, ()


def _query(owner, model=None):
    return query_ast_execution_capability(
        _Target() if model is None else model,
        owner,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
        protocol_binding=_binding(),
        execution_profile=_profile(),
        artifact_identity="fixture@sha256:" + "b" * 64,
    )


def test_pure_attention_owner_publishes_semantics_not_tuning():
    capability = _query(SegmentedKVRows([[KVCache()], [KVCache()]]))
    receipt = capability.receipt()

    assert receipt["available"] is True
    assert receipt["state_families"] == ["kv"]
    assert receipt["state_ownership"] == "independent_request_private_b1_lanes"
    assert receipt["transaction_operations"] == ["begin", "commit", "abort"]
    assert receipt["owner_state_layout_classes"] == [PADDED_RECTANGULAR]
    assert receipt["target_tensor_layout_classes"] == [PADDED_RECTANGULAR]
    assert receipt["supported_layout_classes"] == [PADDED_RECTANGULAR]
    assert receipt["hard_max_position_rows"] is None
    assert receipt["non_capability_tuning"] == sorted(NON_CAPABILITY_TUNING)
    assert "preferred_tile_widths" not in receipt["identity"]
    assert receipt == capability.receipt()
    assert len(receipt["manifest_sha256"]) == 64


def test_installed_kernel_inventory_does_not_claim_execution_reachability():
    class InventoryTarget(_Target):
        def __init__(self):
            self.projection = nn.QuantizedLinear(
                64, 64, bias=False, group_size=64, bits=4
            )
            self.projection.__class__ = row_exact_subclass(
                RowExactQuantizedLinearMixin, nn.QuantizedLinear
            )
        def named_modules(self):
            return (("adapter_name_is_not_a_capability", self.projection),)

    receipt = _query(
        SegmentedKVRows([[KVCache()]]), model=InventoryTarget()
    ).receipt()
    assert receipt["hard_max_position_rows"] is None
    assert len(receipt["kernel_weight_passes"]) == 1
    kernel = receipt["kernel_weight_passes"][0]
    assert kernel["provider"] == "row_exact_qmv"
    assert kernel["execution_reachability"] == "installed_unattested"
    assert kernel["installed_row_capacity"] == 32
    assert kernel["hard_row_capacity"] is None
    assert len(kernel["weight_passes"]) == 32
    assert "adapter_name_is_not_a_capability" not in str(kernel)


def test_qmv_lane_inventory_uses_the_lane_capacity_and_complete_curve(monkeypatch):
    from mlx2.runtime.lane import MAX_ROWS as LANE_MAX_ROWS
    from mlx2.runtime.models import row_exact_qmv as qmv

    class LaneInventoryTarget(_Target):
        def __init__(self):
            self.projection = nn.QuantizedLinear(
                64, 64, bias=False, group_size=64, bits=4
            )

        def named_modules(self):
            return (("projection", self.projection),)

    model = LaneInventoryTarget()
    model.projection.__class__ = row_exact_subclass(
        RowExactQuantizedLinearMixin, nn.QuantizedLinear
    )
    monkeypatch.setattr(qmv, "lane_preparation", lambda module: object())
    monkeypatch.setattr(qmv, "weight_passes", lambda module, rows: rows)

    receipt = _query(SegmentedKVRows([[KVCache()]]), model=model).receipt()
    kernel = receipt["kernel_weight_passes"][0]
    assert kernel["provider"] == "row_exact_qmv_lane"
    assert kernel["installed_row_capacity"] == LANE_MAX_ROWS == 128
    assert len(kernel["weight_passes"]) == LANE_MAX_ROWS
    assert kernel["weight_passes"][-1] == LANE_MAX_ROWS


def test_installed_inventory_deduplicates_module_aliases():
    class AliasedTarget(_Target):
        def __init__(self):
            self.projection = nn.QuantizedLinear(
                64, 64, bias=False, group_size=64, bits=4
            )
            self.projection.__class__ = row_exact_subclass(
                RowExactQuantizedLinearMixin, nn.QuantizedLinear
            )

        def named_modules(self):
            return (
                ("first_alias", self.projection),
                ("second_alias", self.projection),
            )

    receipt = _query(
        SegmentedKVRows([[KVCache()]]), model=AliasedTarget()
    ).receipt()
    assert len(receipt["kernel_weight_passes"]) == 1
    assert receipt["kernel_weight_passes"][0]["count"] == 1


def test_hybrid_owner_reports_both_state_planes_and_exact_rollback():
    rows = [
        [KVCache(), ArraysCache(1)],
        [KVCache(), ArraysCache(1)],
    ]
    receipt = _query(HybridVerifyRows(rows)).receipt()

    assert receipt["available"] is True
    assert set(receipt["state_families"]) == {"kv", "recurrent"}
    assert "gdn_recurrent" in receipt["state_plane_kinds"]
    assert receipt["rollback_law"] == "kv_trim_and_recurrent_record_restore"
    assert receipt["commit_law"] == "accepted_verified_input_prefix_per_lane"


def test_unknown_owner_fails_closed_to_existing_padded_layout():
    class UnknownOwner:
        def begin(self, lengths):  # pragma: no cover - discovery must not call it
            raise AssertionError(lengths)

    receipt = _query(UnknownOwner()).receipt()

    assert receipt["available"] is False
    assert receipt["discovery"] == "unknown_fail_closed"
    assert receipt["state_ownership"] == "unknown"
    assert receipt["transaction_operations"] == []
    assert receipt["supported_layout_classes"] == [PADDED_RECTANGULAR]


def test_malformed_owner_cannot_publish_a_new_layout():
    class MalformedOwner:
        def ast_execution_transaction_contract(self):
            return {
                "schema": OWNER_SCHEMA,
                "state_plane_kinds": ["kv"],
                "state_families": ["kv"],
                "state_contract_fingerprints": ["fixture"],
                "state_ownership": "private",
                "supported_layout_classes": ["native_by_typo"],
                "transaction_operations": {"begin": True, "commit": True, "abort": True},
                "commit_law": "prefix",
                "abort_law": "restore",
                "rollback_law": "exact",
            }

    receipt = _query(MalformedOwner()).receipt()
    assert receipt["available"] is False
    assert receipt["supported_layout_classes"] == [PADDED_RECTANGULAR]


def test_owner_cannot_authorize_tensor_layout_without_executor_attestation():
    class FutureOwner:
        def ast_execution_transaction_contract(self):
            return {
                "schema": OWNER_SCHEMA,
                "state_plane_kinds": ["kv"],
                "state_families": ["kv"],
                "state_contract_fingerprints": ["fixture"],
                "state_ownership": "private",
                "supported_layout_classes": [
                    PADDED_RECTANGULAR,
                    "flat_segmented",
                ],
                "transaction_operations": {
                    "begin": True,
                    "commit": True,
                    "abort": True,
                },
                "commit_law": "prefix",
                "abort_law": "restore",
                "rollback_law": "exact",
            }

    receipt = _query(FutureOwner()).receipt()
    assert receipt["available"] is True
    assert receipt["owner_state_layout_classes"] == [
        PADDED_RECTANGULAR,
        "flat_segmented",
    ]
    assert receipt["target_tensor_layout_classes"] == [PADDED_RECTANGULAR]
    assert receipt["supported_layout_classes"] == [PADDED_RECTANGULAR]
    assert receipt["target_tensor_layout_authority"] == (
        "current_progressive_binding_padded_contract"
    )


def test_broken_discovery_provider_cannot_disturb_padded_execution():
    class BrokenOwner:
        def ast_execution_transaction_contract(self):
            raise RuntimeError("diagnostic provider is unavailable")

    receipt = _query(BrokenOwner()).receipt()
    assert receipt["available"] is False
    assert receipt["decline_reasons"] == [
        "transaction_owner_contract_unavailable"
    ]
    assert receipt["supported_layout_classes"] == [PADDED_RECTANGULAR]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "mlx2.progressive-external-target-binding.v0"),
        ("algorithm", "different-algorithm"),
        ("fidelity", "near_tie_bounded"),
        ("target_protocol", "near-tie-bounded-b1s1-cache-state-v1"),
        ("qualification_gate_schema", "unexpected-gate"),
        ("source_hashes", {"fixture.py": "not-a-sha256"}),
    ],
)
def test_malformed_or_mismatched_protocol_binding_declines(field, value):
    binding = _binding()
    binding[field] = value
    capability = query_ast_execution_capability(
        _Target(),
        SegmentedKVRows([[KVCache()]]),
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
        protocol_binding=binding,
        execution_profile=_profile(),
        artifact_identity="fixture@sha256:" + "b" * 64,
    ).receipt()

    assert capability["available"] is False
    assert "progressive_protocol_binding_invalid" in capability["decline_reasons"]


def test_manifest_defines_default_off_apcv2_memo_and_debug_boundaries():
    receipt = _query(SegmentedKVRows([[KVCache()]])).receipt()
    memo = receipt["memoization_contract"]

    assert memo["authority"] == "APCv2"
    assert memo["implemented"] is True
    assert memo["default_enabled"] is False
    assert memo["memory_only"] is True
    assert {
        "scope_epochs",
        "apc_semantic_epoch",
        "layout_revision_sha256",
        "owner_semantic_contract_sha256",
        "capability_manifest_sha256",
        "cohort_geometry_sha256",
    } <= set(memo["required_key_fields"])
    assert {"acceptance_feedback", "cost_feedback", "tensor_results"} <= set(
        memo["excluded_mutable_or_tensor_state"]
    )
    debug = memo["debug_namespace"]
    assert debug["authoritative"] is False and debug["implemented"] is True
    assert debug["default_enabled"] is False
    assert debug["namespace"] == "mlx2.apcv2.ast-memo-debug.v1"
    assert debug["session_scoped"] and debug["token_free"] and debug["atomic_append"]
    assert {"entry_cap", "byte_cap", "session_ttl", "session_end"} == set(
        debug["eviction"]
    )
    assert debug["identity_fields"] == memo["required_key_fields"]
    assert {"candidate_backends_and_shapes", "memo_outcome", "invalidation_cause"} <= set(
        debug["record_fields"]
    )
    assert debug["shadow_comparison"] == (
        "append_caller_supplied_live_geometry_comparison_only"
    )
    assert "serving_result" in debug["cannot_influence"]


def test_canonical_ragged_geometry_is_token_and_lane_identity_free():
    first = RaggedVerifyLayout.from_draft_depths((7, 8), (0, 2))
    same = RaggedVerifyLayout.from_draft_depths((70, 80), (0, 2))
    changed = RaggedVerifyLayout.from_draft_depths((7, 8), (1, 2))

    assert first.geometry_sha256 == same.geometry_sha256
    assert first.geometry_sha256 != changed.geometry_sha256
    receipt = first.receipt(first.select_backend(padded_capabilities()))
    assert receipt["geometry_sha256"] == first.geometry_sha256
    assert "tokens" not in receipt


def test_opt_in_stage_rows_record_full_and_compacted_geometry():
    from mlx2.runtime import row_exact_verify as rev

    class CompactTarget(_Target):
        progressive_external_compact_exact_rows = True

        def progressive_external_target_rows(self, tokens, caches, capture_layers):
            del caches, capture_layers
            values = mx.zeros((*tokens.shape, 4))
            record = rev.Window(int(values.size))
            with rev.window(record):
                compacted = rev.compact_rows(values, stage="projections")
                assert compacted is not None
                compacted.restore(compacted.values)
            return ProgressiveTargetRows(
                logits=values,
                features=(),
                protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
                row_exact=True,
                detail={},
            )

    tokens = mx.array([[1, 1, 1], [2, 2, 2]])
    _logits, _features, detail = execute_progressive_target_rows(
        CompactTarget(),
        tokens,
        [],
        (),
        protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        fidelity=PROGRESSIVE_FIDELITY_EXACT,
        input_lengths=[1, 3],
        include_stage_rows=True,
    )
    stages = detail["stage_rows"]["stages"]
    assert stages["target.forward"] == {
        "calls": 1,
        "logical_rows": 4,
        "physical_rows": 6,
        "padding_rows": 2,
    }
    assert stages["projections"] == {
        "calls": 1,
        "logical_rows": 4,
        "physical_rows": 6,
        "padding_rows": 2,
    }
