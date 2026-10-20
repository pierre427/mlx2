"""Model-neutral semantic discovery for AST target execution.

This module deliberately does not choose a tile, backend, or route.  It
describes only properties that must remain true for every execution using a
capability: the ordinary reference seam, fidelity, private-state transaction
law, supported physical layouts, installed kernel laws and hard capacities.

Widths, crossover thresholds, measured cost curves, attention traffic prices
and module-name allowlists are tuning inputs.  They are named in the receipt
as exclusions so a future self-configuring planner cannot mistake them for an
adapter capability.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA = "mlx2.ast-execution-capability.v1"
OWNER_SCHEMA = "mlx2.ast-transaction-owner.v1"

PADDED_RECTANGULAR = "padded_rectangular"

NON_CAPABILITY_TUNING = (
    "attention_cost_geometry",
    "cost_curves",
    "module_name_allowlists",
    "packed_width_crossover",
    "preferred_terminal_widths",
    "preferred_tile_widths",
)

# APCv2 owns the optional memory-only store. This module only declares which
# deterministic metadata may enter it; it never owns a cache or authorizes a
# backend from a hit.
MEMOIZABLE_METADATA = (
    "canonical_ragged_geometry",
)
FUTURE_MEMO_CANDIDATES = (
    "capability_manifest",
    "executor_binding",
    "exact_proposal_lattice_and_shared_prefix_analysis",
    "legal_backend_candidates",
    "segment_descriptors_and_masks",
    "source_build_device_kernel_identity",
)
MEMO_KEY_FIELDS = (
    "schema",
    "scope_epochs",
    "apc_semantic_epoch",
    "layout_revision_sha256",
    "target_fingerprint",
    "drafter_fingerprint",
    "source_build_device_sha256",
    "policy_sha256",
    "owner_semantic_contract_sha256",
    "capability_manifest_sha256",
    "cohort_geometry_sha256",
)
MEMO_INVALIDATIONS = (
    "transaction_commit",
    "transaction_abort",
    "apcv2_state_or_layout_revision_change",
    "policy_change",
    "source_or_build_change",
    "target_or_drafter_change",
    "session_end",
)
MEMO_EXCLUSIONS = (
    "acceptance_feedback",
    "cost_feedback",
    "planner_timing_samples",
    "scheduler_state",
    "tensor_results",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _callable_identity(value: object) -> str | None:
    if not callable(value):
        return None
    function = getattr(value, "__func__", value)
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    return f"{module}.{qualname}"


def state_contract_from_rows(rows) -> dict:
    """Build a state declaration from explicit cache-operation contracts.

    Cache class names and model families are intentionally absent.  A new
    cache family is unavailable until it publishes the shared operations
    contract consumed here.
    """

    from .cache_operations import cache_operations

    rows = tuple(tuple(row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("AST transaction rows need one nonempty common layer layout")
    contracts = []
    for row in rows:
        for cache in row:
            contracts.append(
                cache_operations(
                    cache, capability="AST execution capability discovery"
                )
            )
    return {
        "state_plane_kinds": sorted(
            {plane.kind for contract in contracts for plane in contract.planes}
        ),
        "state_families": sorted({contract.family for contract in contracts}),
        "state_contract_fingerprints": sorted(
            {contract.fingerprint for contract in contracts}
        ),
        "state_ownership": "independent_request_private_b1_lanes",
    }


def transaction_owner_contract(owner) -> dict | None:
    """Read and validate an owner's explicit AST transaction declaration."""

    provider = getattr(owner, "ast_execution_transaction_contract", None)
    if not callable(provider):
        return None
    try:
        raw = provider()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != OWNER_SCHEMA:
        return None
    required_strings = (
        "state_ownership",
        "commit_law",
        "abort_law",
        "rollback_law",
    )
    if any(not isinstance(raw.get(key), str) or not raw[key] for key in required_strings):
        return None
    layouts = raw.get("supported_layout_classes")
    families = raw.get("state_families")
    planes = raw.get("state_plane_kinds")
    fingerprints = raw.get("state_contract_fingerprints")
    if any(
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        for value in (layouts, families, planes, fingerprints)
    ):
        return None
    operations = raw.get("transaction_operations")
    if not isinstance(operations, Mapping) or any(
        operations.get(name) is not True for name in ("begin", "commit", "abort")
    ):
        return None
    # Unknown layout vocabulary must never become executable by typo.
    known_layouts = {PADDED_RECTANGULAR, "flat_segmented", "depthwise_segmented"}
    if not set(layouts).issubset(known_layouts):
        return None
    return {
        "state_plane_kinds": tuple(sorted(set(planes))),
        "state_families": tuple(sorted(set(families))),
        "state_contract_fingerprints": tuple(sorted(set(fingerprints))),
        "state_ownership": raw["state_ownership"],
        "supported_layout_classes": tuple(dict.fromkeys(layouts)),
        "transaction_operations": {name: True for name in ("begin", "commit", "abort")},
        "commit_law": raw["commit_law"],
        "abort_law": raw["abort_law"],
        "rollback_law": raw["rollback_law"],
    }


def _target_modules(model) -> tuple[tuple[str, object], ...]:
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return ()
    try:
        return tuple(named())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ()


def _kernel_contracts(model) -> tuple[dict, ...]:
    """Discover installed row laws without retaining adapter module names."""

    from mlx import nn

    from . import row_exact_verify
    from .lane import MAX_ROWS as LANE_MAX_ROWS
    from .lane import installer as lane_installer
    from .models import row_exact_qmv as qmv

    aggregate: dict[bytes, dict] = {}
    seen_modules = set()
    for _name, module in _target_modules(model):
        module_identity = id(module)
        if module_identity in seen_modules:
            continue
        seen_modules.add(module_identity)
        if not row_exact_verify.compacts_valid_rows(module):
            continue
        provider = capacity = passes = None
        try:
            if qmv.lane_preparation(module) is not None:
                provider = "row_exact_qmv_lane"
                capacity = int(LANE_MAX_ROWS)
                passes = tuple(
                    qmv.weight_passes(module, rows)
                    for rows in range(1, capacity + 1)
                )
            elif isinstance(module, nn.QuantizedLinear):
                provider = "row_exact_qmv"
                capacity = int(qmv.MAX_ROWS)
                passes = tuple(
                    qmv.weight_passes(module, rows)
                    for rows in range(1, capacity + 1)
                )
            elif lane_installer.module_weight_passes(module, 1) is not None:
                provider = "lane_installer"
                capacity = lane_installer.module_row_capacity(module)
                if capacity is not None:
                    passes = tuple(
                        lane_installer.module_weight_passes(module, rows)
                        for rows in range(1, capacity + 1)
                    )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Inventory is diagnostic.  A malformed or unavailable installed
            # kernel must not disturb the established padded execution path.
            continue
        if provider is None or passes is None or any(value is None for value in passes):
            continue
        weight = getattr(module, "weight", None)
        shape = tuple(int(value) for value in getattr(weight, "shape", ()))
        contract = {
            "provider": provider,
            "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            "weight_shape": shape,
            # Installed inventory is not proof that the target executor
            # reaches this module (resident draft/sidecar modules may share
            # the graph).  Keep the kernel's own capacity visible but never
            # promote it to a route hard bound in this foundation slice.
            "execution_reachability": "installed_unattested",
            "installed_row_capacity": capacity,
            "hard_row_capacity": None,
            "weight_passes": passes,
            "count": 1,
        }
        key = _canonical({key: value for key, value in contract.items() if key != "count"})
        if key in aggregate:
            aggregate[key]["count"] += 1
        else:
            aggregate[key] = contract
    return tuple(aggregate[key] for key in sorted(aggregate))


@dataclass(frozen=True, slots=True)
class ASTExecutionCapability:
    available: bool
    discovery: str
    decline_reasons: tuple[str, ...]
    ordinary_reference_executor: str | None
    target_executor: str | None
    fidelity: str
    terminal_law: str
    state_plane_kinds: tuple[str, ...]
    state_families: tuple[str, ...]
    state_contract_fingerprints: tuple[str, ...]
    state_ownership: str
    transaction_operations: tuple[str, ...]
    commit_law: str
    abort_law: str
    rollback_law: str
    owner_state_layout_classes: tuple[str, ...]
    target_tensor_layout_classes: tuple[str, ...]
    target_tensor_layout_authority: str
    supported_layout_classes: tuple[str, ...]
    hard_max_position_rows: int | None
    kernel_weight_passes: tuple[dict, ...]
    identity: dict

    def receipt(self) -> dict:
        body = {
            "schema": SCHEMA,
            "available": self.available,
            "discovery": self.discovery,
            "decline_reasons": list(self.decline_reasons),
            "ordinary_reference_executor": self.ordinary_reference_executor,
            "target_executor": self.target_executor,
            "fidelity": self.fidelity,
            "terminal_law": self.terminal_law,
            "state_plane_kinds": list(self.state_plane_kinds),
            "state_families": list(self.state_families),
            "state_contract_fingerprints": list(self.state_contract_fingerprints),
            "state_ownership": self.state_ownership,
            "transaction_operations": list(self.transaction_operations),
            "commit_law": self.commit_law,
            "abort_law": self.abort_law,
            "rollback_law": self.rollback_law,
            "owner_state_layout_classes": list(self.owner_state_layout_classes),
            "target_tensor_layout_classes": list(self.target_tensor_layout_classes),
            "target_tensor_layout_authority": self.target_tensor_layout_authority,
            "supported_layout_classes": list(self.supported_layout_classes),
            "hard_max_position_rows": self.hard_max_position_rows,
            "kernel_weight_passes": [dict(item) for item in self.kernel_weight_passes],
            "identity": dict(self.identity),
            "non_capability_tuning": list(NON_CAPABILITY_TUNING),
            "memoization_contract": {
                "authority": "APCv2",
                "implemented": True,
                "default_enabled": False,
                "memory_only": True,
                "memoizable_deterministic_metadata": list(MEMOIZABLE_METADATA),
                "future_candidates_unimplemented": list(FUTURE_MEMO_CANDIDATES),
                "required_key_fields": list(MEMO_KEY_FIELDS),
                "invalidations": list(MEMO_INVALIDATIONS),
                "excluded_mutable_or_tensor_state": list(MEMO_EXCLUSIONS),
                "debug_namespace": {
                    "namespace": "mlx2.apcv2.ast-memo-debug.v1",
                    "authoritative": False,
                    "implemented": True,
                    "default_enabled": False,
                    "session_scoped": True,
                    "token_free": True,
                    "bounded": True,
                    "atomic_append": True,
                    "eviction": [
                        "entry_cap",
                        "byte_cap",
                        "session_ttl",
                        "session_end",
                    ],
                    "identity_fields": list(MEMO_KEY_FIELDS),
                    "shadow_comparison": (
                        "append_caller_supplied_live_geometry_comparison_only"
                    ),
                    "record_fields": [
                        "lookup_input_sha256",
                        "candidate_backends_and_shapes",
                        "selected_plan",
                        "decline_reasons",
                        "memo_outcome",
                        "invalidation_cause",
                        "shadow_comparison",
                    ],
                    "cannot_influence": [
                        "APCv2_prefix_hit_or_promotion",
                        "planner_feedback",
                        "timing_samples",
                        "scheduling",
                        "serving_result",
                    ],
                },
            },
        }
        body["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        return body


def query_ast_execution_capability(
    model,
    transaction_owner,
    *,
    fidelity: str,
    protocol_binding: Mapping[str, Any] | None,
    execution_profile=None,
    artifact_identity: str | None = None,
) -> ASTExecutionCapability:
    """Return immutable semantic discovery, defaulting to padded-only.

    An unknown or malformed owner does not make an alternative layout legal.
    Existing rectangular execution remains available to its current caller;
    the manifest advertises no newly selectable capability.
    """

    owner = transaction_owner_contract(transaction_owner)
    ordinary = _callable_identity(getattr(model, "forward_with_taps", None))
    target = _callable_identity(
        getattr(model, "progressive_external_target_rows", None)
    )
    kernels = _kernel_contracts(model)
    from .progressive_external_verify import (
        PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA,
        PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
        PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
        PROGRESSIVE_FIDELITY_EXACT,
        PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    binding_schema = binding_algorithm = binding_fidelity = None
    qualification_gate_schema = None
    source_hashes = {}
    target_protocol = None
    if isinstance(protocol_binding, Mapping):
        binding_schema = protocol_binding.get("schema")
        binding_algorithm = protocol_binding.get("algorithm")
        binding_fidelity = protocol_binding.get("fidelity")
        target_protocol = protocol_binding.get("target_protocol")
        qualification_gate_schema = protocol_binding.get(
            "qualification_gate_schema"
        )
        candidate_hashes = protocol_binding.get("source_hashes")
        if isinstance(candidate_hashes, Mapping) and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for key, value in candidate_hashes.items()
        ) and candidate_hashes:
            source_hashes = dict(sorted(candidate_hashes.items()))
    expected_protocol = {
        PROGRESSIVE_FIDELITY_EXACT: PROGRESSIVE_TARGET_PROTOCOL_VERSION,
        PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED: (
            PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
        ),
    }.get(fidelity)
    expected_gate = (
        PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA
        if fidelity == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
        else None
    )
    protocol_binding_valid = all(
        (
            binding_schema == "mlx2.progressive-external-target-binding.v1",
            binding_algorithm == PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
            binding_fidelity == fidelity,
            expected_protocol is not None,
            target_protocol == expected_protocol,
            qualification_gate_schema == expected_gate,
            bool(source_hashes),
        )
    )
    device = getattr(execution_profile, "device", None)
    build = getattr(execution_profile, "build", None)
    discovered_caps = [
        item["hard_row_capacity"]
        for item in kernels
        if item["hard_row_capacity"] is not None
    ]
    hard_capacity = min(discovered_caps) if discovered_caps else None
    identity = {
        "artifact": artifact_identity,
        "binding_schema": binding_schema,
        "algorithm": binding_algorithm,
        "fidelity": binding_fidelity,
        "target_protocol": target_protocol,
        "qualification_gate_schema": qualification_gate_schema,
        "source_hashes": source_hashes,
        "build": build,
        "device": device,
    }
    identity_complete = all(
        isinstance(identity[key], str) and identity[key]
        for key in ("artifact", "target_protocol", "build", "device")
    ) and bool(source_hashes)
    decline_reasons = []
    if owner is None:
        decline_reasons.append("transaction_owner_contract_unavailable")
    if ordinary is None:
        decline_reasons.append("ordinary_reference_executor_unavailable")
    if not protocol_binding_valid:
        decline_reasons.append("progressive_protocol_binding_invalid")
    if not identity_complete:
        decline_reasons.append("source_build_device_artifact_identity_incomplete")
    owner_state_layouts = (
        tuple(owner["supported_layout_classes"]) if owner is not None else ()
    )
    # The state owner can attest transaction and cache legality, but cannot
    # authorize model tensor math.  Until a generic target-executor layout
    # attestation is implemented, only the established rectangular progressive
    # binding plus ordinary reference seam is tensor-authoritative.  Future
    # selectable layouts must be the intersection of these two independent
    # declarations.
    target_tensor_layouts = (
        (PADDED_RECTANGULAR,)
        if ordinary is not None and protocol_binding_valid
        else ()
    )
    supported_layouts = tuple(
        layout for layout in owner_state_layouts if layout in target_tensor_layouts
    )
    if owner is not None and not supported_layouts:
        decline_reasons.append("state_and_tensor_layout_contracts_do_not_intersect")
    available = not decline_reasons
    if owner is None:
        owner = {
            "state_plane_kinds": (),
            "state_families": (),
            "state_contract_fingerprints": (),
            "state_ownership": "unknown",
            "supported_layout_classes": (PADDED_RECTANGULAR,),
            "transaction_operations": {},
            "commit_law": "unknown",
            "abort_law": "unknown",
            "rollback_law": "unknown",
        }
    identity["executor_binding_sha256"] = hashlib.sha256(
        _canonical(
            {
                **identity,
                "ordinary_reference_executor": ordinary,
                "target_executor": target,
                "fidelity": fidelity,
                "state_contract_fingerprints": owner[
                    "state_contract_fingerprints"
                ],
                "owner_state_layout_classes": owner_state_layouts,
                "target_tensor_layout_classes": target_tensor_layouts,
                "supported_layout_classes": supported_layouts,
                "kernel_weight_passes": kernels,
            }
        )
    ).hexdigest()
    return ASTExecutionCapability(
        available=available,
        discovery="semantic_contract" if available else "unknown_fail_closed",
        decline_reasons=tuple(decline_reasons),
        ordinary_reference_executor=ordinary,
        target_executor=target,
        fidelity=fidelity,
        terminal_law=(
            "k_proposals:nonfinal_anchor_plus_first_k_minus_one_inputs="
            "k_rows;final_anchor_plus_all_k_inputs=k_plus_one_rows"
        ),
        state_plane_kinds=tuple(owner["state_plane_kinds"]),
        state_families=tuple(owner["state_families"]),
        state_contract_fingerprints=tuple(owner["state_contract_fingerprints"]),
        state_ownership=owner["state_ownership"],
        transaction_operations=tuple(
            name for name in ("begin", "commit", "abort")
            if owner["transaction_operations"].get(name) is True
        ),
        commit_law=owner["commit_law"],
        abort_law=owner["abort_law"],
        rollback_law=owner["rollback_law"],
        owner_state_layout_classes=owner_state_layouts,
        target_tensor_layout_classes=target_tensor_layouts,
        target_tensor_layout_authority=(
            "current_progressive_binding_padded_contract"
            if target_tensor_layouts
            else "unavailable"
        ),
        supported_layout_classes=(
            supported_layouts if owner_state_layouts else (PADDED_RECTANGULAR,)
        ),
        hard_max_position_rows=hard_capacity,
        kernel_weight_passes=kernels,
        identity=identity,
    )


__all__ = [
    "NON_CAPABILITY_TUNING",
    "OWNER_SCHEMA",
    "PADDED_RECTANGULAR",
    "ASTExecutionCapability",
    "query_ast_execution_capability",
    "state_contract_from_rows",
    "transaction_owner_contract",
]
