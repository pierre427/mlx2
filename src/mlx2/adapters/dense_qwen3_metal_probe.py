"""Default-off exact-artifact Metal correctness probe for dense Qwen3 AST.

This is a direct probe boundary, not a serving route or performance harness.
It admits only the separately labelled Metal flat-segment protocol, computes
the ordinary serial oracle before acquiring the cache transaction, and
publishes accepted KV prefixes only after exact logits, taps, and state checks.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx

from ..runtime.flat_segmented_ast import (
    METAL_EXECUTION_SCOPE,
    FlatSegmentDescriptor,
    FlatSegmentError,
    FlatSegmentUnsupported,
    query_metal_flat_segment_admission,
)
from ..runtime.flat_segmented_kv import (
    MetalFlatSegmentedKVRows,
    canonical_kv_state,
)
from .dense_qwen3_flat import DenseQwen3MetalFlatSegmentTarget

PROBE_SCHEMA = "mlx2.ast-dense-qwen3-metal-exact-artifact-probe.v1"


def _array_exact(actual, expected) -> bool:
    return (
        isinstance(actual, mx.array)
        and isinstance(expected, mx.array)
        and actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and bool(mx.array_equal(actual, expected).item())
    )


def _outputs_exact(actual, expected) -> bool:
    return len(actual) == len(expected) and all(
        _array_exact(actual_logits, expected_logits)
        and _array_exact(actual_taps, expected_taps)
        for (actual_logits, actual_taps), (expected_logits, expected_taps) in zip(
            actual,
            expected,
            strict=True,
        )
    )


def _ordinary_serial_oracles(
    target: DenseQwen3MetalFlatSegmentTarget,
    owner: MetalFlatSegmentedKVRows,
    descriptor: FlatSegmentDescriptor,
    tokens: tuple[int, ...],
    accepted_lengths: tuple[int, ...],
    continuation_tokens: tuple[int, ...],
):
    model = target.model
    base_rows = copy.deepcopy(owner.rows)
    outputs = []
    for lane in range(descriptor.lane_count):
        caches = copy.deepcopy(base_rows[lane])
        for token in tokens[descriptor.lane_slice(lane)]:
            logits, taps = model.forward_with_taps(
                mx.array([[token]], dtype=mx.int32),
                caches,
                target.capture_layers,
            )
            mx.eval(logits, taps)
            outputs.append((logits[0, 0], taps[0, 0]))

    committed_rows = copy.deepcopy(base_rows)
    for lane, count in enumerate(accepted_lengths):
        for token in tokens[descriptor.lane_slice(lane)][:count]:
            mx.eval(
                model(
                    mx.array([[token]], dtype=mx.int32),
                    cache=committed_rows[lane],
                )
            )
    committed_state = canonical_kv_state(committed_rows)
    continuation = []
    for lane, token in enumerate(continuation_tokens):
        logits, taps = model.forward_with_taps(
            mx.array([[token]], dtype=mx.int32),
            copy.deepcopy(committed_rows[lane]),
            target.capture_layers,
        )
        mx.eval(logits, taps)
        continuation.append((logits[0, 0], taps[0, 0]))
    return tuple(outputs), committed_state, tuple(continuation)


@dataclass(frozen=True, slots=True)
class DenseQwen3MetalProbeResult:
    """Token-free result of one bounded exact correctness unit."""

    receipt: dict[str, object]


def run_dense_qwen3_metal_flat_probe(
    owner: MetalFlatSegmentedKVRows,
    target: DenseQwen3MetalFlatSegmentTarget,
    descriptor: FlatSegmentDescriptor,
    flat_tokens: Sequence[int],
    accepted_lengths: Sequence[int],
    continuation_tokens: Sequence[int],
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
) -> DenseQwen3MetalProbeResult:
    """Run one direct Metal AST unit against an ordinary B1/S1 oracle."""

    if type(owner) is not MetalFlatSegmentedKVRows:
        raise TypeError("Metal probe requires its exact flat KV owner")
    if type(target) is not DenseQwen3MetalFlatSegmentTarget:
        raise TypeError("Metal probe requires its exact dense Qwen3 target")
    if not isinstance(descriptor, FlatSegmentDescriptor):
        raise TypeError("Metal probe requires a validated descriptor")
    if target.invocations != 0:
        raise FlatSegmentUnsupported("Metal probe target must be fresh")
    tokens = tuple(flat_tokens)
    accepted = tuple(accepted_lengths)
    continuation = tuple(continuation_tokens)
    if (
        len(tokens) != descriptor.logical_rows
        or any(type(token) is not int or token < 0 for token in tokens)
        or len(accepted) != descriptor.lane_count
        or any(type(value) is not int for value in accepted)
        or any(
            value < 1 or value > length
            for value, length in zip(
                accepted,
                descriptor.query_lengths,
                strict=True,
            )
        )
        or len(continuation) != descriptor.lane_count
        or any(type(token) is not int or token < 0 for token in continuation)
    ):
        raise FlatSegmentError("Metal probe input geometry differs")

    admission = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
    )
    if not admission.available:
        raise FlatSegmentUnsupported(
            "Metal flat probe admission declined: "
            + ",".join(admission.decline_reasons)
        )

    # The complete ordinary oracle is evaluated before the exclusive owner
    # lease. It can neither observe nor perturb the live transaction.
    expected_outputs, expected_state, expected_continuation = (
        _ordinary_serial_oracles(
            target,
            owner,
            descriptor,
            tokens,
            accepted,
            continuation,
        )
    )

    transaction = None
    published = False
    try:
        transaction = owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
        execution = target.execute_flat_segments(tokens, descriptor, transaction)
        if (
            type(execution.invocation_count) is not int
            or execution.invocation_count != 1
            or execution.geometry_sha256 != descriptor.geometry_sha256
            or target.invocations != 1
        ):
            raise FlatSegmentError("Metal probe target invocation contract differs")
        if not _outputs_exact(execution.outputs, expected_outputs):
            raise FlatSegmentError(
                "Metal probe differs from ordinary full-logit/tap oracle"
            )
        prepared = transaction.prepare_commit(accepted)
        if prepared.state != expected_state:
            raise FlatSegmentError("Metal probe prepared KV state differs")
        prepared_rows = transaction._prepared_successor_rows_for_oracle(prepared)
        actual_continuation = []
        for lane, token in enumerate(continuation):
            logits, taps = target.model.forward_with_taps(
                mx.array([[token]], dtype=mx.int32),
                copy.deepcopy(prepared_rows[lane]),
                target.capture_layers,
            )
            mx.eval(logits, taps)
            actual_continuation.append((logits[0, 0], taps[0, 0]))
        if not _outputs_exact(tuple(actual_continuation), expected_continuation):
            raise FlatSegmentError("Metal probe continuation oracle differs")
        transaction.publish_commit(prepared)
        published = True
    finally:
        if transaction is not None and not transaction.closed and not published:
            transaction.abort()

    identity = dict(expected_identity)
    receipt = {
        "schema": PROBE_SCHEMA,
        "route": "dense-qwen3-flat-segment-metal-direct-probe",
        "execution_scope": METAL_EXECUTION_SCOPE,
        "checkpoint": identity["artifact"],
        "source_revision": identity["source_revision"],
        "build": identity["build"],
        "device": identity["device"],
        "device_attestation": admission.device_attestation,
        "geometry_sha256": descriptor.geometry_sha256,
        "lane_count": descriptor.lane_count,
        "logical_rows": descriptor.logical_rows,
        "accepted_lengths": list(accepted),
        "target_invocations": 1,
        "ordinary_serial_oracle": {
            "full_logits": "bit_exact",
            "capture_taps": "bit_exact",
            "kv_state": "bit_exact_prepublication",
            "continuation_logits_and_taps": "bit_exact",
        },
        "packed_matmul_authority": False,
        "projection_fidelity": "ordinary_b1s1_row_exact",
        "direct_probe_only": True,
        "performance_claim": False,
        "serving_wired": False,
        "implemented": True,
        "qualified": False,
        "selected": False,
        "default": False,
        "observed_used": True,
        "admission_manifest_sha256": admission.receipt()["manifest_sha256"],
    }
    return DenseQwen3MetalProbeResult(receipt=receipt)


__all__ = [
    "DenseQwen3MetalProbeResult",
    "run_dense_qwen3_metal_flat_probe",
]
