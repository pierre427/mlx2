"""Fail-closed boundary for exact-prefix serving.

The direct adapter primitive can verify and commit a caller-selected prefix,
but that is not enough to serve a request.  A serving route must derive its
match count and correction token from the live ordinary request's target
sampler while preserving processor, RNG, history, stop and cache ownership.
No such handoff contract exists yet, so this module refuses every open rather
than accepting caller-supplied target observations.
"""

from __future__ import annotations

from typing import Any


class ExactPrefixServingUnsupported(RuntimeError):
    """The adapter or request cannot prove the exact serving contract."""


_REQUIRED_CONTRACT = {
    "schema": "mlx2.exact-prefix-cascade-contract.v1",
    "verification_order": "longest_first",
    "invalid_sibling_pruning": True,
    "accepted_prefix_state": "b1_tokenwise_hybrid_transaction",
    "shared_prefix_reuse": "committed_target_cache_clone",
    "common_tokens_recomputed": False,
    "request_private_only": True,
    "apcv2_publication": False,
    "mtp_cache_reuse": False,
    "twotower_reuse": False,
    "implemented": True,
    "serving_route_implemented": False,
    "http_request_selection_implemented": False,
    "state_binding": (
        "artifact_source_layout_request_owner_checkpoint_planes_b1"
    ),
    "target_observation": "unimplemented_sampler_processors_rng_history",
}


def _adapter_contract(adapter: Any) -> dict:
    declare = getattr(adapter, "exact_prefix_cascade_contract", None)
    if not callable(declare):
        raise ExactPrefixServingUnsupported(
            "adapter does not declare an exact-prefix cascade contract"
        )
    contract = declare()
    if not isinstance(contract, dict):
        raise ExactPrefixServingUnsupported(
            "adapter exact-prefix capability must be a contract object"
        )
    for key, expected in _REQUIRED_CONTRACT.items():
        if contract.get(key) != expected:
            raise ExactPrefixServingUnsupported(
                f"adapter exact-prefix contract does not prove {key}={expected!r}"
            )
    for key in ("qualified", "selected", "observed_used"):
        if type(contract.get(key)) is not bool:
            raise ExactPrefixServingUnsupported(
                f"adapter exact-prefix contract requires boolean {key}"
            )
    return dict(contract)


class ExactPrefixServingSession:
    """Unavailable until ordinary request state has an authoritative handoff."""

    def __init__(
        self,
        adapter: Any,
        cache: Any,
        paths,
        *,
        request_id: str,
        maximum: int,
        state_binding,
        mtp_active: bool = False,
    ):
        del cache, paths, request_id, maximum, state_binding, mtp_active
        _adapter_contract(adapter)
        raise ExactPrefixServingUnsupported(
            "exact-prefix served route requires an authoritative ordinary-request "
            "target observation binding for sampler, processors, RNG and history"
        )


def open_exact_prefix_serving_session(
    adapter,
    cache,
    paths,
    *,
    request_id,
    maximum,
    state_binding,
    mtp_active=False,
) -> ExactPrefixServingSession:
    """Refuse selection until the target-observation handoff is implemented."""

    return ExactPrefixServingSession(
        adapter,
        cache,
        paths,
        request_id=request_id,
        maximum=maximum,
        state_binding=state_binding,
        mtp_active=mtp_active,
    )


__all__ = [
    "ExactPrefixServingSession",
    "ExactPrefixServingUnsupported",
    "open_exact_prefix_serving_session",
]
