"""Import-free validation for request-private exact-prefix state bindings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


class ExactPrefixStateBindingError(ValueError):
    """A caller-owned binding does not identify the reusable live state."""


_FIELDS = {
    "artifact_fingerprint",
    "source_revision",
    "cache_layout",
    "request_id",
    "ownership",
    "checkpoint_kind",
    "execution_domain",
    "checkpoint_position",
    "state_planes",
    "batch_size",
    "mtp_state",
    "twotower_state",
}


def validate_exact_prefix_state_binding(
    binding,
    *,
    expected_artifact_fingerprint: str,
    expected_source_revision: str,
    expected_cache_layout: str,
    expected_request_id: str,
    expected_state_planes: frozenset[str],
    expected_execution_domain: str,
    live_checkpoint_position: int | None = None,
) -> dict:
    """Validate a complete authoritative binding and return a frozen copy.

    Passing ``live_checkpoint_position=None`` performs the identity/ownership
    preflight before a model module is asked to describe its cache.  The
    adapter calls this function again with the inspected live position before
    authorizing suffix-only execution.
    """

    if not isinstance(binding, Mapping):
        raise TypeError("exact-prefix reuse requires a state binding")
    raw_planes = binding.get("state_planes")
    planes = (
        frozenset(raw_planes)
        if isinstance(raw_planes, list)
        and all(isinstance(plane, str) for plane in raw_planes)
        else frozenset()
    )
    checkpoint = binding.get("checkpoint_position")
    failures = []
    if set(binding) != _FIELDS:
        failures.append("binding fields")
    if (
        not isinstance(expected_artifact_fingerprint, str)
        or not expected_artifact_fingerprint
        or binding.get("artifact_fingerprint") != expected_artifact_fingerprint
    ):
        failures.append("artifact fingerprint")
    if (
        not isinstance(expected_source_revision, str)
        or not expected_source_revision
        or binding.get("source_revision") != expected_source_revision
    ):
        failures.append("source revision")
    if (
        not isinstance(expected_cache_layout, str)
        or not expected_cache_layout
        or binding.get("cache_layout") != expected_cache_layout
    ):
        failures.append("cache layout")
    if (
        not isinstance(expected_request_id, str)
        or not expected_request_id
        or binding.get("request_id") != expected_request_id
    ):
        failures.append("request owner")
    if binding.get("ownership") != "request_private":
        failures.append("private ownership")
    if binding.get("checkpoint_kind") != "live_authoritative_exact":
        failures.append("checkpoint kind")
    if binding.get("execution_domain") != expected_execution_domain:
        failures.append("B1 execution domain")
    if (
        type(checkpoint) is not int
        or checkpoint < 0
        or (
            live_checkpoint_position is not None
            and checkpoint != live_checkpoint_position
        )
    ):
        failures.append("checkpoint position")
    if planes != expected_state_planes:
        failures.append("state planes")
    if binding.get("batch_size") != 1:
        failures.append("batch size")
    if binding.get("mtp_state") != "absent":
        failures.append("MTP state")
    if binding.get("twotower_state") != "absent":
        failures.append("TwoTower state")
    if failures:
        raise ExactPrefixStateBindingError(
            "exact-prefix state binding differs: " + ", ".join(failures)
        )
    return deepcopy(dict(binding))


__all__ = [
    "ExactPrefixStateBindingError",
    "validate_exact_prefix_state_binding",
]
