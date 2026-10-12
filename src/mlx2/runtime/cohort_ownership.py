"""Model-neutral ownership helpers for client-declared batch cohorts.

The serving ingress validates and publishes a declared cohort atomically, but
each executor still has to preserve that identity until its first complete
decode frontier.  Keeping the key construction here prevents executor-local
normalisation from accidentally merging tenants or treating malformed
metadata as an undeclared request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DeclaredCohortKey = tuple[str, str, int]


def declared_cohort_key(value: Any) -> DeclaredCohortKey | None:
    """Return a strict ownership key, or ``None`` for an undeclared lane.

    A present but malformed declaration is an error.  Falling back to
    undeclared scheduling would silently weaken the client's atomic-width
    contract precisely when ownership metadata is damaged.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("batch_cohort must be an object")
    tenant_id = value.get("tenant_id")
    cohort_id = value.get("id")
    size = value.get("size")
    if type(tenant_id) is not str or not tenant_id:
        raise ValueError("batch_cohort tenant_id must be a non-empty string")
    if type(cohort_id) is not str or not cohort_id:
        raise ValueError("batch_cohort id must be a non-empty string")
    if type(size) is not int or size < 1:
        raise ValueError("batch_cohort size must be a positive integer")
    return (tenant_id, cohort_id, size)
