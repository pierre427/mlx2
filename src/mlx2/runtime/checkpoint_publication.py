"""Fail-closed eligibility for publishing compact or recurrent checkpoints.

A recurrent implementation may defer part of its temporal update in a small
ring while its materialized state already reflects another part of the step.
Such a cache is useful as private recovery scratch, but it is not a committed
boundary that APCv2 or an MTP sidecar may donate to another request.

Cache implementations can expose ``checkpoint_publication_refusal()`` or bind
an immutable cache-operations contract.  Both declarations inspect host
bookkeeping only and never synchronize device arrays.  Undeclared cache
families fail closed; the publication boundary never infers transaction
semantics from coincidental private fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .cache_operations import CacheOperationsUnsupported, cache_operations


class CheckpointPublicationUnsafe(RuntimeError):
    """A cache graph does not currently represent a committed boundary."""


def _cache_refusal(cache: Any) -> str | None:
    declared = getattr(cache, "checkpoint_publication_refusal", None)
    if callable(declared):
        reason = declared()
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                return "cache returned an invalid checkpoint-publication refusal"
            return reason
        return None

    operations = cache_operations(
        cache, capability="checkpoint publication", required=False
    )
    if operations is None:
        return "cache family has no checkpoint-publication operations declaration"
    try:
        return operations.publication_refusal(cache)
    except CacheOperationsUnsupported as error:
        return f"cache operations declaration is invalid: {error}"


def checkpoint_publication_refusal(caches: Iterable[Any]) -> str | None:
    """Return the first host-visible reason ``caches`` cannot be published."""
    # CacheList is the one transparent cache container in the runtime.  Treat
    # only its exact type as transparent: a subclass may add transaction state
    # of its own and must therefore declare that state before its children are
    # inspected.
    from .models.cache import CacheList

    stack = list(caches)
    seen: set[int] = set()
    while stack:
        cache = stack.pop()
        if id(cache) in seen:
            continue
        seen.add(id(cache))
        if isinstance(cache, (list, tuple)):
            stack.extend(cache)
            continue
        if type(cache) is CacheList:
            stack.extend(cache.caches)
            continue
        reason = _cache_refusal(cache)
        if reason is not None:
            return f"{type(cache).__name__}: {reason}"
        children = getattr(cache, "caches", None)
        if children is not None:
            stack.extend(children)
    return None


def require_checkpoint_publishable(caches: Iterable[Any]) -> None:
    """Raise before checkpoint donation if any cache has uncommitted state."""
    reason = checkpoint_publication_refusal(caches)
    if reason is not None:
        raise CheckpointPublicationUnsafe(
            f"checkpoint publication refused: {reason}"
        )
