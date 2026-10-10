"""Route source identities with conservative full-build fallback.

Dispatch cut points are reviewed against exact bytes. A changed dispatcher or
an unresolved dependency keeps the full-source identity, so reduced cache and
qualification invalidation never means accepting an unknown dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from .source_dependencies import (
    UnresolvedDependency,
    add_resources,
    canonical_digest,
    digest_files,
    file_digest,
    source_closure,
)

SCHEMA = "mlx2.route-source.v1"
_COMMON = ("mlx2.server", "mlx2.serving", "mlx2.qualification", "mlx2.route_identity")
_ROUTES = {"ordinary", "self_mtp", "native_mtp", "prompt_lookup", "external_draft"}


def adapter_roots(adapter):
    roots = set()
    cls = type(adapter)
    if not cls.__module__.startswith("mlx2.adapters."):
        raise UnresolvedDependency("external adapter has no reviewed source boundary")
    pending, seen = [adapter], set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if len(seen) > 32:
            raise UnresolvedDependency("model wrapper graph exceeds reviewed bound")
        for base in type(value).__mro__:
            if base.__module__.startswith("mlx2."):
                roots.add(base.__module__)
        for name in ("model", "language_model", "_model", "draft", "draft_model"):
            child = getattr(value, name, None)
            if child is not None and child is not value:
                pending.append(child)
    return sorted(roots)


def _scoped(build, roots, route, root):
    if (
        route not in _ROUTES
        or not roots
        or any(
            not isinstance(name, str) or not name.startswith("mlx2.") for name in roots
        )
    ):
        raise UnresolvedDependency("invalid route dependency roots")
    manifest_path = root / "route_dispatch.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise UnresolvedDependency("unknown dispatch manifest")
    reviews = manifest["dispatchers"]
    # These exact dispatchers may resolve only the adapter/model roots supplied
    # by the constructed route. Static imports remain included without pruning.
    reviews["mlx2.adapters.registry"] = dict(
        reviews["mlx2.adapters.registry"], targets=list(roots)
    )
    reviews["mlx2.runtime.models.cache"] = dict(
        reviews["mlx2.runtime.models.cache"], targets=list(roots)
    )
    files = source_closure(root, (*_COMMON, *roots), reviewed_dynamic=reviews)
    add_resources(root, files)
    scope = {
        "schema": SCHEMA,
        "route": route,
        "roots": sorted(set(roots)),
        "dispatch_manifest_sha256": file_digest(manifest_path),
    }
    runtime = dict(build)
    runtime["source_sha256"] = canonical_digest(
        {
            "scope": scope,
            "files_sha256": digest_files(files),
        }
    )
    runtime["source_scope"] = scope
    return runtime, len(files)


def bind_route_runtime(build, adapter, route, *, root=None):
    root = Path(root) if root is not None else Path(__file__).parent
    try:
        runtime, count = _scoped(build, adapter_roots(adapter), route, root)
    except (UnresolvedDependency, OSError, ValueError, KeyError) as exc:
        return dict(build), {"mode": "full_source", "reason": str(exc)}
    return runtime, {"mode": "route_source", "dependency_files": count}


def recompute_runtime_identity(runtime, *, build, root=None):
    """Freshly recompute a served identity for qualification's stability gate."""
    scope = runtime.get("source_scope")
    if scope is None:
        return build
    if not isinstance(scope, dict) or scope.get("schema") != SCHEMA:
        raise ValueError("unrecognized runtime source scope")
    root = Path(root) if root is not None else Path(__file__).parent
    result, _ = _scoped(build, scope["roots"], scope["route"], root)
    return result
