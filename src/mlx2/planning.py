"""Read-only artifact planning using adapter-owned serving memory projections.

This command never loads weights, queries Metal, or downloads an artifact.
Serialized weight bytes are a planning proxy, never measured residency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .adapters.artifact_paths import shard_within_artifact
from .cache_sizing import GIB, default_cache_bytes, physical_memory_bytes


def _object(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate metadata key in {path.name}: {key}")
            result[key] = value
        return result

    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"metadata exceeds 64 MiB: {path.name}")
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be an object: {path.name}")  # noqa: TRY004 - invalid JSON document
    return value


def artifact_payload(path: Path) -> dict:
    """Stat declared shards and local binary side assets without tensor reads."""
    root = path.expanduser().resolve(strict=True)
    config = _object(root / "config.json")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weights = _object(index).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("artifact index needs a nonempty weight_map")
        if any(not isinstance(name, str) for name in weights.values()):
            raise ValueError("artifact shard names must be strings")
        names = set(weights.values())
    else:
        names = {p.name for p in root.glob("*.safetensors")}
    if not names:
        raise ValueError("artifact has no safetensors payload")
    # Side assets such as PLE rows must not disappear from the storage budget.
    names.update(p.name for p in root.glob("*.bin"))
    records = []
    for name in sorted(names):
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("payload path escapes the artifact")
        item = root / name
        resolved = item.resolve(strict=True)
        if not shard_within_artifact(root, resolved) or not resolved.is_file():
            raise ValueError(f"foreign or non-file artifact payload: {name}")
        stat = resolved.stat()
        if stat.st_size < 8:
            raise ValueError(f"empty artifact payload: {name}")
        records.append({"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    fingerprint = hashlib.sha256(json.dumps(
        {"config": config, "files": records}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {"path": str(root), "config": config, "files": records,
            "serialized_bytes": sum(r["bytes"] for r in records),
            "metadata_fingerprint": fingerprint,
            "identity_kind": "metadata-and-file-stat-not-weight-content-hash"}


def _count(name, value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def deployment_plan(model_path, *, context=8192, lanes=1, route="ordinary",
                    host_bytes=None, advisory_bytes=None, apc_bytes=None,
                    scratch_bytes=None, draft_path=None, side_cache_bytes=0):
    """Return explicit known costs and unknowns; never an admission decision."""
    from .adapters.registry import inspect_model
    from .contracts import Capability
    from .runtime.memory_policy import SelfMTPLaneAdmissionController

    _count("context", context, minimum=1)
    _count("lanes", lanes, minimum=1)
    _count("side_cache_bytes", side_cache_bytes)
    if route not in {"ordinary", "native_mtp", "external_draft"}:
        raise ValueError("unsupported planning route")
    if (draft_path is not None) != (route == "external_draft"):
        raise ValueError("external_draft requires a draft artifact, and only that route accepts one")
    target = artifact_payload(Path(model_path))
    resolution = inspect_model(model_path)
    requested = {"native_mtp": Capability.MTP, "external_draft": Capability.EXTERNAL_DRAFT}.get(route)
    if requested is not None and requested not in resolution.descriptor.capabilities:
        raise ValueError(f"adapter does not implement {route}")
    host_bytes = physical_memory_bytes() if host_bytes is None else host_bytes
    if host_bytes is not None:
        _count("host_bytes", host_bytes, minimum=1)
    if advisory_bytes is not None:
        _count("advisory_bytes", advisory_bytes, minimum=1)
        if host_bytes is not None and advisory_bytes > host_bytes:
            raise ValueError("advisory bytes exceed physical host memory")
    apc_bytes = default_cache_bytes(host_bytes) if apc_bytes is None else apc_bytes
    _count("apc_bytes", apc_bytes)
    if scratch_bytes is not None:
        _count("scratch_bytes", scratch_bytes)
    unknown = ["loaded_weight_residency", "allocator_and_driver_live_headroom"]
    # Do not inherit a projection measured or declared for a different family.
    method = vars(resolution.adapter_type).get("planning_cache_budget")
    budget = None if method is None else resolution.adapter_type.planning_cache_budget(
        target["config"], mtp=route == "native_mtp"
    )
    lane_bytes = None if budget is None else int(budget.project(context)) * lanes
    if lane_bytes is None:
        unknown.append("adapter_cache_projection")
    if scratch_bytes is None:
        unknown.append("forward_and_load_peak_scratch")
    draft = artifact_payload(Path(draft_path)) if draft_path is not None else None
    if draft is not None:
        unknown.extend(("draft_runtime_compatibility", "draft_cache_and_workspace"))
    service, driver = SelfMTPLaneAdmissionController.host_scaled_reserves(
        None if host_bytes is None else host_bytes / GIB,
        None if advisory_bytes is None else advisory_bytes / GIB,
    )
    components = {
        "serialized_target_proxy": target["serialized_bytes"],
        "serialized_draft_proxy": 0 if draft is None else draft["serialized_bytes"],
        "running_lane_cache_projection": lane_bytes,
        "apcv2_resident_ceiling": apc_bytes,
        "explicit_side_cache": side_cache_bytes,
        "explicit_peak_scratch": scratch_bytes,
        "service_and_driver_reserve": int((service + driver) * GIB),
    }
    known = sum(value for value in components.values() if value is not None)
    limit = advisory_bytes if advisory_bytes is not None else host_bytes
    return {
        "schema": "mlx2.deployment-plan.v1", "operation": "metadata_only",
        "descriptor": resolution.descriptor.key,
        "implemented_capabilities": sorted(c.value for c in resolution.descriptor.capabilities),
        "route": route, "context_tokens": context, "running_lanes": lanes,
        "target": {k: v for k, v in target.items() if k != "config"},
        "draft": None if draft is None else {k: v for k, v in draft.items() if k != "config"},
        "cache_geometry": None if budget is None else budget.as_dict(),
        "host_bytes": host_bytes, "advisory_bytes": advisory_bytes,
        "components_bytes": components, "known_planning_bytes": known,
        "estimated_headroom_bytes": None if limit is None else limit - known,
        "assessment": "over_planning_budget" if limit is not None and known > limit else "indeterminate",
        "unknowns": unknown,
        "assumptions": ["all serialized target and draft payloads counted as resident proxies",
                        "APCv2 ceiling added conservatively; shared buffers are not discounted",
                        "host reserves reuse serving policy; absent advisory is imputed by that policy"],
        "qualified": False, "selected": False, "observed_used": False,
        "admission_authorized": False,
    }


def _gib(value):
    number = float(value)
    if not math.isfinite(number) or number < 0 or not math.isfinite(number * GIB):
        raise argparse.ArgumentTypeError("GiB must be finite and nonnegative")
    return int(number * GIB)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--route", choices=("ordinary", "native_mtp", "external_draft"), default="ordinary")
    parser.add_argument("--draft-path")
    for name in ("host", "advisory", "apc", "scratch", "side-cache"):
        parser.add_argument(f"--{name}-gib", type=_gib, default=0 if name == "side-cache" else None)
    args = parser.parse_args(argv)
    try:
        report = deployment_plan(
            args.model_path, context=args.context, lanes=args.lanes, route=args.route,
            host_bytes=args.host_gib, advisory_bytes=args.advisory_gib,
            apc_bytes=args.apc_gib, scratch_bytes=args.scratch_gib,
            draft_path=args.draft_path, side_cache_bytes=args.side_cache_gib,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"planning failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
