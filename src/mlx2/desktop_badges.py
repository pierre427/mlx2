"""CPU-only qualification applicability for the native model library.

Historical records stay available, but current badges require a fresh artifact
fingerprint and current source, dependency, host and requested serving shape.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from .desktop_records import digest


def aggregate(records):
    groups = {}
    for record in sorted(records, key=lambda row: row.get("created", 0), reverse=True):
        group = groups.setdefault(digest(record["binding"]), {})
        group.setdefault(record["level"], record)
    candidates = []
    for group in groups.values():
        levels = {str(level): bool(group.get(level, {}).get("passed")) for level in (1, 2, 3)}
        coverage = group.get(2, {}).get("coverage", {})
        complete = set(coverage.get("required_routes", [])) <= set(coverage.get("tested_routes", []))
        candidates.append((all(levels.values()) and complete, sum(levels.values()), levels, group))
    if not candidates:
        return {"label": "Untested", "qualified": False, "levels": {"1": False, "2": False, "3": False}}
    qualified, count, levels, group = max(candidates, key=lambda row: (row[0], row[1]))
    binding = next(iter(group.values()))["binding"]
    settings = binding["profile"].get("settings", {})
    scope = {"route": binding["profile"].get("route"), "context": settings.get("max_context"),
             "lanes": settings.get("max_lanes"), "profile": binding["profile"]}
    label = "Qualified" if qualified else f"{count}/3 levels"
    if count == 3 and not qualified:
        label += " · route coverage pending"
    return {"label": label, "qualified": qualified, "levels": levels, "scope": scope,
            "binding": binding, "records": len(records)}


class BadgeResolver:
    def __init__(self, root: Path, *, source_identity=None, build_identity=None, runtime_resolver=None, host=None):
        self.root = Path(root)
        self.host = host or {"hostname": platform.node(), "platform": platform.platform()}
        self.error = None
        self._runtime_cache = {}
        try:
            if source_identity is None:
                from .desktop_runner import digest as source_digest
                from .desktop_runner import source_manifest
                source_hash = source_digest(source_manifest(self.root))
                metadata = self.root / "desktop-source.json"
                if metadata.exists():
                    source_identity = json.loads(metadata.read_text())
                    if source_identity.get("source_sha256") != source_hash:
                        raise ValueError("Installed source differs from its frozen manifest")
                else:
                    revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                        text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                    source_identity = {"source_revision": revision, "source_sha256": source_hash}
            self.source = source_identity
            if build_identity is None:
                # serving deliberately imports MLX lazily; runtime_identity
                # reads Python/resource/native-wheel bytes and package metadata.
                # No adapter, array, tokenizer or device is initialized here.
                from .serving import runtime_identity
                build_identity = runtime_identity()
            self.build = build_identity
            if runtime_resolver is None:
                from .route_identity import recompute_runtime_identity
                runtime_resolver = lambda runtime: recompute_runtime_identity(
                    runtime, build=self.build, root=self.root / "src/mlx2")
            self.runtime_resolver = runtime_resolver
        except Exception as error:  # noqa: BLE001 -- unavailable dependencies mean no current badge, not inherited qualification
            self.error = str(error)
            self.source = source_identity or {}
            self.build = build_identity or {}
            self.runtime_resolver = None

    def runtime_matches(self, runtime):
        key = digest(runtime)
        if key not in self._runtime_cache:
            try:
                self._runtime_cache[key] = self.runtime_resolver(runtime) == runtime
            except Exception:  # noqa: BLE001 -- unknown/malformed source scopes fail closed
                self._runtime_cache[key] = False
        return self._runtime_cache[key]

    @staticmethod
    def profile_matches(binding, profiles, row):
        if profiles is None:
            return True
        profiles = [profile for profile in profiles if profile.get("path") == row.get("path")]
        if not profiles:
            return True  # no configured profile; scope remains explicit in badge
        actual = binding["profile"]
        settings = actual.get("settings", {})
        for profile in profiles:
            route = profile.get("route", "auto")
            if route == "auto":
                route = row.get("default_route")
            if route is not None and route != actual.get("route"):
                continue
            if any(profile.get(key, 0) > 0 and profile[key] != settings.get(actual_key)
                   for key, actual_key in (("context", "max_context"), ("lanes", "max_lanes"))):
                continue
            return True
        return False

    def summarize(self, row, records, configured_profiles=None):
        historical = [record for record in records if record.get("binding", {}).get("model_path") == row.get("path")]
        current = []
        if not self.error and row.get("checkpoint"):
            for record in historical:
                bound = record["binding"]
                if (bound.get("checkpoint") == row["checkpoint"]
                    and bound.get("source_sha256") == self.source.get("source_sha256")
                    and bound.get("source_revision") == self.source.get("source_revision")
                    and bound.get("host") == self.host
                    and isinstance(bound.get("runtime"), dict)
                    and self.runtime_matches(bound["runtime"])
                    and self.profile_matches(bound, configured_profiles, row)):
                    current.append(record)
        summary = aggregate(current)
        summary.update(current=bool(current), historical_records=len(historical), applicability_error=self.error)
        if historical and not current:
            summary["label"] = "Historical evidence · retest"
        return summary
