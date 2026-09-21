"""Revision-bound directory for layered semantic capsule handles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping, Sequence

from .semantic_capsules import CapsuleStore, canonical_json


DIRECTORY_SCHEMA = "mlx2-hyper-directory-v1"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
RELATION_TYPES = frozenset(
    {
        "contains",
        "inherits",
        "overrides",
        "uses",
        "governed_by",
        "evaluated_by",
        "derived_from",
    }
)


class Scope(str, Enum):
    GLOBAL = "global"
    MODEL = "model"
    TENANT = "tenant"
    SESSION = "session"
    REQUEST = "request"


SCOPE_ORDER = (Scope.GLOBAL, Scope.MODEL, Scope.TENANT, Scope.SESSION, Scope.REQUEST)


@dataclass(frozen=True, slots=True)
class DirectoryContext:
    model: str | None = None
    tenant: str | None = None
    session: str | None = None
    request: str | None = None

    def key_for(self, scope: Scope) -> tuple[str, ...]:
        parts = {
            Scope.GLOBAL: (),
            Scope.MODEL: (self.model,),
            Scope.TENANT: (self.model, self.tenant),
            Scope.SESSION: (self.model, self.tenant, self.session),
            Scope.REQUEST: (self.model, self.tenant, self.session, self.request),
        }[scope]
        if any(value is None for value in parts):
            raise ValueError(f"{scope.value} scope is incomplete")
        return tuple(_validate_name(value) for value in parts)


@dataclass(frozen=True, slots=True)
class ResolvedDirectory:
    revision: int
    fingerprint: str
    handles: Mapping[str, str]
    policies: Mapping[str, Any]
    relationships: tuple[Mapping[str, str], ...]
    layers: tuple[Mapping[str, Any], ...]


def _validate_name(value: str) -> str:
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("directory names must be bounded safe identifiers")
    return value


class HyperDirectory:
    """Layered directory with monotonic revisions and CAS updates."""

    def __init__(self, root: str | Path, capsules: CapsuleStore):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("hyper directory root must be a real directory")
        os.chmod(self.root, 0o700)
        self.capsules = capsules
        self._lock = threading.RLock()

    def _path(self, scope: Scope, key: Sequence[str]) -> Path:
        encoded = "--".join((scope.value, *key))
        return self.root / f"{encoded}.json"

    @staticmethod
    def _empty(scope: Scope, key: Sequence[str]) -> dict:
        return {
            "schema": DIRECTORY_SCHEMA,
            "scope": scope.value,
            "key": list(key),
            "revision": 0,
            "handles": {},
            "policies": {},
            "relationships": [],
        }

    def _read(self, scope: Scope, key: Sequence[str]) -> dict:
        path = self._path(scope, key)
        if not path.exists():
            return self._empty(scope, key)
        if path.is_symlink() or not path.is_file():
            raise ValueError("directory layer must be a regular file")
        value = json.loads(path.read_text())
        if (
            value.get("schema") != DIRECTORY_SCHEMA
            or value.get("scope") != scope.value
            or value.get("key") != list(key)
            or type(value.get("revision")) is not int
        ):
            raise ValueError("invalid hyper directory layer")
        return value

    def update(
        self,
        scope: Scope,
        context: DirectoryContext,
        *,
        expected_revision: int,
        handles: Mapping[str, str] | None = None,
        policies: Mapping[str, Any] | None = None,
        relationships: Sequence[Mapping[str, str]] | None = None,
    ) -> dict:
        key = context.key_for(scope)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        with self._lock:
            current = self._read(scope, key)
            if current["revision"] != expected_revision:
                raise ValueError(
                    f"directory revision conflict: expected {expected_revision}, "
                    f"found {current['revision']}"
                )
            next_value = json.loads(json.dumps(current))
            for name, digest in (handles or {}).items():
                name = _validate_name(name)
                self.capsules.get(digest)
                next_value["handles"][name] = digest
            for name, value in (policies or {}).items():
                next_value["policies"][_validate_name(name)] = value
            if relationships is not None:
                validated = []
                for relationship in relationships:
                    if set(relationship) != {"source", "type", "target"}:
                        raise ValueError("directory relationships require source/type/target")
                    item = {name: _validate_name(value) for name, value in relationship.items()}
                    if item["type"] not in RELATION_TYPES:
                        raise ValueError("unsupported directory relationship")
                    validated.append(item)
                next_value["relationships"] = validated
            next_value["revision"] += 1
            path = self._path(scope, key)
            fd, temporary = tempfile.mkstemp(prefix=".directory-", suffix=".tmp", dir=self.root)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(canonical_json(next_value) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                os.chmod(path, 0o600)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return next_value

    def resolve(self, context: DirectoryContext) -> ResolvedDirectory:
        layers = []
        handles: dict[str, str] = {}
        policies: dict[str, Any] = {}
        relationships: dict[tuple[str, str, str], dict[str, str]] = {}
        total_revision = 0
        for scope in SCOPE_ORDER:
            try:
                key = context.key_for(scope)
            except ValueError:
                break
            layer = self._read(scope, key)
            layers.append(layer)
            total_revision += layer["revision"]
            handles.update(layer["handles"])
            policies.update(layer["policies"])
            for item in layer["relationships"]:
                relationships[(item["source"], item["type"], item["target"])] = item
        snapshot = {
            "schema": DIRECTORY_SCHEMA,
            "layers": [
                {"scope": item["scope"], "key": item["key"], "revision": item["revision"]}
                for item in layers
            ],
            "handles": handles,
            "policies": policies,
            "relationships": list(relationships.values()),
        }
        fingerprint = hashlib.sha256(canonical_json(snapshot)).hexdigest()
        return ResolvedDirectory(
            revision=total_revision,
            fingerprint=fingerprint,
            handles=handles,
            policies=policies,
            relationships=tuple(relationships.values()),
            layers=tuple(layers),
        )

    def delete_session(self, context: DirectoryContext) -> bool:
        key = context.key_for(Scope.SESSION)
        path = self._path(Scope.SESSION, key)
        with self._lock:
            if not path.exists():
                return False
            if path.is_symlink() or not path.is_file():
                raise ValueError("session directory layer must be a regular file")
            path.unlink()
            return True


__all__ = [
    "DIRECTORY_SCHEMA",
    "DirectoryContext",
    "HyperDirectory",
    "RELATION_TYPES",
    "ResolvedDirectory",
    "Scope",
]
