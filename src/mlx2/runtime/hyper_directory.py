"""Revision-bound directory for layered semantic capsule handles."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping, Sequence

from .semantic_capsules import DIGEST_PATTERN, CapsuleStore, canonical_json


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


def _is_name(value: object) -> bool:
    return isinstance(value, str) and NAME_PATTERN.fullmatch(value) is not None


def _validate_layer(value: object, scope: Scope, key: Sequence[str]) -> dict:
    """Check a stored layer has the shape ``update`` writes, or raise.

    Resolving and deleting both consume a layer's handles, and a delete
    removes the capsules they reach, so both accept exactly this shape.
    """
    if (
        not isinstance(value, dict)
        or value.get("schema") != DIRECTORY_SCHEMA
        or value.get("scope") != scope.value
        or value.get("key") != list(key)
        or type(value.get("revision")) is not int
        or value["revision"] < 0
    ):
        raise ValueError("invalid hyper directory layer")
    handles = value.get("handles")
    policies = value.get("policies")
    relationships = value.get("relationships")
    if (
        not isinstance(handles, dict)
        or not all(
            _is_name(name)
            and isinstance(digest, str)
            and DIGEST_PATTERN.fullmatch(digest) is not None
            for name, digest in handles.items()
        )
        or not isinstance(policies, dict)
        or not all(_is_name(name) for name in policies)
        or not isinstance(relationships, list)
        or not all(
            isinstance(item, dict)
            and set(item) == {"source", "type", "target"}
            and all(_is_name(part) for part in item.values())
            and item["type"] in RELATION_TYPES
            for item in relationships
        )
    ):
        raise ValueError("invalid hyper directory layer")
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
        self._transaction_depth = 0

    @contextmanager
    def transaction(self):
        """Lock to hold across a capsule put and the update that publishes it.

        delete_session removes capsules that no layer references while
        holding this lock. A writer that puts a capsule and publishes it
        later must hold the lock across both steps, or the sweep could
        remove a capsule (or a parent it reuses) before the handle lands.
        """
        # The file lock extends the transaction to independent instances and
        # processes. Nested resolve/update calls reuse the outer file lock:
        # flock on another descriptor would deadlock against our own lock.
        with self._lock:
            if self._transaction_depth:
                self._transaction_depth += 1
                try:
                    yield
                finally:
                    self._transaction_depth -= 1
                return
            fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._transaction_depth = 1
                try:
                    yield
                finally:
                    self._transaction_depth = 0
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _transaction(self):
        return self.transaction()

    def _path(self, scope: Scope, key: Sequence[str]) -> Path:
        identity = hashlib.sha256(canonical_json([scope.value, *key])).hexdigest()
        # '%' cannot occur in a legacy scope name, including a name that
        # happens to equal an older hashed identity.
        return self.root / f"{scope.value}--%{identity}.json"

    def _legacy_path(self, scope: Scope, key: Sequence[str]) -> Path | None:
        name = f"{'--'.join((scope.value, *key))}.json"
        return self.root / name if len(name.encode()) <= 255 else None

    def _layer_paths(self, scope: Scope, key: Sequence[str]) -> tuple[Path, ...]:
        identity = hashlib.sha256(canonical_json([scope.value, *key])).hexdigest()
        candidates = (
            self._path(scope, key),
            self.root / f"{scope.value}--{identity}.json",
            self._legacy_path(scope, key),
        )
        return tuple(dict.fromkeys(path for path in candidates if path is not None))

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

    def _load(self, path: Path, scope: Scope, key: Sequence[str]) -> dict | None:
        """Read and validate the layer at ``path``; None if another tuple's."""
        if path.is_symlink() or not path.is_file():
            raise ValueError("directory layer must be a regular file")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("invalid hyper directory layer")  # noqa: TRY004 - invalid persisted schema
        if (
            path != self._path(scope, key)
            and value.get("schema") == DIRECTORY_SCHEMA
            and value.get("scope") == scope.value
            and value.get("key") != list(key)
        ):
            # Another valid tuple can occupy an older name, either through
            # delimiter joining or a literal name matching an old digest.
            return None
        return _validate_layer(value, scope, key)

    def _read(self, scope: Scope, key: Sequence[str]) -> dict:
        for path in self._layer_paths(scope, key):
            if path.exists() or path.is_symlink():
                value = self._load(path, scope, key)
                if value is not None:
                    return value
        return self._empty(scope, key)

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
        with self._transaction():
            current = self._read(scope, key)
            if current["revision"] != expected_revision:
                raise ValueError(
                    f"directory revision conflict: expected {expected_revision}, "
                    f"found {current['revision']}"
                )
            next_value = json.loads(json.dumps(current))
            # Publishing into a deleted session starts it afresh; the
            # revision keeps counting from the tombstone.
            next_value.pop("deleted", None)
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
            self._write(self._path(scope, key), next_value)
            return next_value

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".directory-", suffix=".tmp", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_json(value) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def resolve(self, context: DirectoryContext) -> ResolvedDirectory:
        with self._transaction():
            return self._resolve(context)

    def _resolve(self, context: DirectoryContext) -> ResolvedDirectory:
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
        with self._transaction():
            # Validate the layer before touching either current or legacy path.
            current = self._read(Scope.SESSION, key)
            owned = []
            # A migrated session keeps its legacy layer beside the canonical
            # one until this delete unlinks it, and a handle replaced after
            # the migration (a rebuilt neural capsule) is named only there.
            # Doom what every owned layer reaches, not just the one resolved.
            # The canonical layer shadows the legacy one on every read, so
            # validate each here exactly as a read would: the handles of an
            # unchecked legacy file could name any unreferenced capsule.
            owned_roots = []
            for path in self._layer_paths(Scope.SESSION, key):
                if not path.exists() and not path.is_symlink():
                    continue
                value = self._load(path, Scope.SESSION, key)
                if value is None:
                    continue
                owned.append(path)
                owned_roots.extend(value["handles"].values())
            if not owned or current.get("deleted"):
                return False
            # Every commit writes a capsule holding the whole session graph,
            # so the deleted handles' parent chains carry the session's
            # plaintext facts. Remove the ones no remaining layer reaches;
            # content addressing lets another session share a capsule.
            # Liveness is computed before anything changes so an unreadable
            # layer fails the delete instead of dropping a live capsule.
            live_roots = []
            for path in sorted(self.root.glob("*.json")):
                if path in owned:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError("directory layer must be a regular file")
                value = json.loads(path.read_text())
                if not isinstance(value, dict) or value.get("schema") != DIRECTORY_SCHEMA or not isinstance(
                    value.get("handles"), dict
                ):
                    raise ValueError("invalid hyper directory layer")
                live_roots.extend(value["handles"].values())
            doomed = self._capsule_closure(owned_roots)
            doomed -= self._capsule_closure(live_roots)
            # Replace the layer with an empty tombstone one revision later
            # instead of unlinking it. Unlinking reset the revision to 0, so a
            # request prepared before the delete at an earlier revision could
            # pass the commit CAS again and write memory into the deleted
            # session.
            tombstone = self._empty(Scope.SESSION, key)
            tombstone["revision"] = current["revision"] + 1
            tombstone["deleted"] = True
            self._write(self._path(Scope.SESSION, key), tombstone)
            for path in owned:
                if path != self._path(Scope.SESSION, key):
                    path.unlink()
            for digest in sorted(doomed):
                self.capsules.delete(digest)
            return True

    def _capsule_closure(self, digests) -> set[str]:
        """Capsules reachable from ``digests`` through their parents."""
        reached: set[str] = set()
        pending = list(digests)
        while pending:
            digest = pending.pop()
            if digest in reached:
                continue
            try:
                capsule = self.capsules.get(digest)
            except FileNotFoundError:
                continue
            reached.add(digest)
            pending.extend(capsule["parents"])
        return reached


__all__ = [
    "DIRECTORY_SCHEMA",
    "DirectoryContext",
    "HyperDirectory",
    "RELATION_TYPES",
    "ResolvedDirectory",
    "Scope",
]
