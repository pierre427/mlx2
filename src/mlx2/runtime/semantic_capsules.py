"""Immutable, content-addressed bundles for semantic sidecar state.

Capsules contain validated data, while the hyper directory contains only
layered names, relationships, policy and handles.  Keeping these planes
separate makes directory resolution cheap and gives every payload a stable,
verifiable identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence


CAPSULE_SCHEMA = "mlx2-semantic-capsule-v1"
CAPSULE_KINDS = frozenset(
    {
        "classifier",
        "semantic_base",
        "semantic_delta",
        "neural_model",
        "neural_state",
        "policy",
        "model_binding",
        "evaluation",
    }
)
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CapsuleIdentity:
    digest: str
    kind: str
    parents: tuple[str, ...]
    model_binding: str | None
    tokenizer_binding: str | None
    runtime_binding: str | None


class CapsuleIntegrityError(ValueError):
    pass


class CapsuleStore:
    """Owner-private atomic capsule storage with quarantine on corruption."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.quarantine = self.root / "quarantine"
        for directory in (self.root, self.objects, self.quarantine):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise CapsuleIntegrityError("capsule directories must be real directories")
            os.chmod(directory, 0o700)

    @staticmethod
    def _validate_digest(digest: str) -> str:
        if not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None:
            raise CapsuleIntegrityError("capsule digest must be lowercase SHA-256")
        return digest

    def _path(self, digest: str) -> Path:
        return self.objects / f"{self._validate_digest(digest)}.json"

    @staticmethod
    def _validate_parents(parents: Sequence[str]) -> tuple[str, ...]:
        result = tuple(parents)
        if len(result) != len(set(result)):
            raise ValueError("capsule parents must be unique")
        for digest in result:
            CapsuleStore._validate_digest(digest)
        return result

    def put(
        self,
        *,
        kind: str,
        data: Mapping[str, Any],
        parents: Sequence[str] = (),
        model_binding: str | None = None,
        tokenizer_binding: str | None = None,
        runtime_binding: str | None = None,
        provenance: Mapping[str, Any],
    ) -> CapsuleIdentity:
        if kind not in CAPSULE_KINDS:
            raise ValueError(f"unsupported capsule kind: {kind!r}")
        if not isinstance(data, Mapping) or not isinstance(provenance, Mapping):
            raise TypeError("capsule data and provenance must be mappings")
        parents = self._validate_parents(parents)
        missing = [parent for parent in parents if not self._path(parent).is_file()]
        if missing:
            raise ValueError(f"capsule parents are missing: {missing!r}")
        envelope = {
            "schema": CAPSULE_SCHEMA,
            "kind": kind,
            "parents": list(parents),
            "bindings": {
                "model": model_binding,
                "tokenizer": tokenizer_binding,
                "runtime": runtime_binding,
            },
            "provenance": dict(provenance),
            "data": dict(data),
        }
        digest = content_digest(envelope)
        destination = self._path(digest)
        payload = canonical_json({"digest": digest, **envelope}) + b"\n"
        if destination.exists():
            self.get(digest)
        else:
            fd, temporary = tempfile.mkstemp(
                prefix=".capsule-", suffix=".tmp", dir=self.objects
            )
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                os.chmod(destination, 0o600)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return CapsuleIdentity(
            digest=digest,
            kind=kind,
            parents=parents,
            model_binding=model_binding,
            tokenizer_binding=tokenizer_binding,
            runtime_binding=runtime_binding,
        )

    def get(self, digest: str) -> dict:
        path = self._path(digest)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(digest)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise CapsuleIntegrityError("capsule must be owner-private")
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise CapsuleIntegrityError("capsule envelope must be an object")
            embedded = value.pop("digest")
            if embedded != digest or content_digest(value) != digest:
                raise CapsuleIntegrityError("capsule digest mismatch")
            if value.get("schema") != CAPSULE_SCHEMA:
                raise CapsuleIntegrityError("unsupported capsule schema")
            if value.get("kind") not in CAPSULE_KINDS:
                raise CapsuleIntegrityError("unsupported capsule kind")
            return {"digest": digest, **value}
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, CapsuleIntegrityError) as error:
            stamp = time.time_ns()
            target = self.quarantine / f"{digest}.{stamp}.json"
            try:
                os.replace(path, target)
            except FileNotFoundError:
                pass
            raise CapsuleIntegrityError(f"quarantined corrupt capsule {digest}") from error

    def delete(self, digest: str) -> bool:
        """Remove one capsule; callers decide that nothing references it."""
        path = self._path(digest)
        if path.is_symlink() or path.is_dir():
            raise CapsuleIntegrityError("capsule must be a regular file")
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True


__all__ = [
    "CAPSULE_KINDS",
    "CAPSULE_SCHEMA",
    "CapsuleIdentity",
    "CapsuleIntegrityError",
    "CapsuleStore",
    "canonical_json",
    "content_digest",
]
