"""Immutable, content-addressed bundles for semantic sidecar state.

Capsules contain validated data, while the hyper directory contains only
layered names, relationships, policy and handles.  Keeping these planes
separate makes directory resolution cheap and gives every payload a stable,
verifiable identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPSULE_SCHEMA = "mlx2-semantic-capsule-v1"
DEFAULT_MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
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
        "activation_capsule",
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

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES,
    ):
        requested = Path(root).expanduser()
        if requested.exists() and requested.is_symlink():
            raise CapsuleIntegrityError("capsule root must not be a symlink")
        if (
            isinstance(maximum_envelope_bytes, bool)
            or not isinstance(maximum_envelope_bytes, int)
            or maximum_envelope_bytes <= 0
        ):
            raise CapsuleIntegrityError("maximum_envelope_bytes must be a positive integer")
        self.root = requested.resolve()
        self.objects = self.root / "objects"
        self.quarantine = self.root / "quarantine"
        self.maximum_envelope_bytes = maximum_envelope_bytes
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
        if len(payload) > self.maximum_envelope_bytes:
            raise CapsuleIntegrityError("capsule envelope exceeds byte limit")
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

    def _quarantine(self, path: Path, digest: str) -> None:
        target = self.quarantine / f"{digest}.{time.time_ns()}.json"
        try:
            os.replace(path, target)
        except FileNotFoundError:
            pass

    def get(self, digest: str) -> dict:
        path = self._path(digest)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
        except FileNotFoundError:
            raise FileNotFoundError(digest) from None
        except OSError as error:
            self._quarantine(path, digest)
            raise CapsuleIntegrityError("capsule must be a regular non-symlink file") from error
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise CapsuleIntegrityError("capsule must be owner-private and regular")
            if info.st_size > self.maximum_envelope_bytes:
                raise CapsuleIntegrityError("capsule envelope exceeds byte limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                payload = stream.read(self.maximum_envelope_bytes + 1)
            if len(payload) > self.maximum_envelope_bytes:
                raise CapsuleIntegrityError("capsule envelope exceeds byte limit")
            value = json.loads(payload.decode("utf-8"))
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
            self._quarantine(path, digest)
            description = "oversized" if "exceeds byte limit" in str(error) else "corrupt"
            raise CapsuleIntegrityError(
                f"quarantined {description} capsule {digest}"
            ) from error
        finally:
            os.close(fd)

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
    "DEFAULT_MAX_ENVELOPE_BYTES",
    "CapsuleIdentity",
    "CapsuleIntegrityError",
    "CapsuleStore",
    "canonical_json",
    "content_digest",
]
