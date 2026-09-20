"""Process-local or configured authentication for portable reasoning blocks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("empty base64 value")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


PERSISTENT_KEY_NAME = "reasoning-signing.key"


def _read_owner_only_key(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) not in (0o400, 0o600)
        ):
            raise ValueError(
                "reasoning signing key must be a regular file owned by the "
                f"process uid with mode 0400 or 0600: {path}"
            )
        secret = source.read(4097).strip()
    if not secret or len(secret) > 4096:
        raise ValueError(f"invalid reasoning signing key file: {path}")
    return secret


def ensure_persistent_key(path):
    """Return ``path`` holding an owner-only signing secret, creating it once.

    The secret is written to a private temporary file and published with
    ``link``, which fails if the name exists: two servers starting on one
    state directory agree on a single key and never read a partial file.
    """
    path = Path(path)
    try:
        _read_owner_only_key(path)
        return path
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".reasoning-signing-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            os.fchmod(target.fileno(), 0o600)
            target.write(secrets.token_hex(32).encode("ascii"))
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary)
    _read_owner_only_key(path)
    return path


class ReasoningSigner:
    """Sign reasoning without claiming confidentiality.

    Tokens bind the model and tenant scope as well as the text.  The Responses
    token carries the authenticated text so a stateless client can replay it;
    it is signed, not encrypted.
    """

    def __init__(self, secret: bytes | str | None = None):
        if secret is None:
            secret = secrets.token_bytes(32)
            self.ephemeral = True
        else:
            if isinstance(secret, str):
                secret = secret.encode()
            if not isinstance(secret, bytes) or not secret:
                raise ValueError("reasoning signing secret must be nonempty bytes")
            self.ephemeral = False
        self._secret = secret
        self.key_id = hmac.new(
            secret, b"mlx2-key-id", hashlib.sha256
        ).hexdigest()[:16]

    @classmethod
    def configured(cls, *, key_file=None, key_env=None):
        if key_file and key_env:
            raise ValueError("configure only one reasoning signing key source")
        if key_file:
            path = Path(key_file)
            try:
                mode = path.stat().st_mode & 0o7777
                secret = path.read_bytes().strip()
            except OSError as error:
                raise ValueError(
                    f"cannot read reasoning signing key file: {error}"
                ) from error
            if not secret:
                raise ValueError("reasoning signing key file is empty")
            if mode & ~0o600:
                log.warning(
                    "reasoning signing key file %s has permissions %04o, broader than 0600",
                    path,
                    mode,
                )
            return cls(secret)
        if key_env:
            secret = os.environ.get(str(key_env), "").encode()
            if not secret:
                raise ValueError(
                    f"reasoning signing environment variable {key_env!r} is unset or empty"
                )
            return cls(secret)
        log.warning(
            "no reasoning signing key configured; using a random per-process "
            "reasoning signing key"
        )
        return cls()

    def _canonical(self, kind, model, tenant, text):
        return json.dumps(
            {
                "key_id": self.key_id,
                "kind": str(kind),
                "model": str(model),
                "tenant": str(tenant),
                "text": str(text),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _mac(self, kind, model, tenant, text):
        return hmac.new(
            self._secret,
            self._canonical(kind, model, tenant, text),
            hashlib.sha256,
        ).digest()

    def sign_anthropic(self, *, model, tenant, text):
        mac = _b64(self._mac("anthropic", model, tenant, text))
        return f"mlx2.thinking.v1.{self.key_id}.{mac}"

    def verify_anthropic(self, signature, *, model, tenant, text):
        try:
            prefix, kind, version, key_id, encoded = signature.split(".")
            supplied = _unb64(encoded)
        except (AttributeError, ValueError, TypeError):
            return False
        if (prefix, kind, version, key_id) != (
            "mlx2",
            "thinking",
            "v1",
            self.key_id,
        ):
            return False
        expected = self._mac("anthropic", model, tenant, text)
        return hmac.compare_digest(supplied, expected)

    def sign_anthropic_carrying(self, *, model, tenant, text):
        """Signature that also carries the authenticated thinking text.

        Used for Anthropic ``thinking.display: "omitted"``: the client sees an
        empty thinking block, and replaying the signature restores the text.
        The payload is authenticated, not encrypted, exactly like Responses
        ``encrypted_content``.
        """
        payload = _b64(str(text).encode())
        mac = _b64(self._mac("anthropic", model, tenant, text))
        return f"mlx2.thinkingc.v1.{self.key_id}.{payload}.{mac}"

    def verify_anthropic_carrying(self, signature, *, model, tenant):
        try:
            prefix, kind, version, key_id, payload, encoded = signature.split(".")
            text = _unb64(payload).decode()
            supplied = _unb64(encoded)
        except (AttributeError, UnicodeDecodeError, ValueError, TypeError):
            return None
        if (prefix, kind, version, key_id) != (
            "mlx2",
            "thinkingc",
            "v1",
            self.key_id,
        ):
            return None
        expected = self._mac("anthropic", model, tenant, text)
        return text if hmac.compare_digest(supplied, expected) else None

    def sign_responses(self, *, model, tenant, text):
        payload = _b64(str(text).encode())
        mac = _b64(self._mac("responses", model, tenant, text))
        return f"mlx2.reasoning.v1.{self.key_id}.{payload}.{mac}"

    def verify_responses(self, token, *, model, tenant, expected_text=None):
        try:
            prefix, kind, version, key_id, payload, encoded = token.split(".")
            text = _unb64(payload).decode()
            supplied = _unb64(encoded)
        except (AttributeError, UnicodeDecodeError, ValueError, TypeError):
            return None
        if (prefix, kind, version, key_id) != (
            "mlx2",
            "reasoning",
            "v1",
            self.key_id,
        ):
            return None
        if expected_text is not None:
            try:
                expected_bytes = expected_text.encode("utf-8")
            except (AttributeError, UnicodeEncodeError):
                return None
            if not hmac.compare_digest(text.encode("utf-8"), expected_bytes):
                return None
        expected = self._mac("responses", model, tenant, text)
        return text if hmac.compare_digest(supplied, expected) else None
