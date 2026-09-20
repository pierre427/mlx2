"""HTTP request gate: Host allowlist, Origin == Host, optional API key.

Adapted from Splash ``server/http_security.py`` (Apache-2.0, rev f58d36dd);
see ``provenance/splash-11-http-security.json``.

The gate defends a loopback-bound server against DNS rebinding (a browser
page on an attacker domain that resolves to 127.0.0.1 sends ``Host:
evil.example``) and against cross-origin browser requests (``Origin`` names a
different authority than ``Host``).  SDK and CLI clients send no ``Origin``
and a ``Host`` naming the address they dialled, so they pass unchanged.

The API key is off unless the operator configures one.  ``load_api_key`` is
the shared loader for the server flags and for client launchers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import ipaddress
import logging
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit


# Conventional client-side environment variable for the key.  The server does
# not read it implicitly (an ambient export must not silently turn on auth for
# harnesses that send no key); pass ``--api-key-env MLX2_API_KEY`` to opt in.
DEFAULT_API_KEY_ENV = "MLX2_API_KEY"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::"})
HEALTH_PATHS = frozenset({"/health"})
ADMIN_PREFIX = "/v1/admin/"


def load_secret_file(path, *, flag):
    """Read one secret line from an owner-only (0600) regular file."""
    candidate = Path(path).expanduser().resolve()
    with candidate.open("rb") as secret_file:
        metadata = os.fstat(secret_file.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{flag} must be a regular file")
        if metadata.st_uid != os.getuid():
            raise ValueError(f"{flag} must be owned by the process uid")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError(f"{flag} permissions must be exactly 0600")
        raw = secret_file.read(4097)
    if len(raw) > 4096:
        raise ValueError(f"{flag} must contain at most 4096 bytes")
    raw = raw.rstrip(b"\r\n")
    if not raw or b"\n" in raw or b"\r" in raw:
        raise ValueError(f"{flag} must contain one nonempty token")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{flag} must contain UTF-8 text") from error


def validate_api_key(value, *, source="API key"):
    # Header values reach the handler latin-1 decoded; restricting keys to
    # visible ASCII keeps the byte comparison unambiguous.
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError(f"{source} must contain only visible ASCII characters")
    return value


def load_api_key(*, key_file=None, key_env=None, environ=None, required=True):
    """Return the configured API key, or ``None`` when none is configured.

    ``key_file`` uses the admin-token file rules (regular, owner, 0600, one
    line).  ``key_env`` names an environment variable; with ``required`` an
    unset or empty variable is an error (server flags), otherwise it means no
    key (client launchers reading ``MLX2_API_KEY``).
    """
    if key_file is not None and key_env is not None:
        raise ValueError("--api-key-file and --api-key-env are mutually exclusive")
    if key_file is not None:
        return validate_api_key(
            load_secret_file(key_file, flag="--api-key-file"),
            source="--api-key-file",
        )
    if key_env is not None:
        value = (os.environ if environ is None else environ).get(str(key_env), "")
        if not value:
            if required:
                raise ValueError(
                    f"API key environment variable {key_env!r} is unset or empty"
                )
            return None
        return validate_api_key(value, source=f"environment variable {key_env!r}")
    return None


def normalize_host(value):
    """Canonical allowlist form: lowercase, no brackets, no trailing dot."""
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.lower().rstrip(".")
    if not text or any(
        ord(char) <= 32 or ord(char) >= 127 or char in "/@?#[]" for char in text
    ):
        raise ValueError(f"invalid host {value!r}")
    if ":" in text:
        try:
            ipaddress.IPv6Address(text.split("%", 1)[0])
        except ValueError:
            raise ValueError(
                f"allowed host {value!r} must not include a port"
            ) from None
    return text


def is_loopback_host(value):
    host = normalize_host(value)
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def parse_authority(value):
    """Split a ``Host``-style authority into ``(host, port | None)``."""
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid authority")
    parsed = urlsplit("//" + value)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("invalid authority") from None
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid authority")
    return parsed.hostname.lower().rstrip("."), port


@dataclass(frozen=True)
class HTTPSecurityPolicy:
    """Per-server gate configuration; ``None`` fields disable that check."""

    allowed_hosts: frozenset | None = None
    api_key: str | None = None
    check_origin: bool = True

    def __post_init__(self):
        if self.api_key is not None:
            validate_api_key(self.api_key)


def policy_for_bind(bind_host, *, allowed_hosts=(), api_key=None, logger=None):
    """Default gate for a CLI bind address.

    Loopback bind: Host must be a loopback name, the bind value, or an
    ``--allowed-host`` (any port).  Non-loopback bind: the same allowlist when
    ``--allowed-host`` is given; otherwise warn and skip the Host check
    (clients may dial any name for the machine).  ``Origin == Host`` is
    always enforced when an ``Origin`` header is present.
    """
    extra = {normalize_host(host) for host in allowed_hosts}
    bind = str(bind_host or "").strip()
    wildcard = bind.strip("[]") in WILDCARD_BINDS
    if not wildcard and is_loopback_host(bind):
        hosts = frozenset(LOOPBACK_HOSTS | extra | {normalize_host(bind)})
    elif extra:
        hosts = frozenset(
            LOOPBACK_HOSTS | extra | (set() if wildcard else {normalize_host(bind)})
        )
    else:
        hosts = None
        (logger or logging.getLogger("mlx2.server")).warning(
            "non-loopback bind %r without --allowed-host: Host header check is "
            "disabled; pass --allowed-host NAME to defend against DNS rebinding",
            bind,
        )
    return HTTPSecurityPolicy(allowed_hosts=hosts, api_key=api_key)


class GateRejection(Exception):
    """A request the gate refuses; carries the HTTP status and error type."""

    def __init__(self, status, message, error_type, code):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code

    def payload(self, *, anthropic=False):
        if anthropic:
            return {
                "type": "error",
                "error": {"type": self.error_type, "message": self.message},
            }
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        }

    @property
    def headers(self):
        return {"WWW-Authenticate": "Bearer"} if self.status == 401 else None


def check_authority(headers, allowed_hosts, *, local_address=None, check_origin=True):
    """Raise ``GateRejection`` for an untrusted Host or a cross-origin request.

    ``local_address`` (the socket's own address) is always trusted when the
    allowlist is active: a literal-IP Host naming the address the client
    actually reached cannot be a rebinding name.
    """
    hosts = headers.get_all("Host") or []
    origins = headers.get_all("Origin") or []
    if len(hosts) > 1 or len(origins) > 1:
        raise GateRejection(
            400, "ambiguous Host or Origin header", "invalid_request_error",
            "invalid_authority",
        )
    host = port = None
    if hosts:
        try:
            host, port = parse_authority(hosts[0])
        except ValueError:
            raise GateRejection(
                400, "invalid Host header", "invalid_request_error",
                "invalid_authority",
            ) from None
    if allowed_hosts is not None:
        trusted = allowed_hosts
        if local_address:
            trusted = trusted | {str(local_address).split("%", 1)[0].lower()}
        if host is None or host not in trusted:
            raise GateRejection(
                421, "request Host is not served by this server",
                "permission_error", "untrusted_host",
            )
    if check_origin and origins:
        try:
            origin = urlsplit(origins[0])
            if (
                origin.scheme not in ("http", "https")
                or origin.path
                or origin.query
                or origin.fragment
                or host is None
            ):
                raise ValueError
            origin_host, origin_port = parse_authority(origin.netloc)
        except ValueError:
            raise GateRejection(
                403, "request Origin is not allowed", "permission_error",
                "cross_origin",
            ) from None
        default = 443 if origin.scheme == "https" else 80
        if (
            origin_host,
            default if origin_port is None else origin_port,
        ) != (host, default if port is None else port):
            raise GateRejection(
                403, "cross-origin requests are not allowed", "permission_error",
                "cross_origin",
            )


def check_api_key(headers, key):
    """Raise ``GateRejection`` unless a presented credential equals ``key``.

    Accepts ``Authorization: Bearer <key>`` (OpenAI SDKs, Claude Code auth
    token) or ``x-api-key: <key>`` (Anthropic SDK).  Repeated headers are
    rejected as ambiguous.
    """
    if key is None:
        return
    authorization = headers.get_all("Authorization") or []
    api_keys = headers.get_all("x-api-key") or []
    presented = []
    if len(authorization) == 1:
        scheme, separator, value = authorization[0].strip().partition(" ")
        if separator and scheme.lower() == "bearer":
            presented.append(value.strip())
    if len(api_keys) == 1:
        presented.append(api_keys[0].strip())
    ok = len(authorization) <= 1 and len(api_keys) <= 1 and bool(presented)
    expected = key.encode("latin-1")
    matched = False
    for value in presented:
        try:
            candidate = value.encode("latin-1")
        except UnicodeEncodeError:
            candidate = b""
        matched |= hmac.compare_digest(candidate, expected)
    if not (ok and matched):
        raise GateRejection(
            401, "invalid or missing API key", "authentication_error",
            "invalid_api_key",
        )


def authorize(policy, command, path, headers, *, local_address=None,
              admin_token_configured=False):
    """Run the whole gate for one request; raise ``GateRejection`` on refusal.

    ``GET /health`` is exempt from the API key (not from Host/Origin).  Admin
    paths with a configured admin token keep that token as their credential
    (both use ``Authorization: Bearer``); without an admin token they require
    the API key like any other route.
    """
    if policy is None:
        return
    check_authority(
        headers,
        policy.allowed_hosts,
        local_address=local_address,
        check_origin=policy.check_origin,
    )
    if policy.api_key is None:
        return
    if command == "GET" and path in HEALTH_PATHS:
        return
    if admin_token_configured and path.startswith(ADMIN_PREFIX):
        return
    check_api_key(headers, policy.api_key)
